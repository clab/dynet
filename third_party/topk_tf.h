/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// modified version of heap-based topk from tensorflow
// https://github.com/tensorflow/tensorflow/tree/6bcc00668094be8daf8465b8689bbed5ab285b2d/tensorflow/core/kernels/topk_op_gpu.cu.cc

#ifndef _TOPK_TF
#define _TOPK_TF

#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <assert.h>

// tensorflow's implementation using max/min-heap
// - from tensorflow/tensorflow/core/kernels/topk_op_gpu.cu.cc
namespace impl_tf {

  enum class HeapType { kMinHeap, kMaxHeap };
  enum class PreferIndices { kLower, kHigher };   // prefer which index if equal

  template <typename T, typename IndexType>
  struct Entry {
    IndexType index;
    T value;
  };

  template <typename T, typename IndexType>
  struct LinearData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](IndexType index) const { return data[index]; }
    __device__ IndexType get_index(IndexType i) const { return data[i].index; }
    __device__ T get_value(IndexType i) const { return data[i].value; }
    Entry* const data;
  };

  template <typename T, typename IndexType>
  struct IndirectLinearData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](IndexType index) const { return data[index]; }
    __device__ IndexType get_index(IndexType i) const { return backing_data[data[i].index].index; }
    __device__ T get_value(IndexType i) const { return data[i].value; }
    Entry* const data;
    Entry* const backing_data;
  };

  template <typename T, typename IndexType>
  struct StridedData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    // Here distributing among the different threads (shards)
    __device__ Entry& operator[](IndexType index) const { return data[index * blockDim.x + threadIdx.x]; }
    __device__ IndexType get_index(IndexType i) const { return (*this)[i].index; }
    __device__ T get_value(IndexType i) const { return (*this)[i].value; }
    Entry* const data;
  };


  // A heap of Entry<T> that can either work as a min-heap or as a max-heap.
  template <HeapType heapType, PreferIndices preferIndices, template <typename, typename> class Data, typename T, typename IndexType>
  struct IndexedHeap {
    typedef typename Data<T, IndexType>::Entry Entry;
    const Data<T, IndexType> data;

    // indicating whether left should be prior to right
    __device__ bool is_above(IndexType left, IndexType right) {
      T left_value = data.get_value(left);
      T right_value = data.get_value(right);
      if(left_value == right_value) {
        if(preferIndices == PreferIndices::kLower) {
          return data.get_index(left) < data.get_index(right);
        }
        else {
          return data.get_index(left) > data.get_index(right);
        }
      }
      if(heapType == HeapType::kMinHeap) {
        return left_value < right_value;
      }
      else {
        return left_value > right_value;
      }
    }

    // assign one entry
    __device__ void assign(IndexType i, const Entry& entry) { data[i] = entry; }

    // swap
    __device__ void swap(IndexType a, IndexType b) {
      auto tmp = data[b];
      data[b] = data[a];
      data[a] = tmp;
    }

    // upward from i
    __device__ void push_up(IndexType i) {
      IndexType child = i;
      IndexType parent;
      for(; child > 0; child = parent) {
        parent = (child - 1) / 2;
        if(!is_above(child, parent)) {
          // Heap property satisfied.
          break;
        }
        swap(child, parent);
      }
    }

    __device__ void push_root_down(IndexType k) { push_down(0, k); }

    // MAX-HEAPIFY in Cormen, k is the range
    __device__ void push_down(IndexType node, IndexType k) {
      while(true) {
        const IndexType left = 2 * node + 1;
        const IndexType right = left + 1;
        IndexType smallest = node;
        if(left < k && is_above(left, smallest)) {
          smallest = left;
        }
        if(right < k && is_above(right, smallest)) {
          smallest = right;
        }
        if(smallest == node) {
          break;
        }
        swap(smallest, node);
        node = smallest;
      }
    }

    // BUILD-MAX-HEAPIFY in Cormen
    __device__ void build(IndexType k) {
      for(IndexType node = (k - 1) / 2; node >= 0; node--) {
        push_down(node, k);
      }
    }

    // HEAP-EXTRACT-MAX in Cormen
    __device__ void remove_root(IndexType k) {
      data[0] = data[k - 1];
      push_root_down(k - 1);
    }

    // in-place HEAPSORT in Cormen (turn minHeap to max-sorting)
    // This method destroys the heap property.
    __device__ void sort(IndexType k) {
      for(IndexType slot = k - 1; slot > 0; slot--) {
        // This is like remove_root but we insert the element at the end.
        swap(slot, 0);
        // Heap is now an element smaller.
        push_root_down(/*k=*/slot);
      }
    }

    __device__ void replace_root(const Entry& entry, IndexType k) {
      data[0] = entry;
      push_root_down(k);
    }

    __device__ const Entry& root() { return data[0]; }
  };

  template <HeapType heapType, PreferIndices preferIndices, template <typename, typename> class Data, typename T, typename IndexType>
  __device__ IndexedHeap<heapType, preferIndices, Data, T, IndexType> make_indexed_heap(
    typename Data<T, IndexType>::Entry* data) {
    return IndexedHeap<heapType, preferIndices, Data, T, IndexType>{Data<T, IndexType>{data}};
  }

  // heapTopK walks over [input, input+length) with `step_size` stride starting at
  // `start_index`.
  // It builds a top-`k` heap that is stored in `heap_entries` using `Accessor` to
  // access elements in `heap_entries`. If sorted=true, the elements will be
  // sorted at the end.
  // -- start_index and step_size are at cur_size-level, which is over another level stride of inner_size
  // -- only consider the inner_size (inner_stride) when reading the real value from input
  template <typename T, typename IndexType, template <typename, typename> class Data, bool IsMax>
  __device__ void heapTopK(const T* __restrict__ input, IndexType length, IndexType k, Entry<T, IndexType>* __restrict__ heap_entries,
    bool sorted, IndexType start_index, IndexType step_size, IndexType inner_size) {
    // this should be restricted previously
    //assert(k <= (length-start_index+step_size-1)/step_size);
    // the min value as the threshold
    // -- with kHigher preference means prefer lower-indexed top values
    constexpr auto HeapType = IsMax ? HeapType::kMinHeap : HeapType::kMaxHeap;
    auto heap = make_indexed_heap<HeapType, PreferIndices::kHigher, Data, T, IndexType>(heap_entries);

    IndexType heap_end_index = start_index + k * step_size;
    if(heap_end_index > length) {
      heap_end_index = length;
    }
    // Initialize the min-heap with the first k ones.
    for(IndexType index = start_index, slot = 0; index < heap_end_index; index += step_size, slot++) {
      heap.assign(slot, {index, input[index*inner_size]});
    }
    heap.build(k);
    // Now iterate over the remaining items.
    // If an item is smaller than the min element, it is not amongst the top k.
    // Otherwise, replace the min element with it and push upwards.
    for(IndexType index = heap_end_index; index < length; index += step_size) {
      // We prefer elements with lower indices. This is given here.
      // Later elements automatically have higher indices, so can be discarded.
      auto this_value = input[index*inner_size];
      bool to_replace;
      if(IsMax)   // if kmax, ignore values that are smaller than the smallest value in minHeap
        to_replace = (this_value > heap.root().value);
      else
        to_replace = (this_value < heap.root().value);
      if(to_replace) {
        // This element should replace the min.
        heap.replace_root({index, this_value}, k);
      }
    }
    // Sort if wanted.
    if(sorted) {
      heap.sort(k);
    }
  }

  // mergeShards performs a top-k merge on `num_shards` many sorted streams that
  // are sorted and stored in `entries` in a strided way:
  // |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
  // The overall top k elements are written to `top_k_values` and their indices
  // to top_k_indices.
  // `top_k_heap` is used as temporary storage for the merge heap.
  template <typename T, typename IndexType, bool IsMax>
  __device__ void mergeShards(IndexType num_shards, IndexType k,
    Entry<T, IndexType>* __restrict__ entries, Entry<T, IndexType>* __restrict__ top_k_heap,
    T* top_k_values, IndexType* top_k_indices, IndexType inner_size) {
    // If k < num_shards, we can use a min-heap with k elements to get the top k
    // of the sorted blocks.
    // If k > num_shards, we can initialize a min-heap with the top element from
    // each sorted block.
    const IndexType heap_size = k < num_shards ? k : num_shards;

    // TODO: the heaps could have garbage if too many shards or too few length
    // Min-heap part.
    {
      constexpr auto HeapType = IsMax ? HeapType::kMinHeap : HeapType::kMaxHeap;
      auto min_heap = IndexedHeap<HeapType, PreferIndices::kHigher,
        IndirectLinearData, T, IndexType>{IndirectLinearData<T, IndexType>{top_k_heap, entries}};
      // Initialize the heap as a min-heap.
      for(IndexType slot = 0; slot < heap_size; slot++) {
        min_heap.assign(slot, {slot, entries[slot].value});
      }
      min_heap.build(heap_size);

      // Now perform top k with the remaining shards (if num_shards > heap_size).
      for(IndexType shard = heap_size; shard < num_shards; shard++) {
        const auto entry = entries[shard];
        const auto root = min_heap.root();
        //
        if(IsMax){
          if(entry.value < root.value) continue;
        }
        else{
          if(entry.value > root.value) continue;
        }
        // prefer lower index
        if(entry.value == root.value &&
          entry.index > entries[root.index].index) {
          continue;
        }
        // This element should replace the min.
        min_heap.replace_root({shard, entry.value}, heap_size);
      }
    }

    // Max-part.
    {
      // Turn the min-heap into a max-heap in-place.
      constexpr auto HeapType = (!IsMax) ? HeapType::kMinHeap : HeapType::kMaxHeap;
      auto max_heap = IndexedHeap<HeapType, PreferIndices::kLower,
        IndirectLinearData, T, IndexType>{IndirectLinearData<T, IndexType>{top_k_heap, entries}};

      // Heapify into a max heap.
      max_heap.build(heap_size);

      // Now extract the minimum k-1 times.
      // k is treated specially.
      const IndexType last_k = k - 1;
      for(IndexType rank = 0; rank < last_k; rank++) {
        const Entry<T, IndexType>& max_element = max_heap.root();
        top_k_values[rank*inner_size] = max_element.value;
        IndexType shard_index = max_element.index;
        top_k_indices[rank*inner_size] = entries[shard_index].index;
        IndexType next_shard_index = shard_index + num_shards;
        // For rank < k-1, each top k heap still contains at least 1 element,
        // so we can draw a replacement.
        max_heap.replace_root({next_shard_index, entries[next_shard_index].value}, heap_size);
      }

      // rank == last_k.
      const Entry<T, IndexType>& max_element = max_heap.root();
      top_k_values[last_k*inner_size] = max_element.value;
      IndexType shard_index = max_element.index;
      top_k_indices[last_k*inner_size] = entries[shard_index].index;
    }
  }

  extern __shared__ char shared_memory[];

  template <typename T, typename IndexType, bool IsMax>
  __global__ void TopKKernel(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k) {
    // 
    IndexType slice = blockIdx.x;
    if(slice >= outer_size*inner_size) {
      return;
    }
    // prepare data
    T* input_start = input + slice/inner_size*inner_size*cur_size + slice%inner_size;
    T* output_val_start = ouput_val + slice/inner_size*inner_size*k + slice%inner_size;
    IndexType* ouput_idx_start = ouput_idx + slice/inner_size*inner_size*k + slice%inner_size;

    // heap-select with strided elements
    const IndexType thread_index = threadIdx.x;   // index of the shards
    const IndexType thread_count = blockDim.x;    // how many shards
    Entry<T, IndexType>* shared_entries = (Entry<T, IndexType>*)shared_memory;
    heapTopK<T, IndexType, StridedData, IsMax>(input_start, cur_size, k, shared_entries, true, thread_index, thread_count, inner_size);
    __syncthreads();

    // merge
    if(thread_index == 0) {
      // TODO(blackhc): Erich says: Performance can likely be improved
      // significantly by having the merge be done by multiple threads rather than
      // just one.  ModernGPU has some nice primitives that could help with this.
      Entry<T, IndexType>* top_k_heap = shared_entries + thread_count * k;
      mergeShards<T, IndexType, IsMax>(thread_count, k, shared_entries, top_k_heap, output_val_start, ouput_idx_start, inner_size);
    }
  }

  template<typename T, typename IndexType>
  IndexType get_max_num_shards(IndexType k, IndexType length){
    // This code assumes that k is small enough that the computation
    // fits inside shared memory (hard coded to 48KB).
    constexpr IndexType shared_memory_size = 48 << 10;  // 48 KB
    const IndexType heap_size = k * sizeof(Entry<T, IndexType>);
    // - shared_memory_size = (num_shards + 1) * heap_size <=>
    IndexType max_num_shards = shared_memory_size / heap_size - 1;
    if(max_num_shards <= 0){
      // too large k for this implementation
      IndexType maxk = shared_memory_size / (2 * heap_size);
      std::ostringstream ss;
      ss << "Too large k: " << k << " for heap-based topk, with 48K shared gpu memory the max-k is " << maxk;
      std::string err_info = ss.str();
      throw std::runtime_error(err_info);
    }
    //
    IndexType upper_limit = max_num_shards;                 // memory restriction
    upper_limit = std::min(upper_limit, IndexType(1024));   // hard restriction
    upper_limit = std::min(upper_limit, length / k);        // no need to be too large
    return upper_limit;
  }

  template<typename T, typename IndexType>
  IndexType get_auto_num_shards(IndexType k, IndexType length){
    // TODO: simple & kind-of casual rules by observing some of the results
    IndexType max_num_shards = get_max_num_shards<T, IndexType>(k, length);
    if(max_num_shards <= 8)
      return max_num_shards;
    else if(max_num_shards <= 64)
      return std::max((IndexType)8, max_num_shards/2);
    else{
      IndexType down = max_num_shards / 2;
      if(max_num_shards > 256)
        down = std::max((IndexType)128, down / 2);
      // the 32-multi that is close to down
      IndexType left = down / 32 * 32;
      IndexType right = left + 32;
      if((down - left) <= (right - down))
        return left;
      else
        return right;
    }
  }

  // input(outer_size*cur_size*inner_size) -> output(outer_size*k*inner_size)
  // num_shards: how many pieces to split
  template<typename T, typename IndexType, bool IsMax>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k,
    IndexType num_shards=0){
    IndexType grid = outer_size*inner_size;
    if(num_shards <= 0) // auto setting
      num_shards = get_auto_num_shards<T, IndexType>(k, cur_size);
    else{
      IndexType max_num_shards = get_max_num_shards<T, IndexType>(k, cur_size);
      num_shards = std::max(max_num_shards, num_shards);
    }
    //
    auto shared_memory_size = (num_shards + 1) * k * sizeof(Entry<T, IndexType>);   //?? possible different sizes of host/device
    TopKKernel<T, IndexType, IsMax>
      <<<grid, num_shards, shared_memory_size>>>(input, ouput_val, ouput_idx, outer_size, inner_size, cur_size, k);
    return cudaGetLastError();
  }
};

#endif

