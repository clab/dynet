/*
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
and IDIAP Research Institute nor the names of its contributors may be
used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/


// pytroch's implementation using radix-selection
// - from pytorch/aten/src/THC/THCTensorTopK.cuh
// https://github.com/pytorch/pytorch/blob/d792c21f72b6bcace80face2a4157528d977bfa9/aten/src/THC/THCTensorTopK.cuh
// - warning: ignore some types, for eg., does not include THCNumerics for T==half

#ifndef _TOPK_TR
#define _TOPK_TR

#include <assert.h>

namespace impl_tr{
  // 
  template <typename T>
  struct TopKTypeConfig {};

  template <>
  struct TopKTypeConfig<float> {
    typedef uint32_t RadixType;
    // Converts a float to an integer representation with the same
    // sorting; i.e., for floats f1, f2:
    // if f1 < f2 then convert(f1) < convert(f2)
    // We use this to enable radix selection of floating-point values.
    // This also gives a relative order for NaNs, but that's ok, as they
    // will all be adjacent
    static inline __device__ RadixType convert(float v) {
      RadixType x = __float_as_int(v);
      RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
      return (x ^ mask);
    }
    static inline __device__ float deconvert(RadixType v) {
      RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
      return __int_as_float(v ^ mask);
    }
  };

  template <>
  struct TopKTypeConfig<double> {
    typedef uint64_t RadixType;

    static inline __device__ RadixType convert(double v) {
      RadixType x = __double_as_longlong(v);
      RadixType mask = -((x >> 63)) | 0x8000000000000000;
      return (x ^ mask);
    }

    static inline __device__ double deconvert(RadixType v) {
      RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
      return __longlong_as_double(v ^ mask);
    }
  };

  // helpers

  // For CC 3.5+, perform a load using __ldg
  template <typename T>
  __device__ __forceinline__ T doLdg(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
  }

  // Collection of direct PTX functions
  template <typename T>
  struct Bitfield {};
  template <>
  struct Bitfield<unsigned int> {
    static __device__ __forceinline__
      unsigned int getBitfield(unsigned int val, int pos, int len) {
      unsigned int ret;
      asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
      return ret;
    }
    static __device__ __forceinline__
      unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
      unsigned int ret;
      asm("bfi.b32 %0, %1, %2, %3, %4;" :
      "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
      return ret;
    }
  };

  template <>
  struct Bitfield<uint64_t> {
    static __device__ __forceinline__
      uint64_t getBitfield(uint64_t val, int pos, int len) {
      uint64_t ret;
      asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
      return ret;
    }

    static __device__ __forceinline__
      uint64_t setBitfield(uint64_t val, uint64_t toInsert, int pos, int len) {
      uint64_t ret;
      asm("bfi.b64 %0, %1, %2, %3, %4;" :
      "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
      return ret;
    }
  };

  // WARP_BALLOT
  __device__ __forceinline__ int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
  {
#if CUDA_VERSION >= 9000
    return __ballot_sync(mask, predicate);
#else
    return __ballot(predicate);
#endif
  }

  // ACTIVE_MASK
  __device__ __forceinline__ unsigned int ACTIVE_MASK()
  {
#if CUDA_VERSION >= 9000
    return __activemask();
#else
    // will be ignored anyway
    return 0xffffffff;
#endif
  }

  // getLaneId
  __device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.s32 %0, %laneid;" : "=r"(laneId));
    return laneId;
  }

  /**
  Computes ceil(a / b)
  */
  template <typename T>
  __host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
    return (a + b - 1) / b;
  }

  /**
  Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
  multiple of b
  */
  template <typename T>
  __host__ __device__ __forceinline__ T THCRoundUp(T a, T b) {
    return THCCeilDiv(a, b) * b;
  }


  // This function counts the distribution of all input values in a
  // slice we are selecting by radix digit at `radixDigitPos`, but only
  // those that pass the filter `((v & desiredMask) == desired)`.
  // This produces and broadcasts the seen counts for a single block only.
  // `smem` must have at least `RadixSize` elements.
  template <typename DataType, typename BitDataType,
    typename IndexType, typename CountType,
    int RadixSize, int RadixBits>
    __device__ void countRadixUsingMask(CountType counts[RadixSize],
      CountType* smem,
      BitDataType desired,
      BitDataType desiredMask,
      int radixDigitPos,
      IndexType sliceSize,
      IndexType withinSliceStride,
      DataType* data) {
    // Clear out per-thread counts from a previous round
#pragma unroll
    for(int i = 0; i < RadixSize; ++i) {
      counts[i] = 0;
    }

    if(threadIdx.x < RadixSize) {
      smem[threadIdx.x] = 0;
    }
    __syncthreads();

    // Scan over all the data. Upon a read, the warp will accumulate
    // counts per each digit in the radix using warp voting.
    for(IndexType i = threadIdx.x; i < sliceSize; i += blockDim.x) {
      BitDataType val = TopKTypeConfig<DataType>::convert(doLdg(&data[i * withinSliceStride]));

      bool hasVal = ((val & desiredMask) == desired);
      BitDataType digitInRadix = Bitfield<BitDataType>::getBitfield(val, radixDigitPos, RadixBits);

#pragma unroll
      for(unsigned int j = 0; j < RadixSize; ++j) {
        bool vote = hasVal && (digitInRadix == j);
        counts[j] += __popc(WARP_BALLOT(vote, ACTIVE_MASK()));
      }
    }

    // Now, for each warp, sum values
    if(getLaneId() == 0) {
#pragma unroll
      for(unsigned int i = 0; i < RadixSize; ++i) {
        atomicAdd(&smem[i], counts[i]);
      }
    }

    __syncthreads();

    // For each thread, read in the total counts
#pragma unroll
    for(unsigned int i = 0; i < RadixSize; ++i) {
      counts[i] = smem[i];
    }

    __syncthreads();
  }

  // Over what radix we are selecting values
#define RADIX_BITS 2 // digits are base-(2 ^ RADIX_BITS)
#define RADIX_SIZE 4 // 2 ^ RADIX_BITS
#define RADIX_MASK (RADIX_SIZE - 1)

  // 
  // This finds the unique value `v` that matches the pattern
  // ((v & desired) == desiredMask) in our sorted int format
  template <typename DataType, typename BitDataType, typename IndexType>
  __device__ DataType findPattern(DataType* smem,
    DataType* data,
    IndexType sliceSize,
    IndexType withinSliceStride,
    BitDataType desired,
    BitDataType desiredMask) {
    if(threadIdx.x < 32) {
      smem[threadIdx.x] = (DataType)(0);
    }
    __syncthreads();

    // All threads participate in the loop, in order to sync on the flag
    IndexType numIterations = THCRoundUp(sliceSize, (IndexType)blockDim.x);
    for(IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
      bool inRange = (i < sliceSize);
      DataType v = inRange ? doLdg(&data[i * withinSliceStride]) : (DataType)(0);

      if(inRange && ((TopKTypeConfig<DataType>::convert(v) & desiredMask) == desired)) {
        // There should not be conflicts if we are using findPattern,
        // since the result is unique
        smem[0] = (DataType)(1);
        smem[1] = v; // can't use val as the flag, since it could be 0
      }

      __syncthreads();

      DataType found = smem[0];
      DataType val = smem[1];

      __syncthreads();

      // Check to see if a thread found the value
      if(found != (DataType)(0)) {
        // all threads return this value
        return val;
      }
    }

    // should not get here
    assert(false);
    return (DataType)(0);
  }

  // Returns the top-Kth element found in the data using radix selection
  // - find k out of sliceSize with stride of withinSliceStride, Descending if Order
  template <typename DataType, typename BitDataType, typename IndexType, bool Order>
  __device__ void radixSelect(DataType* data,
    IndexType k,
    IndexType sliceSize,
    IndexType withinSliceStride,
    int* smem,
    DataType* topK) {
    // Per-thread buckets into which we accumulate digit counts in our
    // radix
    int counts[RADIX_SIZE];
    // We only consider elements x such that (x & desiredMask) == desired
    // Initially, we consider all elements of the array, so the above
    // statement is true regardless of input.
    BitDataType desired = 0;
    BitDataType desiredMask = 0;
    // We are looking for the top kToFind-th element when iterating over
    // digits; this count gets reduced by elimination when counting
    // successive digits
    int kToFind = k;
    // We start at the most significant digit in our radix, scanning
    // through to the least significant digit
#pragma unroll
    for(int digitPos = sizeof(DataType) * 8 - RADIX_BITS;
      digitPos >= 0;
      digitPos -= RADIX_BITS) {
      // Count radix distribution for the current position and reduce
      // across all threads
      countRadixUsingMask<DataType, BitDataType,
        IndexType, int,
        RADIX_SIZE, RADIX_BITS>(
          counts, smem,
          desired, desiredMask, digitPos,
          sliceSize, withinSliceStride, data);
      // All threads participate in the comparisons below to know the
      // final result
#define CHECK_RADIX(i)                                                  \
    int count = counts[i];                                              \
                                                                        \
    /* All threads have the same value in counts here, so all */        \
    /* threads will return from the function. */                        \
    if (count == 1 && kToFind == 1) {                                   \
      /* There is a unique answer. */                                   \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The answer is now the unique element v such that: */           \
      /* (v & desiredMask) == desired */                                \
      /* However, we do not yet know what the actual element is. We */  \
      /* need to perform a search through the data to find the */       \
      /* element that matches this pattern. */                          \
      *topK = findPattern<DataType, BitDataType, IndexType>(                         \
        (DataType*) smem, data, sliceSize,                              \
        withinSliceStride, desired, desiredMask);                       \
      return;                                                           \
    }                                                                   \
                                                                        \
    if (count >= kToFind) {                                             \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The top-Kth element v must now be one such that: */            \
      /* (v & desiredMask == desired) */                                \
      /* but we haven't narrowed it down; we must check the next */     \
      /* least-significant digit */                                     \
      break;                                                            \
    }                                                                   \
                                                                        \
    kToFind -= count                                                    \

      if(Order) {
        // Process in descending order
#pragma unroll
        for(int i = RADIX_SIZE - 1; i >= 0; --i) {
          CHECK_RADIX(i);
        }
      }
      else {
        // Process in ascending order
#pragma unroll
        for(int i = 0; i < RADIX_SIZE; ++i) {
          CHECK_RADIX(i);
        }
      }
#undef CHECK_RADIX
    } // end digitPos for

      // There is no unique result, but there is a non-unique result
      // matching `desired` exactly
    *topK = TopKTypeConfig<DataType>::deconvert(desired);
  }

  // helpers2
  template <typename T>
  struct AddOp {
    __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
      return lhs + rhs;
    }
  };

  __device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
  }

  // Inclusive prefix sum for binary vars using intra-warp voting +
  // shared memory
  template <typename T, bool KillWARDependency, class BinaryFunction>
  __device__ void inclusiveBinaryPrefixScan(T* smem, bool in, T* out, BinaryFunction binop) {
    // Within-warp, we use warp voting.
    T vote = WARP_BALLOT(in);
    T index = __popc(getLaneMaskLe() & vote);
    T carry = __popc(vote);

    int warp = threadIdx.x / 32;

    // Per each warp, write out a value
    if(getLaneId() == 0) {
      smem[warp] = carry;
    }

    __syncthreads();

    // Sum across warps in one thread. This appears to be faster than a
    // warp shuffle scan for CC 3.0+
    if(threadIdx.x == 0) {
      int current = 0;
      for(int i = 0; i < blockDim.x / 32; ++i) {
        T v = smem[i];
        smem[i] = binop(smem[i], current);
        current = binop(current, v);
      }
    }

    __syncthreads();

    // load the carry from the preceding warp
    if(warp >= 1) {
      index = binop(index, smem[warp - 1]);
    }

    *out = index;

    if(KillWARDependency) {
      __syncthreads();
    }
  }

  // Exclusive prefix sum for binary vars using intra-warp voting +
  // shared memory
  template <typename T, bool KillWARDependency, class BinaryFunction>
  __device__ void exclusiveBinaryPrefixScan(T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
    inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

    // Inclusive to exclusive
    *out -= (T)in;

    // The outgoing carry for all threads is the last warp's sum
    *carry = smem[(blockDim.x / 32) - 1];

    if(KillWARDependency) {
      __syncthreads();
    }
  }

  // the kernal function
  template<typename T, typename IndexType, bool Order>
  __global__ void topk_kernel(T* input, T* out_val, IndexType* out_idx,
    IndexType cur_size, IndexType k, IndexType outer_size, IndexType inner_size){
    // Indices are limited to integer fp precision, so counts can fit in
    // int32, regardless of IndexType
    __shared__ int smem[32]; // one per each warp, up to warp limit
                             // in range, kind of like in the batch-size range (if flattened, that will be)
    IndexType slice = blockIdx.x;
    if(slice >= outer_size*inner_size) {
      return;
    }
    // prepare data
    T* inputSliceStart = input + slice/inner_size*inner_size*cur_size + slice%inner_size;
    T* topKSliceStart = out_val + slice/inner_size*inner_size*k + slice%inner_size;
    IndexType* indicesSliceStart = out_idx + slice/inner_size*inner_size*k + slice%inner_size;
    // Find the k-th highest element in our input
    T topKValue = (T)(0);
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>(
      inputSliceStart, k, cur_size, inner_size, smem, &topKValue);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive prefix sum of
    // `hasTopK`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the prefix sum,
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    IndexType numIterations = THCRoundUp(cur_size, (IndexType)blockDim.x);
    IndexType writeIndexStart = 0;
    for(IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
      bool inRange = (i < cur_size);
      T v = inRange ? doLdg(&inputSliceStart[i * inner_size]) : (T)(0);
      bool hasTopK;
      if(Order) {   // Process in descending order
        hasTopK = inRange && (v > topKValue);
      }
      else {        // Process in ascending order
        hasTopK = inRange && (v < topKValue);
      }
      int index;
      int carry;
      exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());
      if(hasTopK) {
        int writeIndex = writeIndexStart + index;
        assert(writeIndex < k);

        IndexType topKOffset = writeIndex * inner_size;
        IndexType indexOffset = writeIndex * inner_size;

        topKSliceStart[topKOffset] = v;
        indicesSliceStart[indexOffset] = i + 0;
      }
      writeIndexStart += carry;
    }

    // We need to fill in the rest with actual == top-K values.
    // The number that we need is outputSliceSize -
    // writeIndexStart. There might be more than that number available,
    // in which case we have to choose the first seen set. We do this
    // via a prefix sum to calculate indices for writing results.
    assert(k >= writeIndexStart);
    IndexType topKRemaining = (k - writeIndexStart);
    for(IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
      bool inRange = (i < cur_size);
      T v = inRange ? doLdg(&inputSliceStart[i * inner_size]) : (T)(0);
      bool hasTopK = inRange && (v == topKValue);
      int index;
      int carry;
      exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());
      if(hasTopK && index < topKRemaining) {
        int writeIndex = writeIndexStart + index;
        assert(writeIndex < k);

        IndexType topKOffset = writeIndex * inner_size;
        IndexType indexOffset = writeIndex * inner_size;

        topKSliceStart[topKOffset] = v;
        indicesSliceStart[indexOffset] = i + 0;
      }

      if(carry >= topKRemaining) {
        break;
      }
      topKRemaining -= carry;
      writeIndexStart += carry;
    }
    return;
  }

#undef RADIX_BITS
#undef RADIX_SIZE
#undef RADIX_MASK

  // final call for the topk (no sorting, only selecting topk)
  // -- (batch_size,cur_size,inner_size) -> (batch_size,k,inner_size)
  template<typename T, typename IndexType, bool IsMax>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k){
    // todo: check that inputs are valid
    //
    IndexType grid = outer_size*inner_size;
    IndexType block = (std::min(THCRoundUp(cur_size, (IndexType)(32)), (IndexType)(1024)));
    topk_kernel<T, IndexType, IsMax>
      <<<grid, block>>>(input, ouput_val, ouput_idx, cur_size, k, outer_size, inner_size);
    return cudaGetLastError();
  }
};

#endif

