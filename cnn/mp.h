#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include "cnn/dict.h"
#include "cnn/lstm.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <sstream>
#include <random>

namespace cnn {
  namespace mp {
    // TODO: Pass these around instead of having them be global
    std::string queue_name = "cnn_mp_work_queue";
    std::string shared_memory_name = "cnn_mp_shared_memory"; 

    template <class R>
    struct SharedObject {
      SharedObject() : reporter(), mutex(1) {}
      R reporter;
      boost::interprocess::interprocess_semaphore mutex;
    };

    /// XXX: We never delete these objects
    template <class R>
    SharedObject<R>* GetSharedMemory() {
      auto shm = new boost::interprocess::shared_memory_object(boost::interprocess::create_only, shared_memory_name.c_str(), boost::interprocess::read_write);
      shm->truncate(sizeof(SharedObject<R>));
      auto region = new boost::interprocess::mapped_region (*shm, boost::interprocess::read_write);
      void* addr = region->get_address();
      SharedObject<R>* obj = new (addr) SharedObject<R>();
      return obj;
    }

    // Some simple functions that do IO to/from pipes.
    // These are used to send data from child processes
    // to the parent process or vice/versa.
    cnn::real ReadReal(int pipe) {
      cnn::real v;
      read(pipe, &v, sizeof(cnn::real));
      return v;
    }

    void WriteReal(int pipe, cnn::real v) {
      write(pipe, &v, sizeof(cnn::real));
    }

    template <typename T>
    void WriteIntVector(int pipe, const std::vector<T>& vec) {
      unsigned length = vec.size();
      write(pipe, &length, sizeof(unsigned));
      for (T v : vec) {
        write(pipe, &v, sizeof(T));
      }
    }

    template<typename T>
   std::vector<T> ReadIntVector(int pipe) {
      unsigned length;
      read(pipe, &length, sizeof(unsigned));
      std::vector<T> vec(length);
      for (unsigned i = 0; i < length; ++i) {
        read(pipe, &vec[i], sizeof(T));
      }
      return vec;
    }

    cnn::real SumValues(const std::vector<cnn::real>& values) {
      return accumulate(values.begin(), values.end(), 0.0);
    }

    cnn::real Mean(const std::vector<cnn::real>& values) {
      return SumValues(values) / values.size();
    }

    // A simple struct to hold information about a child process
    // TODO: Rename me!
    struct Workload {
      pid_t pid;
      int c2p[2]; // Child to parent pipe
      int p2c[2]; // Parent to child pipe
    };

    unsigned SpawnChildren(std::vector<Workload>& workloads) {
      const unsigned num_children = workloads.size();
      assert (workloads.size() == num_children);
      pid_t pid;
      unsigned cid;
      for (cid = 0; cid < num_children; ++cid) {
        pid = fork();
        if (pid == -1) {
          std::cerr << "Fork failed. Exiting ..." << std::endl;
          return 1;
        }
        else if (pid == 0) {
          // children shouldn't continue looping
          break;
        }
        workloads[cid].pid = pid;
      }
      return cid;
    }

    std::vector<Workload> CreateWorkloads(unsigned num_children) {
      std::vector<Workload> workloads(num_children);
      for (unsigned cid = 0; cid < num_children; cid++) {
        pipe(workloads[cid].p2c);
        pipe(workloads[cid].c2p);
      }
      return workloads;
    }

    // This interface is used by the child processes and called
    // once per datum.
    template<class D>
    class ILearner {
      public:
        virtual ~ILearner() {}
        virtual cnn::real LearnFromDatum(const D& datum) = 0;
    };

    class IStatusReporter {
    public:
      virtual void Update(unsigned i, cnn::real loss) = 0;
    };

    template<class D>
    void RunParent(const std::vector<D>& data, const std::vector<D>& dev_data,
       std::vector<Workload>& workloads, unsigned num_iterations) {
      const unsigned num_children = workloads.size();
      boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned));
      std::vector<unsigned> indices(data.size());
      for (unsigned i = 0; i < data.size(); ++i) {
        indices[i] = i;
      }

      for (unsigned iter = 0; iter < num_iterations; ++iter) {
        // Shuffle the data indices
        random_shuffle(indices.begin(), indices.end());

        // Tell all the children to start up
        for (unsigned cid = 0; cid < num_children; ++cid) {
          bool cont = true;
          write(workloads[cid].p2c[1], &cont, sizeof(bool));
        }

        // Write all the indices to the queue for the children to process
        for (unsigned i : indices) {
          mq.send(&i, sizeof(i), 0);
        }

        // Send a bunch of stop messages to the children
        for (unsigned cid = 0; cid < num_children; ++cid) {
          unsigned stop = -1U;
          mq.send(&stop, sizeof(stop), 0);
        }

        // Wait for each child to finish training its load
        std::vector<cnn::real> losses(num_children);
        for(unsigned cid = 0; cid < num_children; ++cid) {
          losses[cid] = ReadReal(workloads[cid].c2p[0]);
        }

        cnn::real loss = SumValues(losses) / data.size();
        std::cerr << iter << "\t" << "loss = " << loss << std::endl; 
      }

      // Kill all children one by one and wait for them to exit
      for (unsigned cid = 0; cid < num_children; ++cid) { 
        bool cont = false;
        write(workloads[cid].p2c[1], &cont, sizeof(bool));
        wait(NULL);
      }
    }

    template <class D, class R>
    int RunChild(unsigned cid, ILearner<D>* learner, Trainer* trainer,
        std::vector<Workload>& workloads, const std::vector<D>& data, SharedObject<R>* shared_memory) {
      const unsigned num_children = workloads.size();
      assert (cid >= 0 && cid < num_children); 
      unsigned i;
      unsigned priority;
      boost::interprocess::message_queue::size_type recvd_size;
      boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned)); 
      while (true) {
        // Check if the parent wants us to exit
        bool cont = false;
        read(workloads[cid].p2c[0], &cont, sizeof(bool));
        if (!cont) {
          break;
        }

        // Run the actual training loop
        cnn::real loss = 0;
        while (true) {
          mq.receive(&i, sizeof(i), recvd_size, priority);
          if (i == -1U) {
            break;
          }
          assert (i < data.size());
          const D& datum = data[i];
          cnn::real datum_loss = learner->LearnFromDatum(datum);
          loss += datum_loss;
          trainer->update(1.0);

          shared_memory->mutex.wait();
          shared_memory->reporter.Update(i, datum_loss);
          shared_memory->mutex.post();
        }
        trainer->update_epoch();

        // Let the parent know that we're done and return the loss value
        WriteReal(workloads[cid].c2p[1], loss);
      }
      return 0;
    }

    std::string GenerateQueueName() {
      std::ostringstream ss;
      ss << "cnn_mp_work_queue";
      ss << rand();
      return ss.str();
    }

    std::string GenerateSharedMemoryName() {
      std::ostringstream ss;
      ss << "cnn_mp_shared_memory";
      ss << rand();
      return ss.str();
    }

    template<class D, class R>
    void RunMultiProcess(unsigned num_children, ILearner<D>* learner, Trainer* trainer, const std::vector<D>& train_data,
        const std::vector<D>& dev_data, unsigned num_iterations) {
      queue_name = GenerateQueueName();
      shared_memory_name = GenerateSharedMemoryName();

      struct shm_remove
      {
        shm_remove() { boost::interprocess::shared_memory_object::remove(queue_name.c_str()); }
        ~shm_remove(){ boost::interprocess::shared_memory_object::remove(queue_name.c_str()); }
      } remover;

      SharedObject<R>* shared_memory = GetSharedMemory<R>();

      std::vector<Workload> workloads = CreateWorkloads(num_children);
      unsigned cid = SpawnChildren(workloads);
      if (cid < num_children) {
        RunChild(cid, learner, trainer, workloads, train_data, shared_memory);
      }
      else {
        RunParent(train_data, dev_data, workloads, num_iterations);
      }
    }
  }
};
