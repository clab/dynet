#include "mp.h"
using namespace std;
using namespace boost::interprocess;

namespace cnn {
  namespace mp {
    // TODO: Pass these around instead of having them be global
    std::string queue_name = "cnn_mp_work_queue";
    std::string shared_memory_name = "cnn_mp_shared_memory";
    timespec start_time;
    bool stop_requested = false;
    SharedObject* shared_object = nullptr;

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

    cnn::real SumValues(const std::vector<cnn::real>& values) {
      return accumulate(values.begin(), values.end(), 0.0);
    }

    cnn::real Mean(const std::vector<cnn::real>& values) {
      return SumValues(values) / values.size();
    }

    std::string ElapsedTimeString(const timespec& start, const timespec& end) {
      std::ostringstream ss;
      time_t secs = end.tv_sec - start.tv_sec;
      long nsec = end.tv_nsec - start.tv_nsec;
      ss << secs << " seconds and " << nsec << "nseconds";
      return ss.str();
    }

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
      int err;
      std::vector<Workload> workloads(num_children);
      for (unsigned cid = 0; cid < num_children; cid++) { 
        err = pipe(workloads[cid].p2c);
        assert (err == 0);
        err = pipe(workloads[cid].c2p);
        assert (err == 0);
      }
      return workloads;
    }

  }
}
