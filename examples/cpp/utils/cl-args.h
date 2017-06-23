/**
 * \file cl-args.h
 * \brief This is a **very** minimal command line argument parser
 */
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

/**
 * Values used to specify the task at hand, and incidentally the required command line arguments
 */
enum Task {
  TRAIN, /**< Self-supervised learning : Only requires train and dev file */
  TRAIN_SUP, /**< Supervised learning : Requires train and dev data as well as labels */
  TEST
};


using namespace std;
/**
 * \brief Structure holding any possible command line argument
 *
 */
struct Params {
  string exp_name = "encdec";
  string train_file = "";
  string clusters_file = "";
  string paths_file = "";
  string dev_file = "";
  string train_labels_file = "";
  string dev_labels_file = "";
  string model_file = "";
  string dic_file = "";
  string test_file = "";
  string test_labels_file = "";
  string nbest_file = "";
  unsigned LAYERS = 1;
  unsigned INPUT_DIM = 2;
  unsigned HIDDEN_DIM = 4;
  unsigned BATCH_SIZE = 1;
  unsigned DEV_BATCH_SIZE = 16;
  unsigned eta_decay_onset_epoch = 0;
  float eta0 = 1.0;
  float eta_decay_rate = 1.0;
  int NUM_EPOCHS = -1;
  float dropout_rate = 0.f;
  bool bidirectionnal = false;
  bool sample = false;
};

/**
 * \brief Get parameters from command line arguments
 * \details Parses parameters from `argv` and check for required fields depending on the task
 * 
 * \param argc Number of arguments
 * \param argv Arguments strings
 * \param params Params structure
 * \param task Task
 */
void get_args(int argc,
              char** argv,
              Params& params,
              Task task) {
  int i = 0;
  while (i < argc) {
    string arg = argv[i];
    if (arg == "--name" || arg == "-n") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.exp_name;
      i++;
    } else if (arg == "--train" || arg == "-t") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.train_file;
      i++;
    } else if (arg == "--clusters" || arg == "-c") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.clusters_file;
      i++;
    } else if (arg == "--paths" || arg == "-p") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.paths_file;
      i++;
    } else if (arg == "--dev" || arg == "-d") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dev_file;
      i++;
    } else if (arg == "--train_labels" || arg == "-tl") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.train_labels_file;
      i++;
    } else if (arg == "--dev_labels" || arg == "-dl") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dev_labels_file;
      i++;
    } else if (arg == "--dict" || arg == "-dic") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dic_file;
      i++;
    } else if (arg == "--test" || arg == "-ts") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.test_file;
      i++;
    } else if (arg == "--nbest" || arg == "-nb") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.nbest_file;
      i++;
    } else if (arg == "--model" || arg == "-m") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.model_file;
      i++;
    } else if (arg == "--num_layers" || arg == "-l") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.LAYERS;
      i++;
    } else if (arg == "--input_size" || arg == "-i") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.INPUT_DIM;
      i++;
    } else if (arg == "--hidden_size" || arg == "-h") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.HIDDEN_DIM;
      i++;
    } else if (arg == "--batch_size" || arg == "-b") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.BATCH_SIZE;
      i++;
    } else if (arg == "--num_epochs" || arg == "-N") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.NUM_EPOCHS;
      i++;
    } else if (arg == "--eta0" || arg == "-e0") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.eta0;
      i++;
    } else if (arg == "--eta_decay_rate" || arg == "-edr") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.eta_decay_rate;
      i++;
    } else if (arg == "--eta_decay_onset_epoch" || arg == "-edoe") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.eta_decay_onset_epoch;
      i++;
    } else if (arg == "--dropout_rate" || arg == "-D") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dropout_rate;
      i++;
    } else  if (arg == "--bidirectionnal" || arg == "-bid") {
      params.bidirectionnal = true;
    } else  if (arg == "--sample" || arg == "-s") {
      params.sample = true;
    }
    i++;
  }
  if (task == TRAIN) {
    if (params.train_file == "" || params.dev_file == "") {
      stringstream ss;
      ss << "Usage: " << argv[0] << " -t [train_file] -d [dev_file]";
      throw invalid_argument(ss.str());
    }
  } else if (task == TRAIN_SUP) {
    if (params.train_file == "" || params.dev_file == "" || params.train_labels_file == "" || params.dev_labels_file == "") {
      stringstream ss;
      ss << "Usage: " << argv[0] << " -t [train_file] -d [dev_file] -tl [train_labels_file] -d [dev_labels_file]";
      throw invalid_argument(ss.str());
    }
  }
}