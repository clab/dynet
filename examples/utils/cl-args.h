#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

enum Task {TRAIN, ANALYSIS};

struct Params {
    string exp_name = "encdec";
    string train_file = "";
    string dev_file = "";
    string model_file = "";
    string dic_file = "";
    string test_file = "";
    unsigned LAYERS = 1;
    unsigned INPUT_DIM = 2;
    unsigned HIDDEN_DIM = 4;
    unsigned BATCH_SIZE = 1;
    unsigned DEV_BATCH_SIZE = 16;
    int NUM_EPOCHS = -1;
    bool bidirectionnal = false;
    bool cust_l2 = false;
    bool dist_penalty = false;
};

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
        } else if (arg == "--dev" || arg == "-d") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.dev_file;
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
        } else  if (arg == "--bidirectionnal" || arg == "-bid") {
            params.bidirectionnal = true;
        } else  if (arg == "--cust_l2" || arg == "-cl2") {
            params.cust_l2 = true;
        } else if (arg == "--dist_penalty" || arg == "-dp") {
            params.dist_penalty = true;
        }
        i++;
    }
    if (task == TRAIN) {
        if (params.train_file == "" || params.dev_file == "") {
            std::cerr << "Usage: " << argv[0] << " -t train.txt -d dev.txt\n";
            abort();
        }
    } else if (task == ANALYSIS) {
        if (params.dic_file == "" || params.test_file == "" || params.model_file == "") {
            std::cerr << "Usage: " << argv[0] << " -dic corpus_dic.txt -ts test.txt -m model.params\n";
            abort();
        }
    }
}
