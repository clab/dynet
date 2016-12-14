/**
 * Train a multilayer perceptron to classify mnist digits
 *
 * This provide an example of usage of the mlp.h model
 */
#include "mlp.h"
#include "../utils/getpid.h"
#include "../utils/cl-args.h"
#include "../utils/data-io.h"


using namespace std;
using namespace dynet;
using namespace dynet::expr;

int main(int argc, char** argv) {
  // Fetch dynet params ----------------------------------------------------------------------------
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);
  // Fetch program specific parameters (see ../utils/cl-args.h) ------------------------------------
  Params params;

  get_args(argc, argv, params, TRAIN_SUP);

  // Load Dataset ----------------------------------------------------------------------------------
  // Load data
  vector<vector<float>> mnist_train, mnist_dev;

  read_mnist(params.train_file, mnist_train);
  read_mnist(params.dev_file, mnist_dev);

  // Load labels
  vector<unsigned> mnist_train_labels, mnist_dev_labels;

  read_mnist_labels(params.train_labels_file, mnist_train_labels);
  read_mnist_labels(params.dev_labels_file, mnist_dev_labels);

  // Model name (for saving) -----------------------------------------------------------------------
  ostringstream os;
  // Store a bunch of information in the model name
  os << params.exp_name
     << "_" << "mlp"
     << "_" << 784 << "-" << 512 << "-relu-" << 0.2
     << "_" << 512 << "-" << 512 << "-relu-" << 0.2
     << "_" << 512 << "-" << 10 << "-softmax"
     << "_" << getpid()
     << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  // Build model -----------------------------------------------------------------------------------

  Model model;
  // Use Adam optimizer
  AdamTrainer adam(model);
  adam.clip_threshold *= params.BATCH_SIZE;

  // Create model
  MLP nn(model, vector<Layer>({
    Layer(/* input_dim */ 784, /* output_dim */ 512, /* activation */ RELU, /* dropout_rate */ 0.2),
    Layer(/* input_dim */ 512, /* output_dim */ 512, /* activation */ RELU, /* dropout_rate */ 0.2),
    Layer(/* input_dim */ 512, /* output_dim */ 10, /* activation */ LINEAR, /* dropout_rate */ 0.0)
  }));


  // Load preexisting weights (if provided)
  if (params.model_file != "") {
    ifstream in(params.model_file);
    boost::archive::text_iarchive ia(in);
    ia >> model >> nn;
  }

  // Initialize variables for training -------------------------------------------------------------
  // Worst accuracy
  double worst = 0;

  // Number of batches in training set
  unsigned num_batches = mnist_train.size()  / params.BATCH_SIZE - 1;

  // Random indexing
  unsigned si;
  vector<unsigned> order(num_batches);
  for (unsigned i = 0; i < num_batches; ++i) order[i] = i;

  bool first = true;
  unsigned epoch = 0;
  vector<Expression> cur_batch;
  vector<unsigned> cur_labels;

  // Run for the given number of epochs (or indefinitely if params.NUM_EPOCHS is negative)
  while (epoch < params.NUM_EPOCHS || params.NUM_EPOCHS < 0) {
    // Update the optimizer
    if (first) { first = false; } else { adam.update_epoch(); }
    // Reshuffle the dataset
    cerr << "**SHUFFLE\n";
    random_shuffle(order.begin(), order.end());
    // Initialize loss and number of samples processed (to average loss)
    double loss = 0;
    double num_samples = 0;

    // Start timer
    Timer* iteration = new Timer("completed in");

    // Activate dropout
    nn.enable_dropout();

    for (si = 0; si < num_batches; ++si) {
      // build graph for this instance
      ComputationGraph cg;
      // Compute batch start id and size
      int id = order[si] * params.BATCH_SIZE;
      unsigned bsize = std::min((unsigned) mnist_train.size() - id, params.BATCH_SIZE);
      // Get input batch
      cur_batch = vector<Expression>(bsize);
      cur_labels = vector<unsigned>(bsize);
      for (unsigned idx = 0; idx < bsize; ++idx) {
        cur_batch[idx] = input(cg, {784}, mnist_train[id + idx]);
        cur_labels[idx] = mnist_train_labels[id + idx];
      }
      // Reshape as batch (not very intuitive yet)
      Expression x_batch = reshape(concatenate_cols(cur_batch), Dim({784}, bsize));
      // Get negative log likelihood on batch
      Expression loss_expr = nn.get_nll(x_batch, cur_labels, cg);
      // Get scalar error for monitoring
      loss += as_scalar(cg.forward(loss_expr));
      // Increment number of samples processed
      num_samples += bsize;
      // Compute gradient with backward pass
      cg.backward(loss_expr);
      // Update parameters
      adam.update();
      // Print progress every tenth of the dataset
      if ((si + 1) % (num_batches / 10) == 0 || si == num_batches - 1) {
        // Print informations
        adam.status();
        cerr << " E = " << (loss / num_samples) << ' ';
        // Reinitialize timer
        delete iteration;
        iteration = new Timer("completed in");
        // Reinitialize loss
        loss = 0;
        num_samples = 0;
      }
    }

    // Disable dropout for dev testing
    nn.disable_dropout();

    // Show score on dev data
    if (si == num_batches) {
      double dpos = 0;
      for (unsigned i = 0; i < mnist_dev.size(); ++i) {
        // build graph for this instance
        ComputationGraph cg;
        // Get input expression
        Expression x = input(cg, {784}, mnist_dev[i]);
        // Get negative log likelihood on batch
        unsigned predicted_idx = nn.predict(x, cg);
        // Increment count of positive classification
        if (predicted_idx == mnist_dev_labels[i])
          dpos++;
      }
      // If the dev loss is lower than the previous ones, save the ,odel
      if (dpos > worst) {
        worst = dpos;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model << nn;
      }
      // Print informations
      cerr << "\n***DEV [epoch=" << (epoch)
           << "] E = " << (dpos / (double) mnist_dev.size()) << ' ';
      // Reinitialize timer
      delete iteration;
      iteration = new Timer("completed in");
    }

    // Increment epoch
    ++epoch;

  }


}

