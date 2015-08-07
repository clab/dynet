#include "attentional.h"
#include "cnn/cnn-helper.h"
#include "cnn/grad-check.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;

long  LAYERS = 1; // 2
long  HIDDEN_DIM = 64;  // 1024
long  ALIGN_DIM = 32;   // 128
long  SRC_VOCAB_SIZE = 0;
long  TGT_VOCAB_SIZE = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
bool verbose;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string init_file, string out_file, string test,
	bool curriculum);

Corpus read_corpus(const string &filename);

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);

    bool use_external_memory = false; 

    // command line processing
    using namespace boost::program_options;
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("seed,s", value<int>()->default_value(217), "random seed number")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("train,t", value<string>(), "file containing training sentences, with "
        "each line consisting of source ||| target.")
        ("devel,d", value<string>(), "file containing development sentences.")
        ("test,T", value<string>(), "file containing testing source sentences")
        ("initialise,i", value<string>(), "load initial parameters from file")
        ("parameters,p", value<string>(), "save best parameters to this file")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("dglstm", "use depth-gated LSTM for recurrent structure; default RNN")
        ("rnnem", "use ExtMemLSTM RNN (RNNEM) for recurrent struture with past input history; default RNN")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("giza", "use GIZA++ style features in attentional components")
        ("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
        ("lambda", value<float>()->default_value(1e-6), "the L2 regularization coefficient; default 1e-6.")
        ("verbose,v", "be extremely chatty")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && vm.count("test") != 1)) {
        cout << opts << "\n";
        return 1;
    }

    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
    kTGT_SOS = td.Convert("<s>");
    kTGT_EOS = td.Convert("</s>");
    verbose = vm.count("verbose");
    float lambda = vm["lambda"].as<float>();

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel;
    string line;
    cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
    training = read_corpus(vm["train"].as<string>());
    sd.Freeze(); // no new word types allowed
    td.Freeze(); // no new word types allowed
    
    LAYERS = vm["layers"].as<int>(); 
    ALIGN_DIM = vm["align"].as<int>(); 
    HIDDEN_DIM = vm["hidden"].as<int>();
    bool bidir = vm.count("bidirectional");
    bool giza = vm.count("giza");
    string flavour;
    if (vm.count("gru"))	flavour = "gru";
    else if (vm.count("lstm"))	flavour = "lstm";
    else if (vm.count("rnnem"))	flavour = "rnnem";
    else if (vm.count("dglstm")) flavour = "dglstm";
    else			flavour = "rnn";
    SRC_VOCAB_SIZE = sd.size();
    TGT_VOCAB_SIZE = td.size();

    if (vm.count("devel")) {
	cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
	devel = read_corpus(vm["devel"].as<string>());
    }

    string fname;
    if (vm.count("parameters")) {
	fname = vm["parameters"].as<string>();
    } else {
	ostringstream os;
	os << "am"
	    << '_' << LAYERS
	    << '_' << HIDDEN_DIM
	    << '_' << ALIGN_DIM
        << '_' << RNNEM_MEM_SIZE
	    << '_' << flavour
	    << "_b" << bidir
	    << "_g" << giza
	    << "-pid" << getpid() << ".params";
	fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    //bool use_momentum = false;
    Trainer* sgd = nullptr;
    //if (use_momentum)
        //sgd = new MomentumSGDTrainer(&model);
    //else
        sgd = new SimpleSGDTrainer(&model, lambda);
    //sgd = new AdadeltaTrainer(&model);

    string init_file;
    if (vm.count("initialise"))
	init_file = vm["initialise"].as<string>();

    string test;
    if (vm.count("test")) 
	test = vm["test"].as<string>();

    if (vm.count("lstm")) {
        cerr << "%% Using LSTM recurrent units" << endl;
        AttentionalModel<LSTMBuilder> am(model, 
		SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, 2);
        train(model, am, training, devel, *sgd, init_file, fname, test, vm.count("curriculum"));
    }
    else if (vm.count("dglstm")) {
        cerr << "%% Using DGLSTM recurrent units" << endl;
        AttentionalModel<DGLSTMBuilder> am(model,
            SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
            LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, 2);
        train(model, am, training, devel, *sgd, init_file, fname, test, vm.count("curriculum"));
    }
    else if (vm.count("gru")) {
        cerr << "%% Using GRU recurrent units" << endl;
        AttentionalModel<GRUBuilder> am(model,
            SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
            LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, 1);
        train(model, am, training, devel, *sgd, init_file, fname, test, vm.count("curriculum"));
    }
    else if (vm.count("rnnem")) {
        cerr << "%% Using RNNEM recurrent units" << endl;
        AttentionalModel<RNNEMBuilder> am(model,
            SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
            LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, 2, nullptr, nullptr, true);
        train(model, am, training, devel, *sgd, init_file, fname, test, vm.count("curriculum"));
    }
    else {
        cerr << "%% Using RNN recurrent units" << endl;
        AttentionalModel<SimpleRNNBuilder> am(model, 
		SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, 1);
        train(model, am, training, devel, *sgd, init_file, fname, test, vm.count("curriculum"));
    }

    delete sgd;

    return EXIT_SUCCESS;
}

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string init_file, string out_file, string test_file, bool curriculum)
{
    if (!init_file.empty()) {
	ifstream in(init_file);
	boost::archive::text_iarchive ia(in);
	ia >> model;
    }

    if (test_file != "") {
	double tloss = 0;
	int tchars = 0;
	int lno = 0;

	cerr << "Reading test examples from " << test_file << endl;
	ifstream in(test_file);
	assert(in);
	string line;
	while(getline(in, line)) {
	    Sentence source, target;
	    ReadSentencePair(line, &source, &sd, &target, &td);
	    if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
		    (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
		cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
		abort();
	    }

	    ComputationGraph cg;
	    am.BuildGraph(source, target, cg);
	    double loss = as_scalar(cg.forward());
	    for (auto &w: source)
		cout << sd.Convert(w) << " ";
	    cout << "|||";
	    for (auto &w: target)
		cout << " " << td.Convert(w);
	    cout << "||| " << loss << endl;
	    tloss += loss;
	    tchars += target.size() - 1;

	    if (verbose)
		cerr << "chug " << lno++ << "\r" << flush;
	}

	cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
	return;
    }

    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500; 
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    vector<vector<unsigned>> order_by_length; 
    const unsigned curriculum_steps = 10;
    if (curriculum) {
	// simple form of curriculum learning: for the first K epochs, use only
	// the shortest examples from the training set. E.g., K=10, then in
	// epoch 0 using the first decile, epoch 1 use the first & second
	// deciles etc. up to the full dataset in k >= 9.
	multimap<size_t, unsigned> lengths;
	for (unsigned i = 0; i < training.size(); ++i) 
	    lengths.insert(make_pair(training[i].first.size(), i));

	order_by_length.resize(curriculum_steps);
	unsigned i = 0;
	for (auto& landi: lengths) {
	    for (unsigned k = i * curriculum_steps / lengths.size(); k < curriculum_steps; ++k)  
		order_by_length[k].push_back(landi.second);
	    ++i;
	}
    }

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    while(1) {
        Timer iteration("completed in");
        double loss = 0;
        unsigned chars = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd.update_epoch(); }

		if (curriculum && epoch < order_by_length.size()) {
		    order = order_by_length[epoch++];
		    cerr << "Curriculum learning, with " << order.size() << " examples\n";
		} 
	    }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
	    }

	    if (verbose && iter+1 == report_every_i) {
		auto& spair = training[order[si]];
		ComputationGraph cg;
                cerr << "\nDecoding source, greedy Viterbi: ";
                am.decode(spair.first, cg, 1, td);

                cerr << "\nDecoding source, sampling: ";
                am.sample(spair.first, cg, td);
	    }

            // build graph for this instance
	    auto& spair = training[order[si]];
	    ComputationGraph cg;
            chars += spair.second.size() - 1;
            ++si;
            Expression alignment;
            am.BuildGraph(spair.first, spair.second, cg, &alignment);

            /**
            check gradients
            */
//            CheckGrad(model, cg);

            loss += as_scalar(cg.forward());
            
            cg.backward();
            sgd.update();
            ++lines;

	    if (verbose) {
		cerr << "chug " << iter << "\r" << flush;
		if (iter+1 == report_every_i) {
		    // display the alignment
		    am.display(spair.first, spair.second, cg, alignment, sd, td);
		}
	    }
        }
        sgd.status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (auto& spair : devel) {
                ComputationGraph cg;
                am.BuildGraph(spair.first, spair.second, cg);
                dloss += as_scalar(cg.forward());
                dchars += spair.second.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(out_file);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
}

Corpus read_corpus(const string &filename)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    while(getline(in, line)) {
        ++lc;
        Sentence source, target;
        ReadSentencePair(line, &source, &sd, &target, &td);
        corpus.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
                (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    return corpus;
}
