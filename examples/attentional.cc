#include "attentional.h"
#include "cnn/cnn-helper.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;

unsigned LAYERS = 1; // 2
unsigned HIDDEN_DIM = 64;  // 1024
unsigned ALIGN_DIM = 32;   // 128
unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
bool verbose;

float lambda;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

int beam_search_decode = -1; /// -1 is the beam width

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class rnn_t>
int main_body(variables_map vm, int = 2);

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);

    // command line processing
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("seed", value<int>()->default_value(125), "random seed")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("train,t", value<string>(), "file containing training sentences, with "
        "each line consisting of source ||| target.")
        ("devel,d", value<string>(), "file containing development sentences.")
        ("test,T", value<string>(), "file containing testing source sentences (no training)")
        ("rescore,r", "rescore (source, target) pairs in testing, default: translate source only")
        ("testcorpus", value<string>(), "file containing test corpus with target translation")
        ("beamsearchdecode", value<int>()->default_value(-1), "if using beam search decoding; default false")
        ("kbest,K", value<string>(), "test on kbest inputs using mononlingual Markov model")
        ("initialise,i", value<string>(), "load initial parameters from file")
        ("parameters,p", value<string>(), "save best parameters to this file")
        ("outputfile", value<string>(), "save decode and sample results to this file")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("topk,k", value<int>()->default_value(100), "use <num> top kbest entries, used with --kbest")
        ("epochs,e", value<int>()->default_value(50), "maximum number of training epochs")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("dglstm", "use Depth-Gated Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("giza", "use GIZA++ style features in attentional components")
        ("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
        ("lambda", value<float>()->default_value(1e-6), "the L2 regularization coefficient; default 1e-6.")
        ("swap", "swap roles of source and target, i.e., learn p(source|target)")
        ("verbose,v", "be extremely chatty")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && (vm.count("test") != 1 && vm.count("kbest") != 1 && vm.count("testcorpus") != 1))) 
    {
        cout << opts << "\n";
        return 1;
    }

    lambda = vm["lambda"].as<float>();

    if (vm.count("lstm"))
    	return main_body<LSTMBuilder>(vm, 2);
    else if (vm.count("gru"))
        return main_body<GRUBuilder>(vm, 1);
    else if (vm.count("dglstm"))
        return main_body<DGLSTMBuilder>(vm, 2);
    else
    	return main_body<SimpleRNNBuilder>(vm, 1);
}

void initialise(Model &model, const string &filename);

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs);

template <class AM_t>
void test(Model &model, AM_t &am, string test_file);

template <class AM_t>
void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k);

Corpus read_corpus(const string &filename);
void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, int &num);
void ReadNumberedSentencePair(const std::string& line, std::vector<int>* t, Dict* td, int &num);

template <class rnn_t>
int main_body(variables_map vm, int repnumber)
{
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
    kTGT_SOS = td.Convert("<s>");
    kTGT_EOS = td.Convert("</s>");
    verbose = vm.count("verbose");

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel, testcorpus;
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
    bool swap = vm.count("swap");

    string flavour = "RNN";
    if (vm.count("lstm"))	flavour = "LSTM";
    else if (vm.count("gru"))	flavour = "GRU";
    else if (vm.count("dglstm"))	flavour = "DGLSTM";
    SRC_VOCAB_SIZE = sd.size();
    TGT_VOCAB_SIZE = td.size();

    if (vm.count("beamsearchdecode"))
    {
        beam_search_decode = vm["beamsearchdecode"].as<int>();
    }

    if (vm.count("devel")) {
	    cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
	    devel = read_corpus(vm["devel"].as<string>());
    }

    if (vm.count("testcorpus")) {
        cerr << "Reading test corpus from " << vm["testcorpus"].as<string>() << "...\n";
        testcorpus = read_corpus(vm["testcorpus"].as<string>());
    }

    if (swap) {
	cerr << "Swapping role of source and target\n";
        std::swap(sd, td);
        std::swap(kSRC_SOS, kTGT_SOS);
        std::swap(kSRC_EOS, kTGT_EOS);
        std::swap(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE);
        for (auto &sent: training)
            std::swap(sent.first, sent.second);
        for (auto &sent: devel)
            std::swap(sent.first, sent.second);
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

    cerr << "%% Using " << flavour << " recurrent units" << endl;
    AttentionalModel<rnn_t> am(model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
        LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, repnumber);

    if (vm.count("initialise"))
	initialise(model, vm["initialise"].as<string>());

    if (!vm.count("test") && !vm.count("kbest") &&!vm.count("testcorpus"))
    	train(model, am, training, devel, *sgd, fname, vm.count("curriculum"), vm["epochs"].as<int>());
    else if (vm.count("kbest"))
        test_kbest_arcs(model, am, vm["kbest"].as<string>(), vm["topk"].as<int>());
    else if (vm.count("rescore"))
    {
        if (vm.count("outputfile") == 0)
        {
            cerr << "missing recognition output file" << endl;
            abort();
        }
        test_rescore(model, am, vm["test"].as<string>() , vm["outputfile"].as<string>());
    }
    else if (vm.count("testcorpus"))
    {
        if (vm.count("outputfile") == 0)
        {
            cerr << "missing recognition output file" << endl;
            abort();
        }
        test(model, am, testcorpus, vm["outputfile"].as<string>());
    }
    else
    {
        test(model, am, vm["test"].as<string>());
    }

    delete sgd;

    return EXIT_SUCCESS;
}

template <class AM_t>
void test_rescore(Model &model, AM_t &am, string test_file, string out_file)
{
    int lno = 0;

    cerr << "Reading test examples from " << test_file << endl;
    ifstream in(test_file);
    assert(in);

    ofstream of(out_file);
    assert(of);

    string line;

    while (getline(in, line)) {
        Sentence target;
        Sentence source;
        int num;
        ReadNumberedSentencePair(line, &source, &sd, &target, &td, num);
        
        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
            (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
            abort();
        }

        ComputationGraph cg;
        am.BuildGraph(source, target, cg, nullptr);

        double loss = as_scalar(cg.forward());

/*        cout << num << ' ';
        cout << "|||";
        for (auto &w : target)
            cout << " " << td.Convert(w);
        cout << " ||| " << loss << endl;
        */

        of << line << " ||| " << loss << endl; 

//        cerr << "procesed " << lno << " sentences" << endl;

        lno++;
    }
    cerr << "total " << lno << " source sentences" << flush;

    in.close();
    of.close();
    return;
}

template <class AM_t>
void test(Model &model, AM_t &am, string test_file)
{
    double tloss = 0;
    int tchars = 0;
    int lno = 0;

    cerr << "Reading test examples from " << test_file << endl;
    ifstream in(test_file);
    assert(in);
    string line;
    while(getline(in, line)) {
	Sentence source, target;
        int num = -1;
	ReadNumberedSentencePair(line, &source, &sd, &target, &td, num);
	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
	    abort();
	}

	ComputationGraph cg;
	am.BuildGraph(source, target, cg);
	double loss = as_scalar(cg.forward());
	cout << num << " |||";
	for (auto &w: source)
	    cout << " " << sd.Convert(w);
	cout << " |||";
	for (auto &w: target)
	    cout << " " << td.Convert(w);
	cout << " ||| " << loss << endl;
	tloss += loss;
	tchars += target.size() - 1;

	if (verbose)
	    cerr << "chug " << lno++ << "\r" << flush;
    }

    cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
    in.close();
    return;
}

template <class AM_t>
void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k)
{
    // only suitable for monolingual setting, of predicting a sentence given preceeding sentence
    cerr << "Reading test examples from " << test_file << endl;
    unsigned lno = 0;
    ifstream in(test_file);
    assert(in);
    string line, last_id, last_last_id = "-";
    const std::string sep = "|||";
    vector<SentencePair> items, last_items;
    last_items.push_back(SentencePair(Sentence({ kSRC_SOS, kSRC_EOS }), Sentence({ kTGT_SOS, kTGT_EOS })));

    while(getline(in, line)) {
	Sentence source, target;

	istringstream in(line);
	string id, word;
	in >> id >> word;
	assert(word == sep);
	while(in) {
	    in >> word;
	    if (word.empty() || word == sep) break;
	    source.push_back(sd.Convert(word));
	    target.push_back(td.Convert(word));
	}

	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
	    abort();
	}

	if (id != last_id && !items.empty()) {
	    if (items.size() > top_k)
		items.resize(top_k);

	    unsigned count = 0;
	    for (auto &prev: last_items) {
		ComputationGraph cg;
		auto &source = prev.first;
		am.start_new_instance(source, cg);

		for (auto &curr: items) {
		    std::vector<Expression> errs;
		    auto &target = curr.second;
		    const unsigned tlen = target.size() - 1;
		    for (unsigned t = 0; t < tlen; ++t) {
			Expression i_r_t = am.add_input(target[t], t, cg);
			Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
			errs.push_back(i_err);
		    }
		    Expression i_nerr = sum(errs);
		    double loss = as_scalar(cg.incremental_forward());

		    cout << last_last_id << ":" << last_id << " |||";
		    for (auto &w: source) cout << " " << sd.Convert(w);
		    cout << " |||";
		    for (auto &w: target) cout << " " << td.Convert(w);
		    cout << " ||| " << loss << "\n";

		    ++count;
		}
	    }

	    last_items = items;
	    last_last_id = last_id;
	    last_id = id;
	    items.clear();

	    if (verbose)
		cerr << "chug " << lno++ << " [" << count << " pairs]\r" << flush;
	}

	last_id = id;
	items.push_back(SentencePair(source, target));
    }

    in.close();
    return;
}

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs)
{
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

    while (sgd.epoch < max_epochs) {
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
		auto& spair = training[order[si % order.size()]];
		ComputationGraph cg;
                cerr << "\nDecoding source, greedy Viterbi: ";
                am.decode(spair.first, cg, 1, td);

                cerr << "\nDecoding source, sampling: ";
                am.sample(spair.first, cg, td);
	    }

            // build graph for this instance
	    auto& spair = training[order[si % order.size()]];
	    ComputationGraph cg;
            chars += spair.second.size() - 1;
            ++si;
            Expression alignment;
            am.BuildGraph(spair.first, spair.second, cg, &alignment);
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

template <class AM_t>
void test(Model &model, AM_t &am, Corpus &devel, string out_file)
{
    unsigned lines = 0;
    ofstream of(out_file);

    Timer iteration("completed in");
    for (auto& spair : devel)
    {
        ComputationGraph cg;
        cerr << "\nDecoding source, greedy Viterbi: ";
        vector<int> decode_output;
        if (beam_search_decode != -1)
            decode_output = am.beam_decode(spair.first, cg, beam_search_decode, td);
        else
            decode_output = am.decode(spair.first, cg, 1, td);
        
        of << "ref : ";
        for (auto pp : spair.second)
        {
            of << td.Convert(pp) << " ";
        }
        of << endl;
        of << "res : ";
        for (auto pp : decode_output)
        {
            of << td.Convert(pp) << " ";
        }
        of << endl;

/*      to-do: check if there is memory issue, as beam width =3, and there is no result from sampling function      
    cerr << "\nDecoding source, sampling: ";
        vector<int> sample_output = am.sample(spair.first, cg, td);
        of << "sam : ";
        for (auto pp : sample_output)
        {
            of << td.Convert(pp) << " ";
        }
        of << endl;
        */
        of << endl;
    }

    double dloss = 0;
    int dchars = 0;
    for (auto& spair : devel) {
        ComputationGraph cg;
        am.BuildGraph(spair.first, spair.second, cg);
        dloss += as_scalar(cg.forward());
        dchars += spair.second.size() - 1;
    }
    cerr << "\n***TEST E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';

    of.close();
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
    in.close();
    return corpus;
}

void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, int &num) 
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::vector<int>* v = s; 

    if (in) {
        in >> num;
        in >> word;
        assert(word == sep);
    }

    while(in) {
        in >> word;
        if (!in) break;
        if (word == sep) { d = td; v = t; continue; }
        v->push_back(d->Convert(word));
    }
}

/// read kbest list 
/// example of input 
/// 0 ||| can i have a table ? ||| LanguageModel=-5.64651 WordPenalty=-2.60577 SampleCountF=5.35014 IsSingletonFE=0 IsSingletonF=0 MaxLexFgivenE=4.1915 CountEF=4.19728 MaxLexEgivenF=2.02649 EgivenFCoherent=1.24972 ||| -10.2169
void ReadNumberedSentencePair(const std::string& line, std::vector<int>* t, Dict* td, int &num)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    std::vector<int>* v = t;

    if (in) {
        in >> num;
        in >> word;
        assert(word == sep);
    }

    while (in) {
        in >> word;
        if (!in) break;
        if (word == sep) { break; }
        v->push_back(td->Convert(word));
    }
}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> model;
    in.close();
}
