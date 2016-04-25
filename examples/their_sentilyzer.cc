#include "theirsentimentmodel.cc"
#include "sentilyzer_helper.cc"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
bool USE_MOMENTUM = false;

void RunTest(string fname, Model& model, vector<pair<DepTree, vector<int>>>& test, TheirSentimentModel<TreeLSTMBuilder>& sentimodel) {
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;

    double cor = 0;
    double tot = 0;

    auto time_begin = chrono::high_resolution_clock::now();
    for (auto& test_ex : test) {
        ComputationGraph cg;
        int predicted_sentiment;
        sentimodel.BuildTreeCompGraph(test_ex.first, vector<int>(), &cg, &predicted_sentiment);
        EvaluateTags(test_ex.first, test_ex.second, predicted_sentiment, &cor, &tot);
    }

    double acc = cor/tot;
    auto time_end = chrono::high_resolution_clock::now();
    cerr << "TEST accuracy: " << acc << "\t[" << test.size() << " sents in "
    << std::chrono::duration<double, std::milli>(time_end - time_begin).count()
    << " ms]" << endl;
}

void RunTraining(Model& model, Trainer* sgd,
        TheirSentimentModel<TreeLSTMBuilder>& sentimodel,
        vector<pair<DepTree, vector<int>>>& training,
vector<pair<DepTree, vector<int>>>& dev, string* softlinkname) {
    ostringstream os;
    os << "sentanalyzer" << '_' << LAYERS << '_' << LSTM_INPUT_DIM << '_'
    << HIDDEN_DIM << "-pid" << getpid() << ".params";
    const string savedmodelfname = os.str();
    cerr << "Parameters will be written to: " << savedmodelfname << endl;
    bool soft_link_created = false;

    unsigned report_every_i = 100;
    unsigned dev_every_i_reports = 25;
    unsigned si = training.size();

    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    double tot_seen = 0;
    bool first = true;
    int report = 0;
    unsigned trs = 0;
    double llh = 0;
    double best_acc = 0.0;
    int iter = -1;

    while (1) {
        ++iter;
        if (tot_seen > 20 * training.size()) {
            break; // early stopping
        }

        Timer iteration("completed in");
        double llh = 0;

        for (unsigned tr_idx = 0; tr_idx < report_every_i; ++tr_idx) {
            if (si == training.size()) {
                si = 0;
                if (first) {
                    first = false;
                } else {
                    sgd->update_epoch();
                }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance

            auto& sent = training[order[si]];
            int predicted_sentiment;

            ComputationGraph cg;
            sentimodel.BuildTreeCompGraph(sent.first, sent.second, &cg, &predicted_sentiment);

            llh += as_scalar(cg.incremental_forward());
            cg.backward();
            sgd->update(1.0);

            ++si;
            ++trs;
            ++tot_seen;
        }
        sgd->status();
        cerr << "update #" << iter << " (epoch " << (tot_seen / training.size()) << ")\t" << " llh: " << llh << " ppl = " << exp(llh / trs);

        // show score on dev data
        if (report % dev_every_i_reports == 0) {
            //double dloss = 0;
            double dcor = 0;
            double dtags = 0;
            //lm.p_th2t->scale_parameters(pdrop);
            for (auto& dev_ex : dev) {
                ComputationGraph dev_cg;
                int dev_predicted_sentiment;

                sentimodel.BuildTreeCompGraph(dev_ex.first, vector<int>(), &dev_cg,
                &dev_predicted_sentiment);
                //dloss += as_scalar(dev_cg.forward());
                EvaluateTags(dev_ex.first, dev_ex.second, dev_predicted_sentiment, &dcor, &dtags);
            }
            cerr << "\n***DEV [epoch=" << (tot_seen / training.size())
            << "]" << " accuracy = " << (dcor/dtags);

            double acc = dcor/dtags;

            if (acc > best_acc) {
                best_acc = acc;
                ofstream out(savedmodelfname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                cerr << " ^^ Updated model ^^" << endl;

                if (soft_link_created == false) {
                    string softlink = string(" their_latest_model_");
                    if (softlinkname) { // if output model file is specified
                        softlink = " " + *softlinkname;
                    }
                    if (system((string("rm -f ") + softlink).c_str()) == 0
                    && system((string("ln -s ") + savedmodelfname + softlink).c_str()) == 0) {
                        cerr << "Created " << softlink << " as a soft link to "
                        << savedmodelfname << " for convenience.";
                    }
                    soft_link_created = true;
                }
            }
        }
        report++;
    }
    delete sgd;
}

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);

    cerr << "COMMAND:";
    for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
        cerr << ' ' << argv[i];
    cerr << endl;

    po::variables_map conf;
    po::options_description opts("Configuration options");

    opts.add_options()("training_data,T", po::value<string>(),
            "Training corpus in CoNLL format")("dev_data,D",
            po::value<string>(), "Development corpus in CoNLL format")(
            "model,m", po::value<string>(), "load saved model from this file")(
            "out_model,o", po::value<string>(),
            "save output model to this soft link")("use_pos_tags,P",
            "make POS tags visible to parser")("layers",
            po::value<unsigned>()->default_value(1), "number of LSTM layers")(
            "lstm_input_dim", po::value<unsigned>()->default_value(300),
            "LSTM input dimension")("input_dim",
            po::value<unsigned>()->default_value(32), "input embedding size")(
            "hidden_dim", po::value<unsigned>()->default_value(168),
            "hidden dimension")("pretrained_dim",
            po::value<unsigned>()->default_value(300), "pretrained input dim")(
            "pos_dim", po::value<unsigned>()->default_value(12),
            "POS dimension")("deprel_dim",
            po::value<unsigned>()->default_value(10),
            "dependency relation dimension")("dropout",
            po::value<float>()->default_value(0.0f), "Dropout rate")("train,t",
            "Should training be run?")("words,w", po::value<string>(),
            "pretrained word embeddings")("help,h", "Help");

    po::options_description dcmdline_options;
    dcmdline_options.add(opts);
    po::store(parse_command_line(argc, argv, dcmdline_options), conf);
    if (conf.count("help")) {
        cerr << dcmdline_options << endl;
        exit(1);
    }
    if (conf.count("training_data") == 0) {
        cerr << "Please specify --traing_data (-T):"
                " this is required to determine the vocabulary mapping,"
                " even if the parser is used in prediction mode.\n";
        exit(1);
    }

    LAYERS = conf["layers"].as<unsigned>();
//    INPUT_DIM = conf["input_dim"].as<unsigned>();
//    PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
    HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
    LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
//    POS_DIM = conf["pos_dim"].as<unsigned>();
//    REL_DIM = conf["deprel_dim"].as<unsigned>();

    vector<pair<DepTree, vector<int>>> training, dev;

    string training_fname = conf["training_data"].as<string>();
    cerr << "Reading training data from " << training_fname << "...\n";
    ReadCoNLLFile(training_fname, training, &tokdict, &depreldict,
            &sentitagdict);

    tokdict.Freeze(); // no new word types allowed
    tokdict.SetUnk(UNK_STR);
    sentitagdict.Freeze(); // no new tag types allowed
    for (unsigned i = 0; i < sentitagdict.size(); ++i) {
        sentitaglist.push_back(i);
    }
    depreldict.Freeze();

    VOCAB_SIZE = tokdict.size();
    DEPREL_SIZE = depreldict.size();
    SENTI_TAG_SIZE = sentitagdict.size();

    string dev_fname = conf["dev_data"].as<string>();
    cerr << "Reading dev data from " << dev_fname << "...\n";
    ReadCoNLLFile(dev_fname, dev, &tokdict, &depreldict, &sentitagdict);

    Model model;
    Trainer* sgd = nullptr;
    if (USE_MOMENTUM)
        sgd = new MomentumSGDTrainer(&model);
    else
        sgd = new AdamTrainer(&model);

    TheirSentimentModel < TreeLSTMBuilder > sentimodel(model);
    if (conf.count("train")) { // test mode
        string softlinkname;
        if (conf.count("out_model")) {
            softlinkname = conf["out_model"].as<string>();
        }
        RunTraining(model, sgd, sentimodel, training, dev, &softlinkname);
    }

    string model_fname = conf["model"].as<string>();
    RunTest(model_fname, model, dev, sentimodel);

}
