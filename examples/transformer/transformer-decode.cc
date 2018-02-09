/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#include "ensemble-decoder.h"
#include "getpid.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace boost::program_options;

// ---
bool load_data(const variables_map& vm
	, WordIdCorpus& train_cor
	, dynet::Dict& sd, dynet::Dict& td
	, transformer::SentinelMarkers& sm);
// ---

// ---
bool load_model_config(const string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm);
// ---

// ---
void decode(const string test_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned beam_size=5
	, unsigned int lc=0 /*line number to be continued*/
	, bool remove_unk=false /*whether to include <unk> in the output*/
	, bool r2l_target=false /*right-to-left decoding*/);
// ---

//************************************************************************************************************************************************************
int main(int argc, char** argv) {
	cerr << "*** DyNet initialization ***" << endl;
	auto dyparams = dynet::extract_dynet_params(argc, argv);
	dynet::initialize(dyparams);	

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("train,t", value<string>(), "file containing training sentences, with each line consisting of source ||| target.")		
		("train-percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		("max-seq-len", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); none by default")
		("src-vocab", value<string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("tgt-vocab", value<string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")	
		//-----------------------------------------
		("test,T", value<string>(), "file containing testing sentences.")
		("lc", value<unsigned int>()->default_value(0), "specify the sentence/line number to be continued (for decoding only); 0 by default")
		//-----------------------------------------
		("beam,b", value<unsigned>()->default_value(1), "size of beam in decoding; 1: greedy")
		("topk,k", value<unsigned>()->default_value(100), "use <num> top kbest entries, used with --kbest")
		//-----------------------------------------
		("model-cfg,m", value<string>(), "model configuration file (to support ensemble decoding)")
		//-----------------------------------------
		("remove-unk", "remove <unk> in the output; default not")
		//-----------------------------------------
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		("dynet-profiling", value<int>()->default_value(0), "enable/disable auto profiling (https://github.com/clab/dynet/pull/1088/commits/bc34db98fa5e2e694f54f0e6b1d720d517c7530e)")// for debugging only		
		//-----------------------------------------
	;
	
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);

	// print command line
	cerr << endl << "PID=" << getpid() << endl;
	cerr << "Command: ";
	for (int i = 0; i < argc; i++){ 
		cerr << argv[i] << " "; 
	} 
	cerr << endl;
	
	// print help
	if (vm.count("help") 
		|| !(vm.count("train") || (vm.count("src-vocab") && vm.count("tgt-vocab"))) || !vm.count("test"))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// load fixed vocabularies from files if required
	dynet::Dict sd, td;
	load_vocabs(vm["src-vocab"].as<string>(), vm["tgt-vocab"].as<string>(), sd, td);
	cerr << endl;

	transformer::SentinelMarkers sm;
	sm._kSRC_SOS = sd.convert("<s>");
	sm._kSRC_EOS = sd.convert("</s>");
	sm._kTGT_SOS = td.convert("<s>");
	sm._kTGT_EOS = td.convert("</s>");

	// load training data for building vocabularies w/o vocabulary files
	WordIdCorpus train_cor;
	if (!load_data(vm, train_cor, sd, td, sm))
		assert("Failed to load data files!");

	// load models
	std::vector<std::shared_ptr<transformer::TransformerModel>> v_tf_models;
	if (!load_model_config(vm["model-cfg"].as<string>(), v_tf_models, sd, td, sm))
		assert("Failed to load model(s)!");

	// decode the input file
	decode(vm["test"].as<std::string>(), v_tf_models, vm["beam"].as<unsigned>(), vm["lc"].as<unsigned int>(), vm.count("remove-unk"), vm.count("r2l-target"));

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
bool load_data(const variables_map& vm
	, WordIdCorpus& train_cor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm)
{
	bool swap = vm.count("swap");
	bool r2l_target = vm.count("r2l_target");

	if (vm.count("train")){
		cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";		
		train_cor = read_corpus(vm["train"].as<string>(), &sd, &td, true, vm["max-seq-len"].as<unsigned>(), r2l_target & !swap);
		cerr << endl;
	}

	if ("" == vm["src-vocab"].as<string>() 
		&& "" == vm["tgt-vocab"].as<string>()) // if not using external vocabularies
	{
		sd.freeze(); // no new word types allowed
		td.freeze(); // no new word types allowed
	}

	// limit the percent of training data to be used
	unsigned train_percent = vm["train-percent"].as<unsigned>();
	if (train_percent < 100 
		&& train_percent > 0)
	{
		if (vm.count("train")){
			cerr << "Only use " << train_percent << "% of " << train_cor.size() << " training instances: ";
			unsigned int rev_pos = train_percent * train_cor.size() / 100;
			train_cor.erase(train_cor.begin() + rev_pos, train_cor.end());
			cerr << train_cor.size() << " instances remaining!" << endl;
		}
	}
	else if (train_percent != 100){
		cerr << "Invalid --train-percent <num> used. <num> must be (0,100]" << endl;
		return false;
	}

	// set up <s>, </s>, <unk> ids
	sd.set_unk("<unk>");
	sm._kSRC_UNK = sd.get_unk_id();
	td.set_unk("<unk>");
	sm._kTGT_UNK = td.get_unk_id();

	if (swap) {
		cerr << "Swapping role of source and target\n";
		std::swap(sd, td);
		std::swap(sm._kSRC_SOS, sm._kTGT_SOS);
		std::swap(sm._kSRC_EOS, sm._kTGT_EOS);
		std::swap(sm._kSRC_UNK, sm._kTGT_UNK);

		for (auto &sent: train_cor){
			std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				WordIdSentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}		
	}

	return true;
}
// ---

// ---
bool load_model_config(const string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm)
{
	cerr << "Loading model(s) from configuration file: " << model_cfg_file << "..." << endl;	

	v_models.clear();

	ifstream inpf(model_cfg_file);
	assert(inpf);
	
	unsigned i = 0;
	std::string line;
	while (getline(inpf, line)){
		if ("" == line) break;

		// each line has the format: 
		// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <your-trained-model-path>
		// e.g.,
		// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run1
		// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run2
		cerr << "Loading model " << i+1 << "..." << endl;
		stringstream ss(line);

		transformer::TransformerConfig tfc;
		string model_file;

		tfc._src_vocab_size = sd.size();
		tfc._tgt_vocab_size = td.size();
		tfc._sm = sm;
		
		ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
		   >> tfc._encoder_emb_dropout_rate >> tfc._encoder_sublayer_dropout_rate >> tfc._decoder_emb_dropout_rate >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate 
		   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
		   >> tfc._position_encoding >> tfc._max_length
		   >> tfc._attention_type
		   >> tfc._ffl_activation_type
		   >> tfc._use_hybrid_model;		
		ss >> model_file;
		tfc._is_training = false;
		tfc._use_dropout = false;

		v_models.push_back(std::shared_ptr<transformer::TransformerModel>());
		v_models[i].reset(new transformer::TransformerModel(tfc, sd, td));
		v_models[i].get()->initialise_params_from_file(model_file);// load pre-trained model from file
		cerr << "Count of model parameters: " << v_models[i].get()->get_model_parameters().parameter_count() << endl;

		i++;
	}

	cerr << "Done!" << endl << endl;

	return true;
}
// ---

// ---
void decode(const string test_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned beam_size
	, unsigned int lc /*line number to be continued*/
	, bool remove_unk /*whether to include <unk> in the output*/
	, bool r2l_target /*right-to-left decoding*/)
{
	dynet::Dict& sd = v_models[0].get()->get_source_dict();
	dynet::Dict& td = v_models[0].get()->get_target_dict();
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;

	if (beam_size <= 0) assert("Beam size must be >= 1!");

	EnsembleDecoder ens(td);
	ens.set_beam_size(beam_size);

	cerr << "Reading test examples from " << test_file << endl;
	ifstream in(test_file);
	assert(in);

	MyTimer timer_dec("completed in");
	string line;
	WordIdSentence source;
	unsigned int lno = 0;
	while (std::getline(in, line)) {
		if (lno++ < lc) continue;// continued decoding

		source = dynet::read_sentence(line, sd);

		if (source.front() != sm._kSRC_SOS && source.back() != sm._kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;// dynamic computation graph
		WordIdSentence target;//, aligns;

		EnsembleDecoderHypPtr trg_hyp = ens.generate(cg, source, v_models);
		if (trg_hyp.get() == nullptr) {
			target.clear();
			//aligns.clear();
		} 
		else {
			target = trg_hyp->get_sentence();
			//aligns = trg_hyp->get_alignment();
		}

		if (r2l_target)
			std::reverse(target.begin() + 1, target.end() - 1);

		bool first = true;
		for (auto &w: target) {
			if (!first) cout << " ";

			if (remove_unk && w == sm._kTGT_UNK) continue;

			cout << td.convert(w);

			first = false;
		}
		cout << endl;

		//break;//for debug only
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Decoding is finished!" << endl;
	cerr << "Decoded " << (lno - lc) << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}
// ---

