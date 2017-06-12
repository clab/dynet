/**
 * Train a neural phrase selection model 
 *
 * This uses the model in "encdec.h"
 */

#include "encdec.h"
//#include "utils.h"
#include "../utils/getpid.h"
#include <ctime>
//#include "compressed-fstream.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace dynet;
namespace po = boost::program_options;

unsigned LAYERS = 1;
unsigned OUT_LAYERS = 2;
unsigned INPUT_DIM = 50;
unsigned HIDDEN_DIM = 128;
unsigned PRET_SRC_DIM = 0;
unsigned PRET_TGT_DIM = 0;

Dict sd, td;

float ALPHA = 1.f;
float DROPOUT = 0.0f;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "Training parallel corpus")
        ("training_instances", po::value<string>(), "Training instances")
        ("dev_data,d", po::value<string>(), "Dev parallel corpus")
        ("dev_instances", po::value<string>(), "Training instances")
        ("dropout,D", po::value<float>()->default_value(0.0), "Dropout rate")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("layers", po::value<unsigned>()->default_value(1), "encoding LSTM layers")
        ("outlayers", po::value<unsigned>()->default_value(2), "encoding LSTM layers")
        ("input_dim", po::value<unsigned>()->default_value(50), "input word embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(128), "hidden dimension")
        ("pret_source", po::value<string>(), "pretrained word embedding for the source language")
        ("pret_target", po::value<string>(), "pretrained word embedding for the target language")
        ("pret_source_dim", po::value<unsigned>(), "dimension of pretrained word embedding, source lang")
        ("pret_target_dim", po::value<unsigned>(), "dimension of pretrained word embedding, target lang")
        ("train,t", "Should training be run?")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even at decoding.\n";
    exit(1);
  }
}

unsigned ReadEmbeddings_word2vec(const string& fname,
        Dict* dict,
        unordered_map<unsigned, vector<float>>* pretrained) {
  cerr << "Reading pretrained embeddings from " << fname << " ...\n";
  ifstream in(fname);
  assert(in);
  string line;
  getline(in, line);
  bool bad = false;
  int spaces = 0;
  for (auto c : line) {
    if (c == ' ' || c == '\t') ++spaces;
    else if (c < '0' || c > '9') bad = true;
  }
  if (spaces != 1 || bad) {
    cerr << "File does not seem to be in word2vec format\n";
    abort();
  }
  istringstream iss(line);
  unsigned nwords = 0, dims = 0;
  iss >> nwords >> dims;
  cerr << "    file reports " << nwords << " words with " << dims << " dims\n";
  unsigned lc = 1;
  string word;
  unsigned count_pret = 0;
  while(getline(in, line)) {
    ++lc;
    vector<float> v(dims);
    istringstream iss(line);
    iss >> word;
    if(dict->contains(word)) {
      ++count_pret;
      unsigned wordid = dict->convert(word);
      for (unsigned i = 0; i < dims; ++i)
        iss >> v[i];
      (*pretrained)[wordid] = v;
    }
  }
  if ((lc-1) != nwords) {
    cerr << "[WARNING] mismatched number of words reported and loaded\n";
  }
  cerr << "    done.\n";
  return count_pret;
}

vector<string> indicesToString(const vector<int>& indices, Dict& d) {
  vector<string> output;
  for(unsigned i = 0; i < indices.size(); ++i) {
    output.push_back(d.convert(indices[i]));
  }
  assert(output.size() == indices.size());
  return output;
}

string joinStringVec(const vector<string>& stringVec) {
  stringstream output_str;
  for(unsigned i = 0; i < stringVec.size(); ++i) {
    output_str << stringVec[i];
    if(i < (stringVec.size() - 1)) {
      output_str << " ";
    }
  }
  return output_str.str();
}

void printVector(const vector<float>& floatVec) {
  for(unsigned i = 0; i < floatVec.size(); ++i) {
    cerr << floatVec[i] << " ";
  }
  cerr << endl;
}

vector<Expression> computeSpans(unsigned slen, const vector<Expression>& enc, const vector<pair<unsigned, unsigned>>& spans) {
  assert(enc.size() == (slen + 1));
  vector<Expression> output;
  for(unsigned i = 0; i < spans.size(); ++i) {
    unsigned start = spans[i].first;
    unsigned end = spans[i].second;
    assert((start < end) && (end <= slen) && (end - start <= 5) && ((start + 1) <= end) && ((end - start) >= 1));
    if(end - start == 1) 
	assert(start + 1 == end);
    vector<Expression> temp;
    temp.push_back(enc[end] - enc[start]); 
    temp.push_back(enc[end]);
    temp.push_back(enc[start + 1]);
    assert(temp.size() == 3);
    output.push_back(concatenate(temp));
  }
  assert(output.size() == spans.size());
  return output;
}


int main(int argc, char** argv) {
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);

  cerr << "COMMAND LINE:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;

  // initialize the command line
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  LAYERS = conf["layers"].as<unsigned>();
  DROPOUT = conf["dropout"].as<float>(); 
  OUT_LAYERS = conf["outlayers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();

  ostringstream os;
  os << "npsm"
     << '_' << LAYERS
     << '_' << OUT_LAYERS
     << '_' << DROPOUT 
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  // convert kSOS and kEOS symbols 
  kSOS = sd.convert("<s>");
  kEOS = sd.convert("</s>");

  td.convert("<s>"); 
  td.convert("</s>"); 

  assert(td.convert("<s>") == kSOS &&  td.convert("</s>") == kEOS);
  assert(sd.size() == 2 && td.size() == 2);

  vector<vector<int>> train_x, train_y;
  vector<vector<vector<int>>> train_tgt;
  vector<vector<pair<unsigned, unsigned>>> train_spans; 
  vector<map<int, vector<int>>> train_instances;

  string line;
  string trainingf = conf["training_data"].as<string>();
  cerr << "Reading training data from " << trainingf << "...\n";
  {
    ifstream in(trainingf);
    assert(in);
    while(getline(in, line)) {
       //cerr << line << endl;
       // initialize all the temporary containers
       vector<int> x, y;
       vector<vector<int>> tgt;
       vector<pair<unsigned, unsigned>> spans;

       // read triples of src ||| tgt ||| candidates
//        read_sentence_triples(line, x, sd, y, td, spans, tgt);

       // append to the training data
       train_x.push_back(x);
       train_y.push_back(y);
       train_tgt.push_back(tgt);
       train_spans.push_back(spans);
       // the length of the spans and the target must be exactly the same
       assert(spans.size() == tgt.size());
    }
    sd.freeze(); td.freeze();
    cerr << "Number of unique source word types: " << sd.size() << endl;
    cerr << "Number of unique target word types: " << td.size() << endl;
    assert(train_x.size() == train_y.size() && train_x.size() == train_tgt.size() && train_x.size() == train_spans.size());
    cerr << "Total number of training parallel sentences (and their candidates): " << train_x.size() << endl;
  }

  if(conf.count("pret_source")) {
    string pret_source = conf["pret_source"].as<string>();
    assert(conf.count("pret_source_dim"));
    PRET_SRC_DIM = conf["pret_source_dim"].as<unsigned>();
    assert(PRET_SRC_DIM > 0);
    cerr << "Reading the source pretrained embedding from " << pret_source << " with " << PRET_SRC_DIM << " dimensions" << endl; 
    unsigned count_pret = ReadEmbeddings_word2vec(pret_source, &sd, &src_pret); 
    assert(src_pret.size() == count_pret);
    cerr << "UNK Rate: " << 100.0f - ((float) count_pret / (float) sd.size() * 100.0f) << endl;
  }

  if(conf.count("pret_target")) {
    string pret_target = conf["pret_target"].as<string>();
    assert(conf.count("pret_target_dim"));
    PRET_TGT_DIM = conf["pret_target_dim"].as<unsigned>();
    assert(PRET_TGT_DIM > 0);
    cerr << "Reading the target pretrained embedding from " << pret_target << " with " << PRET_TGT_DIM << " dimensions" << endl;
    unsigned count_pret = ReadEmbeddings_word2vec(pret_target, &td, &tgt_pret);
    assert(tgt_pret.size() == count_pret);
    cerr << "UNK Rate: " << 100.0f - ((float) count_pret / (float) td.size() * 100.0f) << endl;
  }
  
  if (conf.count("training_instances")) {
     assert(conf.count("training_instances"));
     string training_instf = conf["training_instances"].as<string>();
     cerr << "Reading the training instances from " << training_instf << endl; 

     ifstream in(training_instf);
     assert(in);
     for(unsigned i = 0; i < train_x.size(); ++i) {
       map<int, vector<int>> instancesDict;
       train_instances.push_back(instancesDict);
     }
     assert(train_instances.size() == train_x.size());
     unsigned train_inst_ctr = 0;
     while(getline(in, line)) {
       ++train_inst_ctr;
       //cerr << train_inst_ctr << endl;
       unsigned sent_num;// = get_sent_num(line); 
       assert(sent_num < train_instances.size());
//       read_instance_triples(line, train_instances[sent_num]);
     }
     cerr << "Read " << train_inst_ctr << " instances from the training data" << endl;
     unsigned total = 0;
     for(unsigned i = 0; i < train_instances.size(); ++i) {
        for(map<int, vector<int>>::iterator it = train_instances[i].begin(); it != train_instances[i].end(); ++it) {
	  total += (it->second).size();
        }
     }
     assert(total == train_inst_ctr);
  }

  vector<vector<int>> dev_x, dev_y;
  vector<vector<vector<int>>> dev_tgt;
  vector<vector<pair<unsigned, unsigned>>> dev_spans;
  vector<map<int, vector<int>>> dev_instances;  

  if(conf.count("dev_data"))   {
    string devf = conf["dev_data"].as<string>();
    cerr << "Reading dev data from " << devf << "...\n";
    ifstream in(devf);
    assert(in);
    while(getline(in, line)) {
       //cerr << line << endl;
       // initialize all the temporary containers
       vector<int> x, y;
       vector<vector<int>> tgt;
       vector<pair<unsigned, unsigned>> spans;

       // read triples of src ||| tgt ||| candidates
//       read_sentence_triples(line, x, sd, y, td, spans, tgt);

       // append to the dev data
       dev_x.push_back(x);
       dev_y.push_back(y);
       dev_tgt.push_back(tgt);
       dev_spans.push_back(spans);
       // the length of the spans and the target must be exactly the same
       assert(spans.size() == tgt.size());
    }
    
    assert(conf.count("dev_instances"));
    {
       string dev_instf = conf["dev_instances"].as<string>();
       cerr << "Reading the dev instances from " << dev_instf << endl;

       ifstream in(dev_instf);
       assert(in);
       for(unsigned i = 0; i < dev_x.size(); ++i) {
         map<int, vector<int>> instancesDict;
         dev_instances.push_back(instancesDict);
       }
       assert(dev_instances.size() == dev_x.size());
       unsigned dev_inst_ctr = 0;
       while(getline(in, line)) {
         ++dev_inst_ctr;
         unsigned sent_num;// = get_sent_num(line);
         assert(sent_num < dev_instances.size());
//         read_instance_triples(line, dev_instances[sent_num]);
       }
       cerr << "Read " << dev_inst_ctr << " instances from the dev data" << endl;
       unsigned total = 0;
       for(unsigned i = 0; i < dev_instances.size(); ++i) {
          for(map<int, vector<int>>::iterator it = dev_instances[i].begin(); it != dev_instances[i].end(); ++it) {
            total += (it->second).size();
          }
       }
       assert(total == dev_inst_ctr);
    }
  }

  // Map the vocabulary
  INPUT_VOCAB_SIZE = sd.size();
  OUTPUT_VOCAB_SIZE = td.size();

  // Create the encoder decoder model
  Model model;
  // Use Adam optimizer
  Trainer* sgd = nullptr;
  //sgd = new AdamTrainer(model, 0.001, 0.9, 0.999, 1e-8);
  cerr << "Training using SGD" << endl;
  sgd = new SimpleSGDTrainer(model);
  sgd->eta_decay = 0.05;
  // Create the model for real
  EncoderDecoder<VanillaLSTMBuilder> lm(model,
                                 LAYERS,
                                 OUT_LAYERS,
                                 INPUT_DIM,
                                 HIDDEN_DIM,
				 PRET_SRC_DIM,
				 PRET_TGT_DIM,
                                 DROPOUT);
  cerr << "Done building the model" << endl;
  cerr << "MODEL DETAILS" << endl;
  cerr << "npsm"
     << '_' << lm.LAYERS
     << '_' << lm.OUT_LAYERS
     << '_' << lm.DROPOUT
     << '_' << lm.INPUT_DIM
     << '_' << lm.HIDDEN_DIM
     << '_' << lm.PRET_SRC_DIM
     << '_' << lm.PRET_TGT_DIM
     << endl; 

  //LSTMBuilder cands_lstm(OUT_LAYERS, INPUT_DIM, HIDDEN_DIM, model); // LSTM to encode all the candidates

  Parameter p_qkey = model.add_parameters({unsigned(4 * HIDDEN_DIM), unsigned(HIDDEN_DIM)}); 
  Parameter p_qkey_bias = model.add_parameters({unsigned(4 * HIDDEN_DIM)}); 

  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    cerr << "Loading model from: " << conf["model"].as<string>().c_str() << endl;
    assert(in);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  if (conf.count("train")) {
    assert(conf.count("training_instances"));
    assert(conf.count("dev_instances") && conf.count("dev_data"));
    assert(train_instances.size() == train_x.size());

    // track the loss
    double loss = 0.0;
    double num_instances = 0.0;

    int sent_ctr = -100;

    bool first = true;

    float best_ppl = 1e+99;

    unsigned sent = -1;
    cerr << "Training started" << endl;

    unsigned num_batches = train_x.size() / 1;
    //unsigned num_batches = train_x.size() / 25 + 1;
    cerr << "NUMBER OF BATCHES: " << num_batches << endl;

    vector<unsigned> order(train_x.size());
    for (unsigned i = 0; i < train_x.size(); ++i)
      order[i] = i;

    clock_t curr_time = clock(); 
    while(true) {
      // iterate over all training sentences 
      vector<Expression> curr_errs;
      for(unsigned i = 0; i < num_batches; ++i) {
	{
	ComputationGraph cg;
        lm.resetCG(cg, true);	
        Expression qkey = parameter(cg, p_qkey);
        Expression qkey_bias = parameter(cg, p_qkey_bias);
	for(unsigned j = 0; j < 1; ++j) { 
          ++sent;
	  ++sent_ctr;
	  if(sent == train_x.size()) {
	    sent = 0;
	    sgd->update_epoch();
	    cerr << endl << "**SHUFFLE" << endl << endl;
	    random_shuffle(order.begin(), order.end());
	    break;
	  }
        //cerr << sent << endl;
        //cerr << sent_ctr << endl;
	// Only process the sentence if there are instances associated with this particular sentence
	  if(train_instances[order[sent]].size() > 0) { 
            //cerr << sent << endl;
             unsigned slen = train_x[order[sent]].size();
             pair<vector<Expression>, Expression> enc = lm.encode(train_x[order[sent]], cg);
             assert(enc.first.size() == (train_x[order[sent]].size() + 1));
             //cerr << "CHECK" << endl;

             vector<Expression> dec = lm.decode(enc.second, train_y[order[sent]], cg);
             assert(dec.size() == (train_y[order[sent]].size() + 1));

             // make sure the target and the spans are of the same length
             assert(train_tgt[order[sent]].size() == train_spans[order[sent]].size());
       
             // now compute the embedding for all the target candidates
             vector<Expression> tgt = lm.encode_tgt(train_tgt[order[sent]], cg);
	     //cerr << "Done encoding the target" << endl;

             // now compute the embedding for all the source spans
             vector<Expression> spans = computeSpans(slen, enc.first, train_spans[order[sent]]); 
	     //cerr << "Done encoding the source spans" << endl;
	     assert(enc.first.size() == (slen + 1));
             assert(tgt.size() == spans.size());

             // now concatenate the span and the target embedding into a single vector
             vector<Expression> span_target_vec;
             for(unsigned i = 0; i < tgt.size(); ++i) {
               vector<Expression> temp;
               temp.push_back(spans[i]);
               temp.push_back(tgt[i]);
               assert(temp.size() == 2);
               span_target_vec.push_back(concatenate(temp));
             }
             assert(span_target_vec.size() == tgt.size());
             Expression span_target_cands = concatenate_cols(span_target_vec); // dim: (4 * HIDDEN_DIM, spans.size())

             vector<Expression> errs;
             unsigned instance_ctr = 0;
             Expression span_target_cands_transpose = transpose(span_target_cands);

             for (auto const& train_instance : train_instances[order[sent]]) {
                    //cerr << "instance: " << instance_ctr << endl; 
	            ++instance_ctr;
	            //cerr << "train instance: " << train_instance.first << endl;
                    assert(train_instance.second.size() > 0); // the correct answer cannot possibly be empty
                    int curr_qkey = train_instance.first;
 
                    // convert the query key (last state of the decoder before the current action) into the right dimension
                    Expression qkey_final = affine_transform({qkey_bias, qkey, dec[curr_qkey]});

                    // get the selection probability of each candidate
                    Expression u_t = span_target_cands_transpose * qkey_final; //PUT OUTSIDE THE LOOP

                   // normalize to sum to 1
                    Expression a_t = softmax(u_t); // dimension: (spans.size() / tgt.size() by 1)
	            //auto a_t_vector = as_vector(cg.incremental_forward(a_t));
	            //assert(a_t_vector.size() == spans.size());

                   // pick all the relevant candidates
                   vector<Expression> total;
                   for(unsigned j = 0; j < train_instance.second.size(); ++j) {
                     assert(train_instance.second[j] < spans.size() && train_instance.second[j] >= 0);
                     total.push_back(pick(a_t, train_instance.second[j])); 
                   }
                   assert(total.size() == train_instance.second.size());
                   errs.push_back(-log(sum(total)));
		   //cerr << "Current sentence: " << sent << endl;
	           //cerr << "Current loss: " << as_scalar(cg.forward(-log(sum(total)))) << endl;
                   curr_errs.push_back(-log(sum(total)));
             }
             assert(errs.size() == train_instances[order[sent]].size());
             //num_instances += (float) errs.size(); 
	     }
             //num_instances += (float) errs.size(); 
             //curr_errs.push_back(sum(errs));
	     //cerr << "Before forward" << endl;
             //if ((i + 1) % 25 == 0) {
             //    cerr << "Computing batch loss" << endl;
	     //    cerr << "curr_errs.size(): " << curr_errs.size() << endl;
	     //    //auto curr_errs_vec = as_vector(cg.forward(curr_errs));
	     //    printVector(curr_errs_vec);
 	     //    Expression batch_errs = sum(curr_errs);
             //    loss += as_scalar(cg.forward(batch_errs));
	     //    cerr << "Done forward" << endl;
	     //    //cerr << "After forward" << endl;
             //    cg.backward(batch_errs);
	     //    cerr << "Done backward" << endl;
             //    sgd->update();
             //    num_instances += (float) curr_errs.size(); 
	     //    cerr << "Done updating" << endl;
	     //    curr_errs.clear();
             //}
	  }
          //num_instances += (float) errs.size(); 
          //curr_errs.push_back(sum(errs));
          //cerr << "Before forward" << endl;
          //cerr << "Computing batch loss" << endl;
          //cerr << "curr_errs.size(): " << curr_errs.size() << endl;
          //auto curr_errs_vec = as_vector(cg.forward(curr_errs));
          //printVector(curr_errs_vec);
	  if(curr_errs.size() > 0) {
             Expression batch_errs = sum(curr_errs);
             loss += as_scalar(cg.forward(batch_errs));
             //cerr << "Done forward" << endl;
             //cerr << "After forward" << endl;
             cg.backward(batch_errs);
             //cerr << "Done backward" << endl;
             sgd->update();
             num_instances += (float) curr_errs.size(); 
             //cerr << "Done updating" << endl;
             curr_errs.clear();
	  }
	}
      
	   
         //cerr << "sent_ctr: " << sent_ctr << endl;
	 //cerr << "sent_ctr % 100: " << sent_ctr % 100 << endl;
         if(sent_ctr % 100 == 0)
	   {
	     //cerr << "sent_ctr: " << sent_ctr << endl;
	     cerr << "REPORTING STATUS" << endl;
	     sgd->status();
	     clock_t end_time = clock();
	     cerr << "*iter= " << sent_ctr / 100 << " E = " << (loss / num_instances) << " ppl = " << exp(loss / num_instances) << " in " << double(end_time - curr_time) / (CLOCKS_PER_SEC) << " seconds " << endl;
	     loss = 0.0;
	     curr_time = clock();
             num_instances = 0.0;
	   }

           if(sent_ctr % 2500 == 0) {
	     cerr << "Validating on the dev set" << endl; 
	     float dloss = 0.0;
	     float dnum_instances = 0.0;
	     int d_ctr = -1;
	     cerr << "d_ctr: " << d_ctr << endl;
	     cerr << "dev_x.size(): " << dev_x.size() << endl;
	     clock_t start_time = clock();
	     while (d_ctr < ( (int) dev_x.size())) {
		//cerr << "d_ctr: " << d_ctr << endl;
	        {
		ComputationGraph cg;
                lm.resetCG(cg, false);
		vector<Expression> batch_errs;
                Expression qkey = parameter(cg, p_qkey);
                Expression qkey_bias = parameter(cg, p_qkey_bias);
		for(unsigned j = 0; j < 1; ++j) {
		  ++d_ctr;
	          if (d_ctr == (int) dev_x.size()) {
		    break;
	          }
                  int sent = d_ctr;
		  //cerr << "sent: " << sent << endl;
 	          if(dev_instances[sent].size() > 0) {	
                    unsigned slen = dev_x[sent].size();
                    pair<vector<Expression>, Expression> enc = lm.encode(dev_x[sent], cg);
                    assert(enc.first.size() == (dev_x[sent].size() + 1));
       
                    vector<Expression> dec = lm.decode(enc.second, dev_y[sent], cg);
                    assert(dec.size() == (dev_y[sent].size() + 1));
       
                    // make sure the target and the spans are of the same length
                    assert(dev_tgt[sent].size() == dev_spans[sent].size());
       
                    // now compute the embedding for all the target candidates
                    vector<Expression> tgt = lm.encode_tgt(dev_tgt[sent], cg) ;
       
                    // now compute the embedding for all the source spans
                    vector<Expression> spans = computeSpans(slen, enc.first, dev_spans[sent]);
                    assert(enc.first.size() == (slen + 1));
                    assert(tgt.size() == spans.size());
       
                    // now concatenate the span and the target embedding into a single vector
                    vector<Expression> span_target_vec;
                    for(unsigned i = 0; i < tgt.size(); ++i) {
                      vector<Expression> temp;
                      temp.push_back(spans[i]);
                      temp.push_back(tgt[i]);
                      assert(temp.size() == 2);
                      span_target_vec.push_back(concatenate(temp));
                    }
                    assert(span_target_vec.size() == tgt.size());
                    Expression span_target_cands = concatenate_cols(span_target_vec); // dim: (4 * HIDDEN_DIM, spans.size())	  

		  vector<Expression> errs;
		  Expression span_target_cands_transpose = transpose(span_target_cands);
       	 	
                  for (auto const& dev_instance : dev_instances[sent]) {
                     assert(dev_instance.second.size() > 0); // the correct answer cannot possibly be empty
                     int curr_qkey = dev_instance.first;
	             //if(errs.size() > 8) assert(false);

                     // convert the query key (last state of the decoder before the current action) into the right dimension
                     Expression qkey_final = affine_transform({qkey_bias, qkey, dec[curr_qkey]});
	             //cerr << "Query key: " << endl;
	             //if(curr_qkey > 0) {
 	             //for(unsigned i = 1; i <= curr_qkey; ++i) {
	             //    cerr << td.convert(dev_y[sent][i - 1]) << " ";
	             //} }
	             //cerr << endl << "END QUERY KEY" << endl;

                     // get the selection probability of each candidate
                     Expression u_t = span_target_cands_transpose * qkey_final;

                     // normalize to sum to 1
                     Expression a_t = softmax(u_t); // dimension: (spans.size() / tgt.size() by 1)
	             //auto a_t_vector = as_vector(cg.incremental_forward(a_t));
 	             //printVector(a_t_vector);
	             //cerr << "---------------------------------------------------------------------------" << endl;

	             
                     //auto a_t_vector = as_vector(cg.incremental_forward(a_t));
                     //assert(a_t_vector.size() == spans.size());

                     // pick all the relevant candidates
                      vector<Expression> total;
                      for(unsigned j = 0; j < dev_instance.second.size(); ++j) {
	                auto indices = indicesToString(dev_tgt[d_ctr][dev_instance.second[j]], td);    
 	                auto str = joinStringVec(indices);
 	                //cerr << str << endl;
	                unsigned first = dev_spans[d_ctr][dev_instance.second[j]].first;
	                //cerr << "First: " << first << endl;
	                unsigned second = dev_spans[d_ctr][dev_instance.second[j]].second;
	                //cerr << "Second: " << second << endl;
	                //cerr << "PRINTING THE SPAN" << endl;
	                //for(unsigned k = first; k < second; ++k) {
	                //   cerr << sd.convert(dev_x[d_ctr][k]) << " ";
	                //}
	                //cerr << endl;
	                //cerr << "DONE PRINTING THE SPAN" << endl;
                        assert(dev_instance.second[j] < spans.size() && dev_instance.second[j] >= 0);
                        total.push_back(pick(a_t, dev_instance.second[j]));
                      }
                      assert(total.size() == dev_instance.second.size());
                      errs.push_back(-log(sum(total)));
                  }
	       assert(errs.size() == dev_instances[sent].size());
               batch_errs.push_back(sum(errs));
	       //cerr << "current sentence error: " << cg.incremental_forward(sum(errs)) << endl;
               //cerr << "Before forward" << endl;
               //cerr << "After forward" << endl;
               dnum_instances += (float) errs.size();
       	     }
	    }
	    if(batch_errs.size() > 0) {
	       Expression curr_errs = sum(batch_errs);
               dloss += as_scalar(cg.forward(curr_errs));
	    }
	  }
          }
          float curr_ppl = exp(dloss / dnum_instances);
          if(curr_ppl < best_ppl) {
             cerr << endl << "**new best, saving the model" << endl;
             best_ppl = curr_ppl;
	     // Save the model
             ofstream out(fname);
             boost::archive::text_oarchive oa(out);
             oa << model;
	     out.close();
          }
	  clock_t end_time = clock();

            cerr << endl << "\n***DEV [epoch=" << (float) sent_ctr / (float) train_x.size()
               << "] E = " << (dloss / dnum_instances)
               << " ppl=" << exp(dloss / dnum_instances) << ' ' << "in " << double(end_time - start_time) / (CLOCKS_PER_SEC) << " seconds" << endl << endl;
	 } // end validating the dev set
	}
	
      
    }
  }
}
