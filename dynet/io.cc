#include "dynet/io.h"
#include "dynet/tensor.h"

using namespace std;
using namespace dynet;

TextFileSaver::TextFileSaver(const string & filename, bool append) :
        datastream(filename, append ? ofstream::app : ofstream::out) {
  if(!datastream)
    DYNET_RUNTIME_ERR("Could not write model to " << filename);
}

void TextFileSaver::save(const ParameterCollection & model,
                         const string & key) {
  const ParameterCollectionStorage & storage = model.get_storage();
  if(key.size() == 0) {
    for (auto & p : storage.params) save(*p, key);
    for (auto & p : storage.lookup_params) save(*p, key);
  } else {
    size_t strip_size = model.get_fullname().size();
    for (auto & p : storage.params) 
      save(*p, key + p->name.substr(strip_size));
    for (auto & p : storage.lookup_params) 
      save(*p, key + p->name.substr(strip_size));
  }
}

void TextFileSaver::save(const Parameter & param,
                         const string & key) {
  save(*param.p, key);
}

void TextFileSaver::save(const LookupParameter & param,
                         const string & key) {
  save(*param.p, key);
}

void TextFileSaver::save(const ParameterStorage & p,
                         const string & key) {
  datastream << "#Parameter# " << (key.size() > 0 ? key : p.name) << ' ' << p.dim << endl;
  datastream << dynet::as_vector(p.values) << endl;
  datastream << dynet::as_vector(p.g) << endl;
}

void TextFileSaver::save(const LookupParameterStorage & p,
                         const string & key) {
  datastream << "#LookupParameter# " << (key.size() > 0 ? key : p.name) << ' ' << p.all_dim << ' ' << endl;
  datastream << dynet::as_vector(p.all_values) << endl;
  datastream << dynet::as_vector(p.all_grads) << endl;
}

void TextFileLoader::populate(ParameterCollection & model, const string & key) {
  ifstream datastream(dataname);
  if(!datastream) DYNET_RUNTIME_ERR("Could not read model from " << dataname);
  string line, type, name;
  Dim dim;
  vector<float> values;
  Tensor *value_t, *grad_t;
  while(getline(datastream, line)) {
    { istringstream iss(line); iss >> type >> name >> dim; }
    // Skip ones that don't match
    if(key.size() != 0 && name.substr(0, key.size()) != key) {
      getline(datastream, line);
      getline(datastream, line);
    // Load a parameter
    } else if(type == "#Parameter#") {
      values.resize(dim.size());
      Parameter param = model.add_parameters(dim);
      param.get_storage().name = name;
      value_t = &param.get_storage().values;
      grad_t = &param.get_storage().g;
    // Load a lookup parameter
    } else if(type == "#LookupParameter#") {
      values.resize(dim.size());
      size_t num = dim[dim.nd-1];
      --dim.nd;
      LookupParameter param = model.add_lookup_parameters(num, dim);
      param.get_storage().name = name;
      value_t = &param.get_storage().all_values;
      grad_t = &param.get_storage().all_grads;
    } else {
      DYNET_RUNTIME_ERR("Bad parameter specification in model: " << line);
    }
    { getline(datastream, line); istringstream iss(line); iss >> values; }
    TensorTools::SetElements(*value_t, values);
    { getline(datastream, line); istringstream iss(line); iss >> values; }
    TensorTools::SetElements(*grad_t, values);
  }
}

void TextFileLoader::populate(Parameter & param,
                    const string & key) {
  if(key == "")
    DYNET_INVALID_ARG("TextFileLoader.populate() requires non-empty key");
  ifstream datastream(dataname);
  if(!datastream) DYNET_RUNTIME_ERR("Could not read model from " << dataname);
  string line, type, name;
  Dim dim;
  while(getline(datastream, line)) {
    { istringstream iss(line); iss >> type >> name >> dim; }
    if(type == "#Parameter#" && name == key) {
      if(param.p->dim != dim)
        DYNET_RUNTIME_ERR("Attempted to populate parameter where arguments don't match (" << param.p->dim << " != " << dim << ")");
      param.get_storage().name = name;
      vector<float> values(dim.size());
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(param.get_storage().values, values);
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(param.get_storage().g, values);
      return;
    } else {
      getline(datastream, line);
      getline(datastream, line);
    }
  }
  DYNET_RUNTIME_ERR("Could not find key " << key << " in the model file");
}

void TextFileLoader::populate(LookupParameter & lookup_param,
                              const string & key) {
  if(key == "")
    DYNET_INVALID_ARG("TextFileLoader.populate() requires non-empty key");
  ifstream datastream(dataname);
  if(!datastream) DYNET_RUNTIME_ERR("Could not read model from " << dataname);
  string line, type, name;
  Dim dim;
  while(getline(datastream, line)) {
    { istringstream iss(line); iss >> type >> name >> dim; }
    if(type == "#LookupParameter#" && name == key) {
      if(lookup_param.p->all_dim != dim)
        DYNET_RUNTIME_ERR("Attempted to populate lookup parameter where arguments don't match (" << lookup_param.p->all_dim << " != " << dim << ")");
      lookup_param.get_storage().name = name;
      vector<float> values(dim.size());
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(lookup_param.get_storage().all_values, values);
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(lookup_param.get_storage().all_grads, values);
      return;
    } else {
      getline(datastream, line);
      getline(datastream, line);
    }
  }
  DYNET_RUNTIME_ERR("Could not find key " << key << " in the model file");
}

Parameter TextFileLoader::load_param(ParameterCollection & model,
                                     const string & key) {
  if(key == "")
    DYNET_INVALID_ARG("TextFileLoader.load_param() requires non-empty key");
  ifstream datastream(dataname);
  if(!datastream) DYNET_RUNTIME_ERR("Could not read model from " << dataname);
  string line, type, name;
  Dim dim;
  while(getline(datastream, line)) {
    { istringstream iss(line); iss >> type >> name >> dim; }
    if(type == "#Parameter#" && name == key) {
      Parameter param = model.add_parameters(dim);
      param.get_storage().name = name;
      vector<float> values(dim.size());
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(param.get_storage().values, values);
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(param.get_storage().g, values);
      return param;
    } else {
      getline(datastream, line);
      getline(datastream, line);
    }
  }
  DYNET_RUNTIME_ERR("Could not find key " << key << " in the model file");
  return this->deserialize_param(model, key);
}

LookupParameter TextFileLoader::load_lookup_param(ParameterCollection & model,
                                                  const string & key) {
  if(key == "")
    DYNET_INVALID_ARG("TextFileLoader.load_lookup_param() requires non-empty key");
  ifstream datastream(dataname);
  if(!datastream) DYNET_RUNTIME_ERR("Could not read model from " << dataname);
  string line, type, name;
  Dim dim;
  while(getline(datastream, line)) {
    { istringstream iss(line); iss >> type >> name >> dim; }
    if(type == "#LookupParameter#" && name == key) {
      vector<float> values(dim.size());
      size_t size = dim[dim.nd-1]; dim.nd--;
      LookupParameter lookup_param = model.add_lookup_parameters(size, dim);
      lookup_param.get_storage().name = name;
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(lookup_param.get_storage().all_values, values);
      { getline(datastream, line); istringstream iss(line); iss >> values; }
      TensorTools::SetElements(lookup_param.get_storage().all_grads, values);
      return lookup_param;
    } else {
      getline(datastream, line);
      getline(datastream, line);
    }
  }
  DYNET_RUNTIME_ERR("Could not find key " << key << " in the model file");
}

TextFileLoader::TextFileLoader(const string & filename) :
        dataname(filename) { }
