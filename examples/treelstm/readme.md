# Tree LSTM
This is a re-implementation of [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075) by Kai Sheng Tai, Richard Socher, and Christopher Manning. The model is used to handle a sentiment classification task on [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html).
A highest accuracy (root fine-grain accuracy) of 0.5213 on test set is achieved with parameters purely tuned on dev set, which is a competitive result to that claimed in the original paper(mean value 0.51). However, it should be admitted that the variance of the final result isn't small.

## Data
[Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) and [Glove word vectors](http://nlp.stanford.edu/projects/glove/) are used in this implementation. 

Pre-processed data and trained model (68M) can be accessed by
```
wget https://github.com/zhiyong1997/large-repo/raw/master/packed_data_and_model.zip
unzip packed_data_and_model.zip
```

Alternatively, you can download original files(~2G) and process them by
```
wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip trainDevTestTrees_PTB.zip 
python filter_glove.py
```
## Usage
There are three available modes:train, search and test. 

In the train mode, a group of parameters are used to train a model. The training process will stop automatically if the best accuracy on the dev set has not been improved for 10 turns, and the best model are then used for evaluation on the test set.

For example:
```
python main.py --use_glove --dropout_rate 0.3
```
In the search mode, parameters and their values can be added to 'grid_search.txt' in this format:
```
hidden_size int 100,150,200
dropout_rate float 0.3,0.5,0.7
```
All combinations of these parameters will be used for training in the way described before. A model will be saved for each group of parameters, and the best one on dev set will be used for final evaluation.
```
python main.py --mode search --use_glove
```
The saving logic is a little complex, which includes the meta file: used to specify network structure, the parameter file: used to store parameters and the embed file: used to store the fine-tuned embeddings. They are supposed to be saved at saved_models/meta/, saved_models/param/ and saved_models/embed/.

To restore a model and evaluate its performance:
```
python main.py --mode test --model_meta_file saved_models/meta/model_name
```
## Discussions 
1. The most tricky thing is that about 700 words in dev and test set do not appear in the train set, which means the trained model hasn't even seen them, and has to classify them as <__UNK__>. However, by using glove vectors, the model will have a relatively precise understanding of these unseen words, which results in great progress in the experiments. The other two famous implementations of Tree LSTM in [Tensorflow Fold](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb) and [Torch](https://github.com/stanfordnlp/treelstm) (Authoritative Implementation) both utilize this feature, so the environment is equal.
2. We use a hidden size of 150 in our provided best model, contrast to that of 300 in Tensorflow Fold implementation. However, we do not observe any improvement but a quicker overfit when turn to 300, even with the more powerful dropout mechanism mentioned in that implementation. The dropout mechanism may require further investigation.