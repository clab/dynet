# Tree-LSTM
This is a re-implementation of the following paper:

> [**Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**](http://arxiv.org/abs/1503.00075) 
> *Kai Sheng Tai, Richard Socher, and Christopher Manning*. 

The model is suitable for sentiment classification. This provided implementation trains the model solely on dev set, and is able to achieve a highest accuracy of 0.5213 on test set, compared to the reported accuracy (mean value) 0.51 in the original paper.

## Data
[Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) and [Glove word vectors](http://nlp.stanford.edu/projects/glove/) are used in this implementation. 

The script will automatically download necessary data, including the pre-processed SST dataset and a trained model checkpoint (68M) achieving our reported testing accuracy (0.5213). 

Alternatively, you can download original data files (~2G) and process them by
```
wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip trainDevTestTrees_PTB.zip 
python filter_glove.py
```
## Usage
There are two available modes: train (default) and test.  For example:
```
python main.py --use_glove --dropout_rate 0.3
```
The above command will invoke the training using specified hyperparameters. The training process will stop automatically if the best accuracy on the dev set has not been improved for 10 turns, and the best model are then used for evaluation on the test set.

A model can be checkpointed if desired. A checkpoint contains three parts: 

 - **Meta file**: used to specify network structure at *saved_models/meta/*
 - **Parameter**: store parameters at *saved_models/param/*
 - **Embed file**: that stores the fine-tuned embeddings at *saved_models/embed/*
 
To restore a checkpoint and evaluate its performance, simply use the test mode:
```
python main.py --mode test --model_meta_file saved_models/meta/model_name
```
## Performance
- Operating System: Ubuntu 16.04
- Batch Size: 100
- LSTM Hidden Units: 150

The following "TF Fold" refers to [this implementation](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb).


| Device | Framework | Speed (time per epoch) |
| --- | --- | --- |
| GeForce GTX 1080 Ti | DyNet | 11.77 (±0.06) s |
| GeForce GTX 1080 Ti | TF Fold | 13.85 (±0.35) s |
| 3.20 GHz Intel Core i7-6900K | DyNet (single cpu)| 18.47 (±0.22) s |
| 3.20 GHz Intel Core i7-6900K | TF Fold (single cpu)| 50.70 (±0.47) s|
| 3.20 GHz Intel Core i7-6900K | TF Fold (multiple cpus: 16 available)| 18.09 (±0.18) s|


## Implementation Notes 
- In SST, there are ~700 words in dev and test set which do not appear in the training set. In order to recognize them (as <__UNK__>), Glove vectors are used to handle unseen words. The other two implementations of Tree-LSTM in [Tensorflow Fold](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb) and [Torch](https://github.com/stanfordnlp/treelstm) (Authoritative Implementation) both utilize this feature, so the environment is equal.
- We use a hidden size of 150 in our provided implementation, in contrast to that of 300 in Tensorflow Fold implementation. However, we do not observe any improvement but a quicker overfit when turn to 300, even with the more powerful dropout mechanism mentioned in that implementation. 
