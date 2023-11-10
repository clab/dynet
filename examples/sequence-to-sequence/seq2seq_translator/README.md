# Seq2seq Translator Benchmarks

Here is the comparison between Dynet and PyTorch on the [seq2seq translator example](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

The data we used is a set of many thousands of English to French translation pairs. Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory.

## Usage (DyNet)

The architecture of the Dynet model `seq2seq_dynet.py` is the same as that in [PyTorch Example](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). We here implement the attention mechanism in the model.

Install the GPU version of Dynet according to the instructions on the [official website](http://dynet.readthedocs.io/en/latest/python.html#installing-a-cutting-edge-and-or-gpu-version).

Then, run the training:

    python seq2seq_dynet.py --dynet_gpus 1

## Usage (PyTorch)

The code of `seq2seq_pytorch.py` follows the same line in [PyTorch Example](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). 

Install CUDA version of PyTorch according to the instructions on the [official website](http://pytorch.org/).

Then, run the training:

    python seq2seq_pytorch.py

## Performance

We run our codes on a desktop with NVIDIA TITAN X. We here have D stands for Dynet and P stands for PyTorch.

| Time (D) | Iteration (D) | Loss (D) | Time (P) | Iteration (P) | Loss (P)|
| --- | --- | --- | --- | --- | --- |
| 0m 0s | 0% | 7.9808 | 0m 0s | 0% | 7.9615 |
| 0m 28s | 5000 5% | 3.2687 | 1m 30s | 5000 5% | 2.8794 |
| 0m 56s | 10000 10% | 2.6397 | 2m 55s | 10000 10% | 2.3103 |
| 1m 25s | 15000 15% | 2.3537 | 4m 5s | 15000 15% | 1.9939 |
| 1m 54s | 20000 20% | 2.1538 | 5m 16s | 20000 20% | 1.7537 |
| 2m 22s | 25000 25% | 1.9636 | 6m 27s | 25000 25% | 1.5796 |
| 2m 51s | 30000 30% | 1.8166 | 7m 39s | 30000 30% | 1.3795 |
| 3m 20s | 35000 35% | 1.6305 | 9m 13s | 35000 35% | 1.2712 |
| 3m 49s | 40000 40% | 1.5026 | 10m 31s | 40000 40% | 1.1374 |
| 4m 18s | 45000 45% | 1.4049 | 11m 41s | 45000 45% | 1.0215 |
| 4m 47s | 50000 50% | 1.2827 | 12m 53s | 50000 50% | 0.9307 |
| 5m 17s | 55000 55% | 1.2299 | 14m 5s | 55000 55% | 0.8312 |
| 5m 46s | 60000 60% | 1.1067 | 15m 17s | 60000 60% | 0.7879 |
| 6m 15s | 65000 65% | 1.0442 | 16m 48s | 65000 65% | 0.7188 |
| 6m 44s | 70000 70% | 0.9789 | 18m 6s | 70000 70% | 0.6532 |
| 7m 13s | 75000 75% | 0.8694 | 19m 18s | 75000 75% | 0.6273 |
| 7m 43s | 80000 80% | 0.8219 | 20m 34s | 80000 80% | 0.6021 |
| 8m 12s | 85000 85% | 0.7621 | 21m 44s | 85000 85% | 0.5210 |
| 8m 41s | 90000 90% | 0.7453 | 22m 55s | 90000 90% | 0.5054 |
| 9m 10s | 95000 95% | 0.6795 | 24m 9s | 95000 95% | 0.4417 |
| 9m 39s | 100000 100% | 0.6442 | 25m 24s | 100000 100% | 0.4297 |

We then show some evaluation results as follows.

Format:

<pre>
> input 
= target 
< output
</pre>

### Dynet

```
> elle est convaincue de mon innocence .
= she is convinced of my innocence .
< she is convinced of my innocence . <EOS>

> je ne suis pas folle .
= i m not crazy .
< i m not mad . <EOS>

> je suis ruinee .
= i m ruined .
< i m ruined . <EOS>

> je ne suis certainement pas ton ami .
= i m certainly not your friend .
< i m not your best your friend . <EOS>

> c est un pleurnichard comme toujours .
= he s a crybaby just like always .
< he s a little nothing . <EOS>

> je suis sure qu elle partira tot .
= i m sure she ll leave early .
< i m sure she ll leave early . <EOS>

> vous etes toujours vivantes .
= you re still alive .
< you re still alive . <EOS>

> nous n avons pas encore tres faim .
= we aren t very hungry yet .
< we re not not desperate . <EOS>

> vous n etes pas encore morts .
= you re not dead yet .
< you re not dead yet . <EOS>

> nous sommes coinces .
= we re stuck .
< we re stuck . <EOS>
```

### PyTorch

```
> il est deja marie .
= he s already married .
< he s already married . <EOS>

> on le dit decede .
= he is said to have died .
< he are said to have died . <EOS>

> il est trop saoul .
= he s too drunk .
< he s too drunk . <EOS>

> je suis assez heureux .
= i m happy enough .
< i m happy happy . <EOS>

> je n y suis pas interessee .
= i m not interested in that .
< i m not interested in that . <EOS>

> il a huit ans .
= he s eight years old .
< he is thirty . <EOS>

> je ne suis pas differente .
= i m no different .
< i m no different . <EOS>

> je suis heureux que vous l ayez aime .
= i m happy you liked it .
< i m happy you liked it . <EOS>

> ils peuvent chanter .
= they re able to sing .
< they re able to sing . <EOS>

> vous etes tellement belle dans cette robe !
= you re so beautiful in that dress .
< you re so beautiful in that dress . <EOS>
```
