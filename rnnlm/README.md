### Obtaining LM data

    ./install-examples.sh

This downloads the data used in this tutorial.

### Training example

Train an LSTM LM using a class-factor softmax:

    ./rnnlm/lm -x -s -t ../rnnlm/ptb-mikolov/train.txt -d ../rnnlm/ptb-mikolov/valid.txt \
         -c ../rnnlm/ptb-mikolov/clusters-mkcls.txt -D 0.3 -H 256 --eta_decay_onset_epoch 10 --eta_decay_rate 0.5

Train an LSTM LM with a standard softmax:

    ./rnnlm/lm -x -s -t ../rnnlm/ptb-mikolov/train.txt -d ../rnnlm/ptb-mikolov/valid.txt \
         -D 0.3 -H 256 --eta_decay_onset_epoch 10 --eta_decay_rate 0.5

### Evaluation example

Evaluate a trained model:

    ./rnnlm/lm -t ../rnnlm/ptb-mikolov/train.txt -c ../rnnlm/ptb-mikolov/clusters-mkcls.txt \
         -m lm_0.3_2_128_256-pid7865.params -H 256 -p ../rnnlm/ptb-mikolov/test.txt

### PTB Baselines

| Model | dev | test |
| ----- | ---:| ----:|
| 5-gram KN | 188.0 | 178.9 |
| 2x128, dropout=0.3, class-factored softmax | 164.4 | 157.7 |
| 2x256, dropout=0.3, CFSM, decay 0.5@>10 | 129.7 | 125.4 |
