extern crate dynet;
extern crate rand;

use std::cmp::min;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, stdout, BufRead, BufReader, Write};
use std::path::Path;

use dynet::*;
use rand::{seq::SliceRandom, thread_rng};

#[derive(Debug)]
pub struct RNNLanguageModel<B> {
    p_c: LookupParameter,
    p_R: Parameter,
    p_bias: Parameter,
    builder: B,
    dropout_rate: f32,
}

impl<B: RNNBuilder> RNNLanguageModel<B> {
    pub fn new(
        model: &mut ParameterCollection,
        input_dim: u32,
        hidden_dim: u32,
        vocab_size: u32,
        dropout_rate: f32,
        builder: B,
    ) -> Self {
        RNNLanguageModel {
            p_c: model.add_lookup_parameters(
                vocab_size,
                [input_dim],
                &ParameterInitNormal::new(0., 1.),
            ),
            p_R: model.add_parameters([vocab_size, hidden_dim], &ParameterInitGlorot::default()),
            p_bias: model.add_parameters([vocab_size], &ParameterInitConst::new(0.)),
            builder,
            dropout_rate,
        }
    }

    pub fn forward(
        &mut self,
        sent: &[u32],
        cg: &mut ComputationGraph,
        apply_dropout: bool,
    ) -> Vec<Expression> {
        if apply_dropout {
            self.builder.set_dropout(self.dropout_rate);
        } else {
            self.builder.disable_dropout();
        }
        self.builder.new_graph(cg, true);
        self.builder
            .start_new_sequence::<Vec<Expression>, Expression>(vec![]);
        let R = parameter(cg, &mut self.p_R);
        let bias = parameter(cg, &mut self.p_bias);
        let mut outputs = Vec::with_capacity(sent.len() - 1);
        for t in 0..sent.len() - 1 {
            let x_t = lookup_one(cg, &mut self.p_c, sent[t]);
            let h_t = self.builder.add_input(x_t);
            let u_t = affine_transform([&bias, &R, &h_t]);
            outputs.push(u_t);
        }
        outputs
    }
}

fn make_vocab<P: AsRef<Path>>(filename: P) -> Result<HashMap<String, u32>, io::Error> {
    let reader = BufReader::new(File::open(filename.as_ref())?);
    let mut vocab = HashMap::<String, u32>::new();
    for line in reader.lines() {
        let l = format!("<s> {} </s>", line.unwrap().trim());
        for word in l.split(" ") {
            if !vocab.contains_key(word) {
                let id = vocab.len() as u32;
                vocab.insert(word.to_string(), id);
            }
        }
    }
    Ok(vocab)
}

fn load_corpus<P: AsRef<Path>>(
    filename: P,
    vocab: &HashMap<String, u32>,
) -> Result<Vec<Vec<u32>>, io::Error> {
    let reader = BufReader::new(File::open(filename.as_ref())?);
    let mut corpus = vec![];
    for line in reader.lines() {
        let l = format!("<s> {} </s>", line.unwrap().trim());
        corpus.push(l.split(" ").map(|word| vocab[word]).collect::<Vec<_>>());
    }
    Ok(corpus)
}

fn count_labels<Corpus, Sentence>(corpus: Corpus) -> usize
where
    Corpus: AsRef<[Sentence]>,
    Sentence: AsRef<[u32]>,
{
    corpus
        .as_ref()
        .iter()
        .fold(0, |sum, sent| sum + sent.as_ref().len() - 1)
}

fn main() {
    dynet::initialize(&mut DynetParams::from_args(false));

    let vocab = make_vocab("data/train.txt").unwrap();
    println!("#vocab: {}", vocab.len());

    let train_corpus = load_corpus("data/train.txt", &vocab).unwrap();
    let valid_corpus = load_corpus("data/valid.txt", &vocab).unwrap();
    let num_train_sents = train_corpus.len();
    let num_valid_sents = valid_corpus.len();
    let num_train_labels = count_labels(&train_corpus);
    let num_valid_labels = count_labels(&valid_corpus);
    println!(
        "train: {} sentences, {} labels",
        num_train_sents, num_train_labels
    );
    println!(
        "valid: {} sentences, {} labels",
        num_valid_sents, num_valid_labels
    );

    let mut m = ParameterCollection::new();
    let mut trainer = AdamTrainer::default(&mut m);
    trainer.set_clip_threshold(5.0);

    let LAYERS = 2;
    let INPUT_DIM = 256;
    let HIDDEN_DIM = 256;
    let builder = VanillaLSTMBuilder::new(LAYERS, INPUT_DIM, HIDDEN_DIM, &mut m, false, 1.);
    let mut lm = RNNLanguageModel::new(&mut m, 256, 256, vocab.len() as u32, 0.5, builder);

    let mut rng = thread_rng();

    let mut train_ids = (0..num_train_sents).collect::<Vec<_>>();
    let valid_ids = (0..num_valid_sents).collect::<Vec<_>>();

    let BATCH_SIZE = 32;
    let MAX_EPOCH = 100;
    let mut cg = ComputationGraph::new();

    for epoch in 0..MAX_EPOCH {
        println!("epoch {}/{}:", epoch + 1, MAX_EPOCH);
        train_ids.shuffle(&mut rng);

        let mut train_loss = 0.0;
        let mut ofs = 0;
        while ofs < num_train_sents {
            let batch_ids = &train_ids[ofs..min(ofs + BATCH_SIZE, num_train_sents)];
            cg.clear();
            let mut batch_loss = Vec::with_capacity(batch_ids.len());
            for sent_id in batch_ids {
                let sent = &train_corpus[*sent_id];
                let outputs = lm.forward(sent, &mut cg, true);
                batch_loss.push(sum(outputs
                    .into_iter()
                    .enumerate()
                    .map(|(t, y)| pickneglogsoftmax_one(y, sent[t + 1]))
                    .collect::<Vec<_>>()));
            }
            let loss_expr = sum(batch_loss);
            train_loss += cg.forward(&loss_expr).as_scalar();

            cg.backward(&loss_expr);
            trainer.update();

            print!("{}\r", ofs);
            stdout().flush().unwrap();
            ofs += BATCH_SIZE;
        }

        let train_ppl = (train_loss / num_train_labels as f32).exp();
        println!("  train ppl = {}", train_ppl);

        let mut valid_loss = 0.0;
        let mut ofs = 0;
        while ofs < num_valid_sents {
            let batch_ids = &valid_ids[ofs..min(ofs + BATCH_SIZE, num_valid_sents)];
            cg.clear();
            let mut batch_loss = Vec::with_capacity(batch_ids.len());
            for sent_id in batch_ids {
                let sent = &train_corpus[*sent_id];
                let outputs = lm.forward(sent, &mut cg, false);
                batch_loss.push(sum(outputs
                    .into_iter()
                    .enumerate()
                    .map(|(t, y)| pickneglogsoftmax_one(y, sent[t + 1]))
                    .collect::<Vec<_>>()));
            }
            let loss_expr = sum(batch_loss);
            valid_loss += cg.forward(&loss_expr).as_scalar();

            print!("{}\r", ofs);
            stdout().flush().unwrap();
            ofs += BATCH_SIZE;
        }

        let valid_ppl = (valid_loss / num_valid_labels as f32).exp();
        println!("  valid ppl = {}", valid_ppl);
    }
}
