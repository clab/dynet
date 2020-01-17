extern crate dynet;
extern crate rand;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use dynet::*;
use rand::{seq::SliceRandom, thread_rng};

#[derive(Copy, Clone, Debug)]
pub enum Activation {
    Sigmoid,
    Tanh,
    Relu,
    Softmax,
    Linear,
}

impl Activation {
    fn forward<E: AsRef<Expression>>(&self, x: E) -> Expression {
        let x = x.as_ref();
        match *self {
            Activation::Sigmoid => logistic(x),
            Activation::Tanh => tanh(x),
            Activation::Relu => rectify(x),
            Activation::Softmax => softmax(x, 0),
            Activation::Linear => x.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pw: Parameter,
    pb: Parameter,
    activation: Activation,
    dropout_rate: f32,
}

impl Layer {
    pub fn new(
        in_size: u32,
        out_size: u32,
        activation: Activation,
        dropout_rate: f32,
        model: &mut ParameterCollection,
    ) -> Self {
        Self::with_initializer(
            in_size,
            out_size,
            activation,
            dropout_rate,
            model,
            &ParameterInitGlorot::default(),
        )
    }

    pub fn with_initializer<I: ParameterInit>(
        in_size: u32,
        out_size: u32,
        activation: Activation,
        dropout_rate: f32,
        model: &mut ParameterCollection,
        initializer: &I,
    ) -> Layer {
        Layer {
            pw: model.add_parameters([out_size, in_size], initializer),
            pb: model.add_parameters([out_size], &ParameterInitConst::new(0.)),
            activation,
            dropout_rate,
        }
    }

    pub fn forward<E: AsRef<Expression>>(
        &mut self,
        x: E,
        cg: &mut ComputationGraph,
        train: bool,
    ) -> Expression {
        let w = parameter(cg, &mut self.pw);
        let b = parameter(cg, &mut self.pb);
        let mut y = self
            .activation
            .forward(affine_transform([&b, &w, x.as_ref()]));
        if train && self.dropout_rate > 0. {
            y = dropout(y, self.dropout_rate)
        }
        y
    }
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(
        units: &[u32],
        out_size: u32,
        activation: Activation,
        dropout_rate: f32,
        model: &mut ParameterCollection,
    ) -> Self {
        let n_layers = units.len();
        if n_layers < 1 {
            panic!("number of layers must be greater than 0.");
        }
        MLP {
            layers: units
                .iter()
                .enumerate()
                .map(|(i, &u)| {
                    if i < n_layers - 1 {
                        Layer::new(u, units[i + 1], activation, dropout_rate, model)
                    } else {
                        Layer::new(u, out_size, Activation::Linear, 0.0, model)
                    }
                })
                .collect(),
        }
    }

    pub fn forward<E: AsRef<Expression>>(
        &mut self,
        x: E,
        cg: &mut ComputationGraph,
        train: bool,
    ) -> Expression {
        let mut y = x.as_ref().clone();
        for layer in &mut self.layers {
            y = layer.forward(y, cg, train);
        }
        y
    }
}

const NUM_TRAIN_SAMPLES: u32 = 60000;
const NUM_TEST_SAMPLES: u32 = 10000;
const NUM_INPUT_UNITS: u32 = 28 * 28;
const NUM_HIDDEN_UNITS: u32 = 512;
const NUM_OUTPUT_UNITS: u32 = 10;
const BATCH_SIZE: u32 = 200;
const NUM_TRAIN_BATCHES: u32 = NUM_TRAIN_SAMPLES / BATCH_SIZE;
const NUM_TEST_BATCHES: u32 = NUM_TEST_SAMPLES / BATCH_SIZE;
const MAX_EPOCH: u32 = 100;

fn load_images<P: AsRef<Path>>(filename: P, n: u32) -> Vec<f32> {
    let mut reader = BufReader::new(File::open(filename.as_ref()).unwrap());
    reader.seek(SeekFrom::Start(16)).unwrap();
    let size = (n * NUM_INPUT_UNITS) as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(size);
    reader.read_to_end(&mut buf).unwrap();
    let mut ret: Vec<f32> = Vec::with_capacity(size);
    for i in 0..size {
        ret.push(buf[i] as f32 / 255.0);
    }
    ret
}

fn load_labels<P: AsRef<Path>>(filename: P, n: u32) -> Vec<u8> {
    let mut reader = BufReader::new(File::open(filename.as_ref()).unwrap());
    reader.seek(SeekFrom::Start(8)).unwrap();
    let mut ret: Vec<u8> = Vec::with_capacity(n as usize);
    reader.read_to_end(&mut ret).unwrap();
    ret
}

fn main() {
    dynet::initialize(&mut DynetParams::from_args(false));

    let train_images = load_images("data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES);
    let train_labels = load_labels("data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES);
    let test_images = load_images("data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES);
    let test_labels = load_labels("data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES);

    let mut m = ParameterCollection::new();
    let mut trainer = AdamTrainer::default(&mut m);

    let mut nn = MLP::new(
        &[NUM_INPUT_UNITS, NUM_HIDDEN_UNITS],
        NUM_OUTPUT_UNITS,
        Activation::Relu,
        0.2,
        &mut m,
    );

    let mut rng = thread_rng();
    let mut ids: Vec<usize> = (0usize..NUM_TRAIN_SAMPLES as usize).collect();

    let mut cg = ComputationGraph::new();

    for epoch in 0..MAX_EPOCH {
        ids.shuffle(&mut rng);
        let mut loss = 0.;

        for batch in 0..NUM_TRAIN_BATCHES {
            print!("\rTraining... {} / {}", batch + 1, NUM_TRAIN_BATCHES);
            let mut inputs: Vec<f32> = Vec::with_capacity((BATCH_SIZE * NUM_INPUT_UNITS) as usize);
            let mut labels: Vec<u32> = vec![0; BATCH_SIZE as usize];
            for i in 0..BATCH_SIZE {
                let id = ids[(i + batch * BATCH_SIZE) as usize];
                let from = id * NUM_INPUT_UNITS as usize;
                let to = (id + 1) * NUM_INPUT_UNITS as usize;
                inputs.extend_from_slice(&train_images[from..to]);
                labels[i as usize] = train_labels[id] as u32;
            }

            cg.clear();

            let x = input(&mut cg, ([NUM_INPUT_UNITS], BATCH_SIZE), &inputs);
            let y = nn.forward(x, &mut cg, true);
            let loss_expr = sum_batches(pickneglogsoftmax(y, &labels));
            loss += cg.forward(&loss_expr).as_scalar();

            cg.backward(&loss_expr);
            trainer.update();
        }

        println!(", E = {}", loss);

        let mut correct = 0;

        for batch in 0..NUM_TEST_BATCHES {
            print!("\rTesting... {} / {}", batch + 1, NUM_TEST_BATCHES);
            let mut inputs: Vec<f32> = Vec::with_capacity((BATCH_SIZE * NUM_INPUT_UNITS) as usize);
            let from = (batch * BATCH_SIZE * NUM_INPUT_UNITS) as usize;
            let to = ((batch + 1) * BATCH_SIZE * NUM_INPUT_UNITS) as usize;
            inputs.extend_from_slice(&test_images[from..to]);

            cg.clear();

            let x = input(&mut cg, ([NUM_INPUT_UNITS], BATCH_SIZE), &inputs);
            let y = nn.forward(x, &mut cg, false);

            let y_val = cg.forward(&y).as_vector();
            for i in 0..BATCH_SIZE {
                let mut maxval = -1e10;
                let mut argmax: i32 = -1;
                for j in 0..NUM_OUTPUT_UNITS {
                    let v = y_val[(j + i * NUM_OUTPUT_UNITS) as usize];
                    if v > maxval {
                        maxval = v;
                        argmax = j as i32;
                    }
                }
                if argmax == test_labels[(i + batch * BATCH_SIZE) as usize] as i32 {
                    correct += 1;
                }
            }
        }

        let accuracy = 100.0 * correct as f32 / NUM_TEST_SAMPLES as f32;
        println!("\nepoch {}: accuracy: {:.2}%", epoch, accuracy);
    }
}
