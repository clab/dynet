extern crate dynet;

use std::env;

use dynet::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    dynet::initialize(&mut DynetParams::from_args(false));

    let ITERATIONS = 30;

    let mut m = ParameterCollection::new();
    let mut trainer = SimpleSGDTrainer::default(&mut m);

    let HIDDEN_SIZE = 8;
    let initializer = ParameterInitGlorot::default();
    let mut p_W = m.add_parameters([HIDDEN_SIZE, 2], &initializer);
    let mut p_b = m.add_parameters([HIDDEN_SIZE], &initializer);
    let mut p_V = m.add_parameters([1, HIDDEN_SIZE], &initializer);
    let mut p_a = m.add_parameters([1], &initializer);
    if args.len() == 2 {
        m.load(&args[1]);
    }

    let mut cg = ComputationGraph::new();
    for iter in 0..ITERATIONS {
        let mut loss = 0.;
        for mi in 0..4 {
            let x1 = mi % 2 == 1;
            let x2 = (mi / 2) % 2 == 1;
            let x_values = [if x1 { 1. } else { -1. }, if x2 { 1. } else { -1. }];
            let y_value = if x1 != x2 { 1. } else { -1. };

            cg.clear();
            let W = parameter(&mut cg, &mut p_W);
            let b = parameter(&mut cg, &mut p_b);
            let V = parameter(&mut cg, &mut p_V);
            let a = parameter(&mut cg, &mut p_a);
            let x = input(&mut cg, [2], &x_values);
            let y = input_scalar(&mut cg, y_value);
            let h = tanh(W * x + b);
            let y_pred = V * h + a;
            let loss_expr = squared_distance(y_pred, y);

            if iter == 0 && mi == 0 {
                cg.print_graphviz();
            }

            loss += cg.forward(&loss_expr).as_scalar();
            cg.backward(&loss_expr);
            trainer.update();
        }
        loss /= 4.;
        println!("E = {}", loss);
    }
    m.save("xor.model");
}
