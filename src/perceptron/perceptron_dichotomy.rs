#![allow(unused)]

use crate::data::*;
use ndarray::prelude::*;
use ndarray::{Array, Axis, RemoveAxis};

pub struct Perceptron {
    logger: bool,             // 记录
    epochs: usize,            // 代
    step_size: f64,           // 步长
    baise: f64,               // 偏置
    weights: Array<f64, Ix2>, // 权值
}

impl Perceptron {
    pub fn new() -> Self {
        Self {
            logger: false,
            epochs: 10,
            step_size: 0.001,
            baise: 0.0,
            weights: Array::default((0, 0)),
        }
    }

    pub fn enable_log(self) -> Self {
        Self {
            logger: true,
            ..self
        }
    }

    pub fn set_epochs(self, epochs: usize) -> Self {
        Self { epochs, ..self }
    }

    pub fn set_step_size(self, step_size: f64) -> Self {
        Self { step_size, ..self }
    }

    pub fn train(mut self, labeled_dataset: L1D2) -> Self {
        let (m, n) = labeled_dataset.records().dim();
        let mut weights: Array<f64, Ix2> = Array::zeros((1, n));

        for epoch in 1..=self.epochs {
            for (xi, yi) in labeled_dataset
                .records()
                .outer_iter()
                .zip(labeled_dataset.labels().iter())
            {
                if (-1.0 * yi * (&weights * &xi.t() + self.baise)).sum() >= 0.0 {
                    weights = weights + self.step_size * yi * &xi.t();
                    self.baise += self.step_size * yi;
                }
            }

            if self.logger {
                println!("Epoch: {}/{}", epoch, self.epochs);
            }
        }
        Self { weights, ..self }
    }

    pub fn test(self, labeled_dataset: L1D2) -> f64 {
        let (m, n) = labeled_dataset.records().dim();
        let mut err_cnt = 0.0;
        for (xi, yi) in labeled_dataset
            .records()
            .outer_iter()
            .zip(labeled_dataset.labels().iter())
        {
            if (-1.0 * yi * (&self.weights * &xi.t() + self.baise)).sum() >= 0.0 {
                err_cnt += 1.0;
            }
        }
        1.0 - err_cnt / m as f64
    }
}

#[test]
fn test_perceptron() {
    let train_data = LabeledDataLoader::from_csv("assets/mnist_train.csv");
    let test_data = LabeledDataLoader::from_csv("assets/mnist_test.csv");

    let model = Perceptron::new()
        .set_epochs(50)
        .set_step_size(0.0001)
        .enable_log()
        .train(train_data);

    let acc = model.test(test_data);
    println!("acc: {}", acc);
}
