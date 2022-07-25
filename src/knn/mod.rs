#![allow(unused)]
use crate::data::*;
use ndarray::prelude::*;
use ndarray::{Array, Data};
use std::cmp::Ordering;
use std::io::{BufRead, BufReader};
use std::{fs::File, path::Path};

pub struct Knn {
    logger: bool, // 记录
    topk: usize,  // 邻近点
}

pub fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
where
    S: Data,
    F: FnMut(&S::Elem, &S::Elem) -> Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}

impl LabeledDataLoader for Knn {
    fn from_csv(path: impl AsRef<Path>) -> L1D2 {
        let f = File::open(path).expect("File does not exist!");
        let br = BufReader::new(f);

        let mut labels: Vec<f64> = Vec::new();
        let mut records: Vec<_> = Vec::new();

        for line in br.lines() {
            let line = line.unwrap();
            let line: Vec<&str> = line.split(',').collect();

            labels.push(line[0].to_string().parse::<f64>().unwrap());

            line.iter()
                .skip(1)
                .for_each(|num| records.push(num.to_string().parse::<f64>().unwrap()));
        }

        LabeledDataset::new(
            Array::from_shape_vec((labels.len(), records.len() / labels.len()), records).unwrap(),
            Array::from_vec(labels),
        )
    }
}

impl Knn {
    pub fn new() -> Self {
        Self {
            logger: false,
            topk: 0,
        }
    }

    pub fn enable_log(self) -> Self {
        Self {
            logger: true,
            ..self
        }
    }

    pub fn set_topk(self, topk: usize) -> Self {
        Self { topk, ..self }
    }

    fn cal_dist(x1: &Array<f64, Ix1>, x2: &Array<f64, Ix1>) -> f64 {
        // (x1 - x2).sum() // 曼哈顿
        (x1 - x2).mapv(|a| a.powi(2)).sum().sqrt() // 欧氏距
    }

    pub fn get_closest(&self, labeled_dataset: &L1D2, x: &Array<f64, Ix1>) -> usize {
        let (m, n) = labeled_dataset.records().dim();
        let mut dist_list: Array<f64, Ix1> = Array::zeros(m);

        labeled_dataset
            .records()
            .outer_iter()
            .zip(dist_list.iter_mut())
            .for_each(|(xi, cur_dist)| *cur_dist = Knn::cal_dist(&xi.to_owned(), x));

        let top_list: Vec<usize> = argsort_by(&dist_list, |a, b| {
            a.partial_cmp(b).expect("Elements must not be NaN.")
        });

        let top_list = &top_list[..self.topk];
        let mut label_list = vec![0; 10];

        top_list.iter().for_each(|index| {
            let idx = labeled_dataset.labels().index_axis(Axis(0), *index).sum() as usize;
            label_list[idx] += 1;
        });

        label_list.iter().enumerate().max().unwrap().0
    }

    pub fn model_test(&self, train_data: L1D2, test_data: L1D2) -> f64 {
        let mut err_cnt = 0;
        let mut idx = 1;
        let (m, n) = test_data.records().dim();

        test_data
            .records()
            .outer_iter()
            .zip(test_data.labels().iter())
            .take(200)
            .for_each(|(xi, yi)| {
                if self.logger {
                    println!(
                        "test: {}/200 acc: {}",
                        idx,
                        1.0 - (err_cnt as f64 / idx as f64)
                    );
                    idx += 1;
                }

                if *yi as usize != self.get_closest(&train_data, &xi.to_owned()) {
                    err_cnt += 1;
                }
            });
        1.0 - (err_cnt as f64 / 200.0)
    }
}

#[test]
fn test_cal_dist() {
    let x1 = array![1., 2.];
    let x2 = array![2., 3.];
    assert_eq!(Knn::cal_dist(&x1, &x2), 1.4142135623730951);
}