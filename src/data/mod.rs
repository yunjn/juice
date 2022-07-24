#![allow(unused)]
use ndarray::{Array, Axis, Dim, RemoveAxis};
use std::io::{BufRead, BufReader};
use std::{fs::File, path::Path};

pub struct LabeledDataset<D1, D2> {
    records: Array<f64, D1>,
    labels: Array<f64, D2>,
}

impl<D1: RemoveAxis, D2: RemoveAxis> LabeledDataset<D1, D2> {
    fn new(records: Array<f64, D1>, labels: Array<f64, D2>) -> Self {
        Self { records, labels }
    }

    /// Returns a reference to the records.
    pub fn records(&self) -> &Array<f64, D1> {
        &self.records
    }

    /// Returns a reference to the labels.
    pub fn labels(&self) -> &Array<f64, D2> {
        &self.labels
    }

    /// Returns the number of records stored in the labeled dataset.
    pub fn len(&self) -> usize {
        self.records.len_of(Axis(0))
    }

    /// Check whether the labeled dataset is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type L1D2 = LabeledDataset<Dim<[usize; 2]>, Dim<[usize; 1]>>;

pub struct LabeledDataLoader {}

impl LabeledDataLoader {
    pub fn from_csv(path: impl AsRef<Path>) -> L1D2 {
        let mut f = File::open(path).expect("File does not exist!");
        let br = BufReader::new(f);

        let mut labels: Vec<f64> = Vec::new();
        let mut records: Vec<_> = Vec::new();

        for line in br.lines() {
            // let mut data = Vec::new();
            let line = line.unwrap();
            let line: Vec<&str> = line.split(',').collect();

            // 二分类任务，所以将 >=5 的作为 1，<5 为 -1
            if line[0].to_string().parse::<f64>().unwrap() >= 5.0 {
                labels.push(1.0);
            } else {
                labels.push(-1.0);
            }

            line.iter()
                .skip(1)
                .for_each(|num| records.push(num.to_string().parse::<f64>().unwrap() / 255.0));
        }

        LabeledDataset::new(
            Array::from_shape_vec((labels.len(), records.len() / labels.len()), records).unwrap(),
            Array::from_vec(labels),
        )
    }
}

// #[test]
// fn test_001() {
//     let dataset = LabeledDataLoader::from_csv("assets/mnist_test.csv");
//     assert_eq!(dataset.labels().dim(), 10000);
//     assert_eq!(dataset.records().dim(), (10000, 784));
// }
