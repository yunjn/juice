mod data;
mod knn;
mod perceptron;

use data::*;
use knn::*;

fn main() {
    let train_data = Knn::from_csv("assets/mnist_train.csv");
    let test_data = Knn::from_csv("assets/mnist_test.csv");
    let acc = Knn::new()
        .enable_log()
        .set_topk(25)
        .model_test(train_data, test_data);
    println!("{}", acc);
    println!("hello juice");
}
