<div align="center">

## 一些 DL 算法实现

</div>

### Quick Start

* Single Layer Perceptron 

 ```Rust
mod single_layer_perceptron;

use single_layer_perceptron::*;

fn main() {
    let x = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0],
    ];

    let y = vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];

    let model = SingleLayerPerceptron::new()
        .set_epoch(100)
        .set_learning_rate(0.01)
        .set_baise(0.3)
        .load_dataset(DataSet { x, y })
        .enable_log()
        .train();

    let input = vec![0.0, 1.0, 1.0];
    let predict = model.predict(&input);
    println!("input: {:?} predict: {}", input, predict);
}
 ```

* Perceptron

```Rust
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
```

* KNN

```Rust
let train_data = Knn::from_csv("assets/mnist_train.csv");
let test_data = Knn::from_csv("assets/mnist_test.csv");
let acc = Knn::new()
    .enable_log()
    .set_topk(25)
    .model_test(train_data, test_data);
println!("{}", acc);
println!("hello juice");
```

* Test

```shell
cargo test  --release --  --show-output
```

### Implemented

- [x] Single Layer Perceptron
- [x] Perceptron
- [x] KNN