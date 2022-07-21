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

### Implemented

- [x] Single Layer Perceptron