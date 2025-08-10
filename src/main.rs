use rand::Rng;
// Use the mathematical constant E for our sigmoid function.
use std::f64::consts::E;

/// The sigmoid activation function.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + E.powf(-z))
}

/// The Neuron struct holds the neuron's weights and its bias.
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
        
    pub fn new(num_inputs: usize) -> Self {
        let mut rng = rand::rng();

        let weights = (0..num_inputs)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let bias = rng.random_range(-1.0..1.0);
        
        Self { weights, bias }
    }

    pub fn feed_forward(&self, inputs: &[f64]) -> f64 {
        if inputs.len() != self.weights.len() {
            panic!("Input vector length must match number of weights.");
        }

        let z: f64 = self.weights.iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum();

        sigmoid(z + self.bias)
    }

    /// Calculates the cost (mean squared error) for a given input and desired output.
    pub fn cost(&self, inputs: &[f64], desired_output: f64) -> f64 {
        let predicted_output = self.feed_forward(inputs);
        (desired_output - predicted_output) * (desired_output - predicted_output)
    }
    
    /// Provides a binary prediction (0 or 1) based on the neuron's output.
    pub fn predict(&self, inputs: &[f64]) -> f64 {
        let output = self.feed_forward(inputs);
        if output > 0.5 {
            1.0
        } else {
            0.0
        }
    }

    /// Trains the neuron using the backpropagation algorithm.
    pub fn train(&mut self, inputs: &[f64], desired_output: f64, learning_rate: f64) {
        // Step 1: Get the neuron's current prediction.
        let predicted_output = self.feed_forward(inputs);

        // Step 2: Calculate the derivative of the cost function.
        let cost_derivative = 2.0 * (predicted_output - desired_output);

        // Step 3: Calculate the derivative of the sigmoid function.
        let sigmoid_derivative = predicted_output * (1.0 - predicted_output);
        
        // Step 4: Combine the derivatives to get the backpropagation error.
        let backprop_error = cost_derivative * sigmoid_derivative;

        // Step 5: Update the weights and bias.
        
        for i in 0..self.weights.len() {
            let weight_change = learning_rate * backprop_error * inputs[i];
            self.weights[i] -= weight_change;
        }

        let bias_change = learning_rate * backprop_error;
        self.bias -= bias_change;
    }
}

fn main() {
    let mut last_brain_cell = Neuron::new(2);

    // Training data for a logical gate.
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let outputs = vec![0.0, 0.0, 0.0, 1.0];
    let epochs = 1000;

    println!("First set of predictions (before training):");
    for i in inputs.iter() {
        println!("input [{} {}] : predicted output : [{}]", i[0], i[1], last_brain_cell.predict(i));
    }

    println!("\nTraining for {} epochs...", epochs);
    for _ in 0..epochs {
        for (i, o) in inputs.iter().zip(outputs.iter()) {
            last_brain_cell.train(i, *o, 0.1);
        }
    }

    println!("\nSecond set of predictions (after training):");
    for i in inputs.iter() {
        println!("input [{} {}] : predicted output : [{}]", i[0], i[1], last_brain_cell.predict(i));
    }
    
    println!("\nFinal weights: {:?}", last_brain_cell.weights);
    println!("Final bias: {}", last_brain_cell.bias);
}

