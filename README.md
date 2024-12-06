# MLP from Scratch

This repository contains an implementation of a **Multi-Layer Perceptron (MLP)** from scratch using Python. It demonstrates the fundamental concepts of building and training a neural network, including forward propagation, backward propagation, and parameter optimization.

## Features

- **Custom implementation** of MLP without external machine learning libraries.
- Implements activation functions such as **Tanh** and **ReLU**.
- Supports **gradient computation** and backpropagation for weight updates.
- Includes training, validation, and testing phases with early stopping.
- Data preprocessing with normalization and handling missing values.

## Code Highlights

### Core Components
- **`value` Class**: Represents a scalar value with support for auto-differentiation.
- **Neuron and Layers**: Define the architecture of the MLP.
- **Training Function**: Trains the model with batch updates and monitors loss.

### Visualization
- Generates plots for:
  - Training Loss
  - Validation Loss
  - Testing Loss

## Prerequisites

- Python 3.x
- Required libraries: `numpy`, `matplotlib`, `scipy`, `graphviz`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MLP_from_Scratch.git

## Navigate to the directory 
```bash
   cd MLP_from_Scratch
```
## Install dependencies 
```bash
pip install -r requirements.txt
```

## Usage 
- Prepare the training, validation, and testing datasets.
- Configure the Model
- ```bash
  n_net = mlp(input_dim, [hidden_layers, ..., output_dim])
  ```
- Train the Model
- ```bash
  trained_net, train_loss, val_loss, test_loss = training(x_train, y_train, x_test, y_test, x_val, y_val, n_net, epochs, learning_rate)
- Visulaize the Loss
- ```bash
  plt.plot(train_loss)
  plt.title("Training Loss")
  plt.show()
  ```

 



   
