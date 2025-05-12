# Simple Artificial Neural Network for MNIST Classification (C Implementation)

This project implements a simple artificial neural network in C for classifying handwritten digits from the MNIST dataset. It includes functionalities for training the network, evaluating its performance, saving and loading the model, and making predictions on custom grayscale images.

---

## Functions Used

The following functions have been used in the code:

- `relu(float *input, int size)`: Returns 0 for every x < 0.
- `relu_derivative(float x)`: Returns 1 for x > 0 else 0.
- `softmax(float *input, int size)`: Converts logits to probabilities.
- `initialize_weights()`: Initializes weights with He initialization and biases with small constants.
- `forward(...)`: Runs a forward pass through the 3-layer neural network.
- `bubble_sort(...)`: Basic bubble sort.
- `predict_custom_image(...)`: Takes a 28x28 array as input and predicts the digit.
- `evaluate(...)`: Evaluates the trained model using MNIST test dataset.
- `load_mnist_data(...)`: Loads the MNIST dataset to train/test the neural network.
- `load_custom_image(...)`: Converts a `.raw` file to a 28x28 array for prediction.
- `save_model(...)`: Saves weights and biases to a `.bin` file.
- `load_model(...)`: Loads model from a `.bin` file if it exists.
- `train(float learning_rate)`: Performs one epoch of backpropagation.
- `get_learning_rate(int epoch)`: Applies learning rate decay.
- `file_exists(const char *filename)`: Returns 1 if file exists, else 0.

---

## Prerequisites

- A C compiler (e.g., GCC).
- MNIST dataset files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

These files should be in the same directory as the source code.

---

## Compilation

To compile:
```bash
gcc <filename.c> -o <outputname.exe> -lm
```
To predict a custom image:
```bash
./<outputname.exe> predict <rawfile.raw>
```
The -lm flag links the math library (for exp() and sqrt()).

## Running the Program
1. Training and Evaluation
If 10k.bin does not exist:

Initializes weights and biases

- Loads MNIST training set

- Trains for 100 epochs

- Saves model to 10k.bin

- Evaluates on MNIST test set and prints metrics

- Saves predictions to predictions.csv

2. Predicting a Custom Image
Ensure the image:

- Is grayscale

- Is 28x28 pixels

- Is converted to .raw (784 bytes)

Run:
```bash
./<outputname.exe> predict <path/to/your_image.raw>
```
Outputs top 3 predictions with confidence scores.

## File Structure
- main.c: Contains all logic

- *.idx3-ubyte, *.idx1-ubyte: MNIST datasets

- 10k.bin: Saved model

- predictions.csv: Predictions and metrics

## Notes
Training might take time based on your system.

Custom prediction requires a .raw binary file (784 bytes, pixel intensities 0–255).

