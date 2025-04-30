# 🧠 COVID-19 X-ray Classification with Standard Neural Network

This project explores a **fully connected neural network from scratch** to classify chest X-ray images for the presence of COVID-19. The goal is not only performance but to gain a **deep understanding of how neural networks work at the matrix level** — specifically **forward propagation, backpropagation, and gradient descent optimization**. It uses hyperparameter tuning to improve generalization and reduce overfitting, even at the cost of training accuracy.


## 🧠 Project Motivation

Most neural network libraries abstract away the math behind deep learning. In this project, I chose to:

- **Build the neural network entirely from scratch**, using only basic libraries like NumPy and SciPy.
- Focus on **matrix operations** to implement each layer’s computation manually.
- Learn and properly implement **forward propagation**, **backward propagation**, **weight and bias updates**, and **loss functions**.

This hands-on approach strengthens theoretical understanding while highlighting practical challenges like overfitting and generalization.



## 🚀 Project Summary

- **Task**: Binary classification (COVID-19 vs. non-COVID) using chest X-ray images.
- **Architecture**: Fully connected neural network.
- **Framework**: Implemented in Python using libraries such as `NumPy`, `SciPy`, `PIL`, and `Matplotlib`.

## 🔧 Features

- Pure Python implementation using `numpy` for all tensor operations
- Custom training loop with:
  - Batch-wise updates
  - Manual loss and accuracy tracking
  - Validation performance monitoring
- Hyperparameter tuning engine
- Dynamic learning rate adjustment with patience and rollback


## 🧪 Methodology

### ⚙️ Hyperparameter Tuning

To improve generalization and reduce overfitting, the following strategy is used:

- **Learning rate decay**: Starts with a learning rate between `0.1` and `0.0001`.
- **Patience strategy**: Each learning rate is given **100 iterations** (patience) to prove it can reduce **validation loss**.
- If the model fails to reduce validation loss within those 100 iterations:
  - The learning rate is reduced by a **factor of 0.95**.
  - The model is **restored to the previous best weights and biases**.
- This approach ensures:
  - The model avoids moving in a direction that worsens validation loss.
  - Emphasis is placed on **learning in the right direction**, rather than merely improving training performance.

### 📉 Trade-off Strategy

- **Training accuracy may be sacrificed** to achieve better **generalization on the validation/testing set**.
- This is critical in medical imaging, where **overfitting to training data** can be harmful in real-world deployment.

## 📁 File Structure
    .
    ├── x_train.npy                    # Training images (preprocessed)
    ├── y_train.npy                    # Training labels
    ├── x_test.npy                     # Test images (preprocessed)
    ├── y_test.npy                     # Test labels
    │
    ├── Standard_Neural_Network.ipynb  # Main notebook: model, training, evaluation
    ├── README.md                      # Project overview and documentation
    ├── requirements.txt               # (Optional) Dependencies list for pip install


## 🛠 Installation

Install the required Python libraries:

```bash
pip install numpy scipy pillow matplotlib
```

## 🚀 Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. Open the notebook:

```bash
jupyter notebook Standard_Neural_Network.ipynb
```

3. Follow the steps in the notebook to:
    - Load and preprocess data

    - Initialize and train the neural network

    - Monitor validation loss and test performance

## 📊 Results

> The notebook includes detailed metrics, plots, and performance comparisons during different learning rate

# COVID-19-X-ray-Classification-
