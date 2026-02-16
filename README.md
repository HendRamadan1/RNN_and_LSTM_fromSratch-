# RNN & LSTM From Scratch

This project is a **hands-on implementation** of **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM) networks** from scratch using **Python and NumPy**, without using deep learning libraries like TensorFlow or PyTorch.  
The project is part of a course led by **Andrew**.

---

## 📝 Project Steps

### 1. Data Preparation
- Generated random input data for testing the networks.
- Defined data dimensions:
  - `x` : input sequence `(n_x, m, T_x)`
  - `a0` : initial hidden state `(n_a, m)`
  - `parameters` : dictionary containing all weights and biases of the network.

### 2. Building RNN from Scratch
- Implemented a single RNN cell (`rnn_cell_forward`) that computes:
  - Next hidden state `a_next`
  - Output `y_hat`
  - Stores a `cache` for backpropagation
- Built the forward pass across all timesteps (`rnn_forward`).

### 3. Building LSTM from Scratch
- Implemented a single LSTM cell (`lstm_cell_forward`) that computes:
  - Next hidden state `a_next`
  - Next cell state `c_next`
  - Output `y_hat`
  - Stores a cache for each timestep
- Built the forward pass across the entire sequence (`lstm_forward`).

### 4. Implementing Backpropagation
#### 4.1 Single-step LSTM Backward
- `lstm_cell_backward` computes gradients:
  - `dxt` : gradient w.r.t. input at timestep t
  - `da_prev` : gradient w.r.t. previous hidden state
  - `dc_prev` : gradient w.r.t. previous cell state
  - Gradients for weights and biases: `dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo`
- Used **Chain Rule** to propagate gradients through the LSTM gates (forget, update, output, candidate).

#### 4.2 Full-sequence LSTM Backward
- `lstm_backward` computes gradients across all timesteps.
- Aggregates gradients for weights and biases over all timesteps.
- Computes gradient for inputs and the initial hidden state `a0`.

### 5. Testing the Model
- Tested forward and backward passes using random data.
- Verified shapes of all gradient matrices:
  - `dx, da0, dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo`
- Printed sample outputs to ensure correctness.

---

## ⚡ Features
- Built the network **from scratch without any deep learning libraries**.
- Provides a deep understanding of forward and backward passes in RNNs and LSTMs.
- Explains how **gates** in LSTM control the flow of information.

---

## 📌 Requirements
- Python 3.x
- NumPy

---

## 💡 Notes
- This project is a **course-based hands-on application**, with step-by-step calculations of gradients.
- The code can be extended to larger applications like text processing or time-series prediction.

---
