# UE24CS645BC2_PES1PG25CS046_Fashion_MNIST_CNN
Project Name: Fashion MNIST CNN

The CNN is trained on the Fashion MNIST dataset.

# Fashion-MNIST CNN from Scratch using NumPy

## Project Overview

This project implements a **Convolutional Neural Network (CNN) from scratch using NumPy** for Fashion-MNIST image classification without using built-in deep learning layers.

The CNN consists of:

- Convolution Layer
- ReLU Activation
- Max Pooling Layer
- Softmax Classification Layer

The model is trained on the Fashion-MNIST dataset and evaluated using accuracy and loss metrics.

---

## Dataset

Dataset Used:

**Fashion-MNIST**

Contains grayscale images of clothing items:

Classes:

0 → T-shirt/top  
1 → Trouser  
2 → Pullover  
3 → Dress  
4 → Coat  
5 → Sandal  
6 → Shirt  
7 → Sneaker  
8 → Bag  
9 → Ankle boot  

Image Size:

28 × 28 pixels

Training samples:

60,000 images

Testing samples:

10,000 images

Dataset loading:

```python
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

---

# CNN Architecture

Input:

28 × 28 grayscale image

↓

Convolution Layer

- Filters = 16
- Filter Size = 3 × 3

Output:

26 × 26 × 16

↓

ReLU Activation

↓

Max Pooling (2 × 2)

Output:

13 × 13 × 16

↓

Flatten

Output:

2704 neurons

↓

Softmax Layer

Output:

10 classes

---

# Project Structure

```text
project/

│
├── cnn_layers.py
│       ConvLayer
│       ReLU
│       MaxPool
│       Softmax
│
├── train.py
│       Training code
│
└── README.md
```

---

# Features Implemented

Custom implementation of:

✓ Convolution operation  
✓ ReLU activation  
✓ Max Pooling  
✓ Softmax classifier  
✓ Forward propagation  
✓ Backpropagation  
✓ Gradient descent updates  
✓ Accuracy calculation  
✓ Loss calculation  
✓ Accuracy vs Epoch graph  
✓ Loss vs Epoch graph

---

# Training Configuration

Epochs:

```text
5
```

Learning Rate:

```text
0.005
```

Filters:

```text
16
```

Training Samples:

```text
10000
```

Testing Samples:

```text
5000
```

---

# Performance Results

Example training output:

```text
Epoch 1/5
Loss: 54.6949
Training Accuracy: 0.1307

Epoch 2/5
Loss: 53.3006
Training Accuracy: 0.2637

Epoch 3/5
Loss: 50.5705
Training Accuracy: 0.4097

Epoch 4/5
Loss: 45.7423
Training Accuracy: 0.5313

Epoch 5/5
Loss: 39.1364
Training Accuracy: 0.5863


===================================
Test Accuracy: 62.60%
===================================
```

---

# Accuracy Analysis

Training Accuracy:

| Epoch | Accuracy |
|------|-----------|
|1|13.07%|
|2|26.37%|
|3|40.97%|
|4|53.13%|
|5|58.63%|

Final Test Accuracy:

```text
62.60%
```

Observation:

Training accuracy increases with epochs, showing the model is learning meaningful image features.

---

# Loss Analysis

| Epoch | Loss |
|------|------|
|1|54.69|
|2|53.30|
|3|50.57|
|4|45.74|
|5|39.13|

Observation:

Loss decreases over epochs indicating improvement in model prediction.

---

# Graphs Generated

## 1. Epoch vs Accuracy

Shows improvement in classification accuracy during training.

Expected trend:

Increasing curve ↑

---

## 2. Epoch vs Loss

Shows reduction in prediction error.

Expected trend:

Decreasing curve ↓

---

# How to Run

Install dependencies:

```bash
pip install numpy
pip install matplotlib
pip install tensorflow
```

Run training:

```bash
python train.py
```

---

# Output Generated

Program outputs:

- Training loss
- Training accuracy
- Final test accuracy
- Epoch vs Accuracy graph
- Epoch vs Loss graph

---

# Future Improvements

Possible improvements:

- Increase epochs
- Increase number of filters
- Add additional convolution layers
- Add dropout
- Use batch training
- Implement Adam optimizer
- Improve accuracy to 85–90%

---

# Technologies Used

Python  
NumPy  
Matplotlib  
TensorFlow (Dataset only)

---

# Conclusion

This project demonstrates how CNNs work internally by implementing convolution, activation, pooling, and softmax layers manually using NumPy. The model achieved approximately **62.6% test accuracy** on Fashion-MNIST while showing consistent reduction in loss and improvement in classification accuracy over epochs.

