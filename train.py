import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

from cnn_layers import ConvLayer, ReLU, MaxPool, Softmax


# -----------------------------------
# Load Dataset
# -----------------------------------

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


# -----------------------------------
# CNN Architecture
# -----------------------------------

conv = ConvLayer(16, 3)
relu = ReLU()
pool = MaxPool()

# 28x28 -> Conv => 26x26
# 26x26 -> Pool => 13x13
# 13x13x16 = 2704

softmax = Softmax(13 * 13 * 16, 10)


# -----------------------------------
# Forward Pass
# -----------------------------------

def forward(image, label):

    out = conv.forward(image)
    out = relu.forward(out)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])

    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


# -----------------------------------
# Training Function
# -----------------------------------

def train(image, label, lr=0.005):

    out, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    grad_back = softmax.backward(gradient, lr)
    grad_back = pool.backward(grad_back)
    grad_back = relu.backward(grad_back)

    conv.backward(grad_back, lr)

    return loss, acc


# -----------------------------------
# Train Model
# -----------------------------------

epochs = 5

epoch_loss = []
epoch_accuracy = []

print("Training Started...\n")

for epoch in range(epochs):

    total_loss = 0
    total_correct = 0

    permutation = np.random.permutation(len(train_images))

    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train on 10000 samples
    for image, label in zip(train_images[:10000], train_labels[:10000]):

        loss, acc = train(image, label)

        total_loss += loss
        total_correct += acc


    avg_loss = total_loss / 10000
    train_acc = total_correct / 10000

    epoch_loss.append(avg_loss)
    epoch_accuracy.append(train_acc)


    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.4f}\n")


# -----------------------------------
# Final Test Accuracy
# -----------------------------------

correct = 0

for image, label in zip(test_images[:5000], test_labels[:5000]):

    out, _, _ = forward(image, label)

    if np.argmax(out) == label:
        correct += 1


final_accuracy = (correct / 5000) * 100


print("=" * 35)
print(f"Test Accuracy: {final_accuracy:.2f}%")
print("=" * 35)


# -----------------------------------
# Graph 1 : Epoch vs Accuracy
# -----------------------------------

plt.figure(figsize=(7,5))

plt.plot(
    range(1, epochs+1),
    epoch_accuracy,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy")

plt.grid(True)

plt.show()



# -----------------------------------
# Graph 2 : Epoch vs Loss
# -----------------------------------

plt.figure(figsize=(7,5))

plt.plot(
    range(1, epochs+1),
    epoch_loss,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")

plt.grid(True)

plt.show()
