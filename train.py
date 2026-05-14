import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

from cnn import ConvLayer, ReLU, MaxPool, Softmax


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

epoch_accuracy = []

print("Training Started...\n")

for epoch in range(epochs):

    print(f"Epoch {epoch + 1}")

    permutation = np.random.permutation(len(train_images))

    train_images = train_images[permutation]

    train_labels = train_labels[permutation]

    loss = 0

    num_correct = 0

    # Train on 10000 samples
    for i, (image, label) in enumerate(
            zip(train_images[:10000], train_labels[:10000])):

        l, acc = train(image, label)

        loss += l

        num_correct += acc

        if i % 500 == 499:

            print(
                f"[Step {i+1}] "
                f"Avg Loss {loss/500:.3f} | "
                f"Accuracy {(num_correct/500)*100:.2f}%"
            )

            loss = 0

            num_correct = 0

    # -----------------------------------
    # Evaluate after each epoch
    # -----------------------------------

    correct = 0

    for image, label in zip(test_images[:2000], test_labels[:2000]):

        out, _, _ = forward(image, label)

        if np.argmax(out) == label:
            correct += 1

    accuracy = (correct / 2000) * 100

    epoch_accuracy.append(accuracy)

    print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%\n")


# -----------------------------------
# Final Evaluation
# -----------------------------------

correct = 0

for image, label in zip(test_images[:5000], test_labels[:5000]):

    out, _, _ = forward(image, label)

    if np.argmax(out) == label:
        correct += 1

final_accuracy = (correct / 5000) * 100

print("\nFinal Test Accuracy:", final_accuracy, "%")


# -----------------------------------
# Accuracy vs Epoch Graph
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    epoch_accuracy,
    marker='o'
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("Accuracy vs Epoch")

plt.grid(True)

plt.show()
