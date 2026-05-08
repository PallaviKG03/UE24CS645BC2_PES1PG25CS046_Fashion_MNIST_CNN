import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add channel dimension
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Label names
class_names = [
    'T-shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
]

# Build CNN model
model = models.Sequential()

# Convolution Layer
model.add(layers.Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))

# MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolution Layer
model.add(layers.Conv2D(
    64,
    (3, 3),
    activation='relu'
))

# Second MaxPooling
model.add(layers.MaxPooling2D((2, 2)))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Layer
model.add(layers.Dense(64, activation='relu'))

# Output Layer
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

print("\nStarting Training...\n")

# Train model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

print("\nTesting Model...\n")

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTest Accuracy:", test_acc)
print("Test Loss:", test_loss)

# Predict one image
predictions = model.predict(test_images)

predicted_label = predictions[0].argmax()

print("\nPredicted Label:", class_names[predicted_label])
print("Actual Label:", class_names[test_labels[0]])