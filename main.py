 
# MNIST HANDWRITTEN DIGIT CLASSIFICATION
# TensorFlow / Keras
 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


 
# 1. LOAD AND EXPLORE MNIST DATASET
 

print("TensorFlow Version:", tf.__version__)

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\nDataset Information")
print("-------------------")
print("Training images :", x_train.shape)
print("Training labels :", y_train.shape)
print("Testing images  :", x_test.shape)
print("Testing labels  :", y_test.shape)

print("\nPixel value range before normalization:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())


 
# DISPLAY SAMPLE IMAGES
 

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.suptitle("Sample MNIST Images")
plt.tight_layout()
plt.show()


 
# CHECK CLASS DISTRIBUTION
 

unique, counts = np.unique(y_train, return_counts=True)

print("\nClass Distribution")
print("------------------")

for digit, count in zip(unique, counts):
    print(f"Digit {digit}: {count} images")


 
# NORMALIZE DATA
 

# Convert pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nPixel value range after normalization:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())


 
# 2. BUILD THE NEURAL NETWORK
 

model = tf.keras.Sequential([

    # Input image: 28 x 28
    tf.keras.layers.Input(shape=(28, 28)),

    # Convert 28x28 image into 784 values
    tf.keras.layers.Flatten(),

    # Hidden layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    # 10 neurons = digits 0 to 9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


 
# DISPLAY MODEL ARCHITECTURE
 

print("\nOriginal Model Architecture")
print("---------------------------")

model.summary()


 
# COMPILE MODEL
 

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


 
# 3. TRAIN THE MODEL
 

print("\nTraining Original Model...")
print("--------------------------")

history = model.fit(
    x_train,
    y_train,

    # Number of times the complete dataset is used
    epochs=10,

    # Number of images processed at once
    batch_size=32,

    # 20% of training data used for validation
    validation_split=0.2
)


 
# EVALUATE ORIGINAL MODEL
 

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nOriginal Model Results")
print("----------------------")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


 
# 4. VISUALIZE TRAINING AND VALIDATION ACCURACY
 

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()


 
# VISUALIZE TRAINING AND VALIDATION LOSS
 

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()


 
# 5. TEST MODEL ON 5 HANDWRITTEN IMAGES
 

# Select 5 images from the MNIST test dataset
sample_images = x_test[:5]
actual_labels = y_test[:5]

# Make predictions
predictions = model.predict(
    sample_images,
    verbose=0
)

# Convert probabilities to predicted digit
predicted_labels = np.argmax(
    predictions,
    axis=1
)


 
# DISPLAY 5 IMAGES WITH ACTUAL AND PREDICTED LABELS
 

plt.figure(figsize=(12, 4))

for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(
        sample_images[i],
        cmap="gray"
    )

    plt.title(
        f"Actual: {actual_labels[i]}\n"
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.suptitle("Actual vs Predicted Labels")
plt.tight_layout()
plt.show()


 
# PRINT PREDICTION TABLE
 

print("\nPrediction Results")
print("------------------")

print("Image\tActual\tPredicted\tResult")

for i in range(5):

    result = (
        "Correct"
        if actual_labels[i] == predicted_labels[i]
        else "Wrong"
    )

    print(
        f"{i + 1}\t"
        f"{actual_labels[i]}\t"
        f"{predicted_labels[i]}\t\t"
        f"{result}"
    )


 
# 6. EXPERIMENT
 
# Experiment: Add Dropout
#
# Original:
# Flatten -> Dense(128) -> Dense(10)
#
# New:
# Flatten -> Dense(128) -> Dropout(0.2) -> Dense(10)
#
# Dropout randomly disables 20% of neurons during training.
# This can help reduce overfitting.


print("\n\nExperiment: Adding Dropout")
print("===========================")


 
# CREATE EXPERIMENTAL MODEL
 

dropout_model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout experiment
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


 
# COMPILE DROPOUT MODEL
 

dropout_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


 
# TRAIN DROPOUT MODEL
 

print("\nTraining Dropout Model...")
print("-------------------------")

dropout_history = dropout_model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


 
# EVALUATE DROPOUT MODEL
 

dropout_loss, dropout_accuracy = dropout_model.evaluate(
    x_test,
    y_test,
    verbose=0
)


 
# COMPARE ORIGINAL AND DROPOUT MODEL
 

print("\n\nMODEL COMPARISON")
print("================")

print(
    f"Original Model Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Dropout Model Test Accuracy  : "
    f"{dropout_accuracy * 100:.2f}%"
)

print(
    f"\nOriginal Model Best "
    f"Validation Accuracy: "
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)

print(
    f"Dropout Model Best "
    f"Validation Accuracy: "
    f"{max(dropout_history.history['val_accuracy']) * 100:.2f}%"
)


 
# COMPARE VALIDATION ACCURACY
 

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["val_accuracy"],
    label="Original Model"
)

plt.plot(
    dropout_history.history["val_accuracy"],
    label="Dropout Model"
)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")

plt.legend()
plt.grid(True)

plt.show()


 
# COMPARE VALIDATION LOSS
 

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["val_loss"],
    label="Original Model"
)

plt.plot(
    dropout_history.history["val_loss"],
    label="Dropout Model"
)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

plt.legend()
plt.grid(True)

plt.show()


 
# FINAL SUMMARY
 

print("\n")
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

print(
    f"Original Model Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Dropout Model Accuracy  : "
    f"{dropout_accuracy * 100:.2f}%"
)

print("=" * 50)
