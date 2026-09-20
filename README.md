# MNIST Handwritten Digit Classification

A deep learning project that classifies handwritten digits from **0 to 9** using the **MNIST dataset** and a neural network built with **TensorFlow/Keras**.

The project demonstrates the complete machine learning workflow, including dataset exploration, preprocessing, neural network construction, model training, evaluation, prediction, visualization, and an experiment using **Dropout regularization**.

---

## 📌 Project Overview

Handwritten digit recognition is a classic computer vision and deep learning problem. In this project, a neural network is trained to recognize handwritten digits from the MNIST dataset.

The implementation includes two models:

1. **Original Neural Network**
2. **Neural Network with Dropout**

The models are trained and evaluated on the MNIST test dataset, and their validation performance is compared to understand the effect of Dropout on the model.

---

## 🎯 Objectives

* Load and explore the MNIST handwritten digit dataset.
* Understand the shape and distribution of the dataset.
* Normalize image pixel values.
* Build a neural network using TensorFlow/Keras.
* Train the model on handwritten digit images.
* Evaluate the model using test accuracy and loss.
* Visualize training and validation performance.
* Predict handwritten digits using the trained model.
* Experiment with Dropout to reduce overfitting.
* Compare the performance of the original and Dropout models.

---

## 🧠 Technologies Used

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Python     | Programming language      |
| TensorFlow | Deep learning framework   |
| Keras      | Neural network API        |
| NumPy      | Numerical operations      |
| Matplotlib | Data visualization        |
| MNIST      | Handwritten digit dataset |

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**, which is directly loaded using TensorFlow/Keras.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

The images are grayscale handwritten digits with a size of:

```text
28 × 28 pixels
```

The dataset is divided into:

* Training images and labels
* Testing images and labels

The code also displays the dataset shapes and checks the minimum and maximum pixel values before normalization.

---

## 🔄 Data Preprocessing

Before training, pixel values are normalized from the range:

```text
0 – 255
```

to:

```text
0 – 1
```

This is performed using:

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

Normalization helps provide appropriately scaled input values for neural network training.

---

## 🏗️ Model Architecture

The original neural network uses a simple feed-forward architecture:

```text
Input Image
   │
   ▼
28 × 28 Image
   │
   ▼
Flatten
   │
   ▼
784 Values
   │
   ▼
Dense Layer
128 Neurons
ReLU Activation
   │
   ▼
Dense Layer
10 Neurons
Softmax Activation
   │
   ▼
Predicted Digit
0 – 9
```

### Model Structure

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

The `Flatten` layer converts the `28 × 28` image into 784 values. The hidden Dense layer contains 128 neurons with ReLU activation, while the output layer contains 10 neurons representing digits from 0 to 9.

---

## ⚙️ Model Compilation

The model is compiled using:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metric:** Accuracy

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## 🚀 Model Training

The original model is trained using:

```text
Epochs: 10
Batch Size: 32
Validation Split: 20%
```

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
```

Twenty percent of the training data is used for validation during training.

---

## 📈 Model Evaluation

After training, the model is evaluated using the MNIST test dataset.

The project reports:

* Test Loss
* Test Accuracy

```python
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
```

The test accuracy is displayed as a percentage.

> **Note:** The README intentionally does not hard-code an accuracy value because the actual result should come from running the code in your environment.

---

## 📉 Training Visualization

The project visualizes both training and validation performance.

### Accuracy

The training and validation accuracy are plotted across epochs to observe how the model learns.

### Loss

Training and validation loss are also plotted to analyze the model's learning behavior.

These visualizations can help identify issues such as overfitting or underfitting.

---

## 🔢 Digit Prediction

The trained model is tested on five images from the MNIST test dataset.

The model generates class probabilities using:

```python
predictions = model.predict(sample_images)
```

The predicted digit is obtained using:

```python
predicted_labels = np.argmax(predictions, axis=1)
```

The project displays each image together with its:

* Actual label
* Predicted label

It also prints whether the prediction was **Correct** or **Wrong**.

---

## 🧪 Dropout Experiment

As an experiment, a second neural network is created by adding a **Dropout layer**.

### Original Architecture

```text
Flatten
   ↓
Dense(128)
   ↓
Dense(10)
```

### Dropout Architecture

```text
Flatten
   ↓
Dense(128)
   ↓
Dropout(0.2)
   ↓
Dense(10)
```

The Dropout layer randomly disables 20% of the neurons during training.

```python
tf.keras.layers.Dropout(0.2)
```

The purpose of this experiment is to investigate whether Dropout can help reduce overfitting.

---

## 🔬 Model Comparison

The project compares the original and Dropout models using:

* Test accuracy
* Best validation accuracy
* Validation accuracy curves
* Validation loss curves

The comparison makes it possible to observe how adding Dropout affects model performance.

---

## 📁 Project Structure

Recommended repository structure:

```text
MNIST-HANDWRITTEN-DIGIT-CLASSIFICATION/
│
├── main.py
├── README.md
└── requirements.txt
```

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/vijaygowda93/MNIST-HANDWRITTEN-DIGIT-CLASSIFICATION.git
```

### 2. Navigate to the project directory

```bash
cd MNIST-HANDWRITTEN-DIGIT-CLASSIFICATION
```

### 3. Install dependencies

```bash
pip install tensorflow numpy matplotlib
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Python program using:

```bash
python main.py
```

The program will:

1. Load the MNIST dataset.
2. Display dataset information.
3. Display sample handwritten digits.
4. Show class distribution.
5. Normalize the images.
6. Build the neural network.
7. Train the original model.
8. Evaluate the model.
9. Display accuracy and loss graphs.
10. Predict five test images.
11. Train the Dropout model.
12. Compare both models.

---

## 📌 Key Concepts Demonstrated

This project provides practical implementation of several deep learning concepts:

* MNIST dataset
* Image preprocessing
* Data normalization
* Artificial neural networks
* Dense layers
* ReLU activation
* Softmax activation
* Forward propagation
* Model training
* Adam optimizer
* Cross-entropy loss
* Validation data
* Model evaluation
* Classification
* Prediction
* Dropout regularization
* Overfitting analysis
* Training visualization

---

## 📊 Expected Output

When the program is executed, it produces:

* MNIST dataset information
* Pixel value ranges
* Sample digit images
* Class distribution
* Neural network architecture summary
* Training progress
* Test loss and accuracy
* Training vs. validation accuracy graph
* Training vs. validation loss graph
* Actual vs. predicted digit visualization
* Prediction result table
* Original vs. Dropout model comparison
* Validation accuracy comparison
* Validation loss comparison

---

## 🔮 Future Improvements

The current project uses a basic fully connected neural network. It can be extended in several ways:

* Implement a **Convolutional Neural Network (CNN)** for improved image recognition.
* Add more hidden layers.
* Perform hyperparameter tuning.
* Add confusion matrix visualization.
* Calculate precision, recall, and F1-score.
* Implement early stopping.
* Save and load trained models.
* Create a web interface for handwritten digit prediction.
* Allow users to draw a digit and classify it in real time.
* Experiment with different Dropout rates.
* Compare Dense, CNN, and other architectures.

---

## 🎓 Learning Outcomes

After completing this project, you can understand how a neural network can be used for image classification and how preprocessing, model architecture, training, validation, and regularization affect machine learning performance.

The project also provides a practical introduction to TensorFlow/Keras and demonstrates how model performance can be analyzed visually rather than relying only on a single accuracy value.

---

## 👨‍💻 Author

**Rihan Dastagir Khureshi**

GitHub:(https://github.com/rihankhureshi)

---

## 📄 License

This project is intended for educational and learning purposes.
