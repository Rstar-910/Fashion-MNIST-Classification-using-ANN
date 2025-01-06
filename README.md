# Fashion-MNIST-Classification-using-Artificial Neural Network (ANN)
This project demonstrates the classification of images in the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using an **Artificial Neural Network (ANN)** implemented with TensorFlow and Keras.

---

## 📜 Dataset

The **Fashion MNIST dataset** contains:
- **70,000 grayscale images**: \(60,000\) for training and \(10,000\) for testing.
- **Image dimensions**: \(28 \times 28\).
- **Classes**: 10 categories of clothing and accessories:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

---

## 🛠️ Model Overview

The model is an **Artificial Neural Network (ANN)** with the following architecture:
- **Flatten layer**: Converts the 2D input (\(28 \times 28\)) into a 1D array of 784 features.
- **Hidden layer**: A dense layer with 128 neurons and **ReLU** activation function.
- **Output layer**: A dense layer with 10 neurons (one for each class) and a **softmax** activation function.



### **Model Code**
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


'''markdown
🚀 Features
Image Classification: Predicts the category of clothing or accessories for a given image.
Simple ANN Architecture: A feedforward neural network with fully connected layers.
Accuracy Evaluation: Measures the model's performance on a test dataset.
Visualization: Displays sample predictions and training metrics.


