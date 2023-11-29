# Fashion MNIST Classification with TensorFlow

## Setup

Make sure you have TensorFlow installed:

```bash
pip install tensorflow
```

## 1. Check TensorFlow Version

```python
import tensorflow as tf
print(tf.__version__)
```

## 2. Import TensorFlow and Keras

```python
from tensorflow import keras
print(keras)
```

## 3. Load and Preprocess Data

```python
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# ... (preprocessing steps)
```

## 4. Define Tags

```python
tags = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "BAG", "Ankle"]
print(tags[y_train[0]])
```

## 5. Build the Neural Network Model

```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="softmax"))
```

## 6. Model Summary and Compilation

```python
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
```

## 7. Training the Model

```python
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
```

## 8. Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```

## 9. Model Evaluation

```python
model.evaluate(x_test, y_test)
```

## License

This notebook is distributed under the MIT License. See [LICENSE](LICENSE) for more information.
