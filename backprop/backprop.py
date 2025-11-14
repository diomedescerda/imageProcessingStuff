import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

digits = load_digits()
X = digits.data        # shape (1797, 64)
y = digits.target.reshape(-1, 1)  # shape (1797, 1)

# Normalize
X = X / 16.0  # pixel values 0-16 -> 0-1

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

input_size = 64
hidden_size = 32
output_size = 10
lr = 0.1

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-8)) / pred.shape[0]

epochs = 50
batch_size = 32
num_batches = X_train.shape[0] // batch_size

for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    for i in range(num_batches):
        Xb = X_train[i*batch_size:(i+1)*batch_size]
        yb = y_train[i*batch_size:(i+1)*batch_size]

        # Forward pass
        z1 = Xb @ W1 + b1
        a1 = sigmoid(z1)

        z2 = a1 @ W2 + b2
        y_pred = softmax(z2)

        # Loss
        loss = cross_entropy(y_pred, yb)

        # Backpropagation
        dz2 = (y_pred - yb) / batch_size
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * sigmoid_derivative(z1)
        dW1 = Xb.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

def predict(X):
    a1 = sigmoid(X @ W1 + b1)
    y_pred = softmax(a1 @ W2 + b2)
    return np.argmax(y_pred, axis=1)

y_test_labels = np.argmax(y_test, axis=1)
preds = predict(X_test)

accuracy = np.mean(preds == y_test_labels)
print("Test Accuracy:", accuracy)