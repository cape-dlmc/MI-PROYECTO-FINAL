import joblib
import numpy as np
from .preprocess import load_and_preprocess_data
from sklearn.metrics import accuracy_score

class FuzzyCognitiveModel:
    def __init__(self, input_size, lr=0.05, epochs=1500):
        self.weights = np.random.randn(input_size)
        self.bias = 0.0
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1.5 * z))  # curva más empinada

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y):
        for _ in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)
            error = y_hat - y

            grad_w = np.dot(X.T, error) / len(X)
            grad_b = np.mean(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

def train_fcm_model():
    X, y, scaler = load_and_preprocess_data()
    model = FuzzyCognitiveModel(input_size=X.shape[1], epochs=1500, lr=0.05)
    model.fit(X, y)
    joblib.dump(model, 'models/fcm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Exactitud del FCM: {acc * 100:.2f}%")