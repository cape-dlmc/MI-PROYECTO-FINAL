# training/train_ann.py
from sklearn.neural_network import MLPClassifier
import joblib
from .preprocess import load_and_preprocess_data

def train_ann_model():
    X, y, scaler = load_and_preprocess_data()
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X, y)
    joblib.dump(model, 'models/ann.pkl')