# training/train_logistic.py
from sklearn.linear_model import LogisticRegression
import joblib
from .preprocess import load_and_preprocess_data

def train_logistic_model():
    X, y, scaler = load_and_preprocess_data()
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, 'models/logistic_regression.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')