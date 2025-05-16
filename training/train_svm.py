# training/train_svm.py
from sklearn.svm import SVC
import joblib
from .preprocess import load_and_preprocess_data

def train_svm_model():
    X, y, scaler = load_and_preprocess_data()
    model = SVC(probability=True)
    model.fit(X, y)
    joblib.dump(model, 'models/svm.pkl')