# prediction/predict_individual.py
import numpy as np
import joblib

def predict_single_input(data_dict, model_name):
    model = joblib.load(f'models/{model_name}.pkl')
    scaler = joblib.load('models/scaler.pkl')
    data = np.array([[float(v) for k, v in data_dict.items() if k != 'model']])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    return 'FGR' if prediction == 1 else 'Normal'