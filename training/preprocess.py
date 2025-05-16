# training/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path='data/fgr_dataset.xlsx'):
    df = pd.read_excel(path)
    X = df.drop(columns=['C31'])
    y = df['C31']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler