# prediction/predict_batch.py
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
import time

STATIC_FOLDER = 'static'
IMAGE_LIFETIME_SECONDS = 300  # Tiempo de vida de las imágenes en segundos (5 minutos)

def cleanup_old_images():
    now = time.time()
    for fname in os.listdir(STATIC_FOLDER):
        if fname.endswith('.png'):
            path = os.path.join(STATIC_FOLDER, fname)
            if now - os.path.getmtime(path) > IMAGE_LIFETIME_SECONDS:
                os.remove(path)

def predict_from_batch(filepath, model_name):
    cleanup_old_images()

    df = pd.read_excel(filepath)
    y_true = df['C31']
    X = df.drop(columns=['C31'])

    model = joblib.load(f'models/{model_name}.pkl')
    scaler = joblib.load('models/scaler.pkl')
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión')

    image_id = str(uuid.uuid4())
    image_path = os.path.join(STATIC_FOLDER, f'{image_id}.png')
    plt.savefig(image_path)
    plt.close()

    return {
        'confusion_matrix': cm.tolist(),
        'accuracy': round(acc * 100, 2),
        'image_path': image_path
    }