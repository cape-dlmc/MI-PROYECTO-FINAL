# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import os

from prediction.predict_individual import predict_single_input
from prediction.predict_batch import predict_from_batch
from training.train_logistic import train_logistic_model
from training.train_svm import train_svm_model
from training.train_ann import train_ann_model
from training.train_fcm import train_fcm_model

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)
CSRFProtect(app)
UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        model_type = request.form['model']
        if model_type == 'logistic':
            train_logistic_model()
        elif model_type == 'svm':
            train_svm_model()
        elif model_type == 'ann':
            train_ann_model()
        elif model_type == 'fcm':
            train_fcm_model()
        flash(f'Modelo {model_type.upper()} entrenado correctamente!', 'success')
        return redirect(url_for('training'))
    return render_template('training.html')

@app.route('/predict_individual', methods=['GET', 'POST'])
def predict_individual():
    if request.method == 'POST':
        data = request.form.to_dict()
        model = request.form['model']
        prediction = predict_single_input(data, model)
        return render_template('predict_individual.html', prediction=prediction)
    return render_template('predict_individual.html')

@app.route('/predict_batch', methods=['GET', 'POST'])
def predict_batch():
    if request.method == 'POST':
        file = request.files['file']
        model = request.form['model']
        filename = secure_filename(file.filename or '')
        if filename.lower().endswith('.xlsx'):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            results = predict_from_batch(filepath, model)
            return render_template('predict_batch.html', results=results)
        else:
            flash('Formato de archivo no válido. Solo se acepta .xlsx', 'danger')
    return render_template('predict_batch.html')

if __name__ == '__main__':
    app.run(debug=True)