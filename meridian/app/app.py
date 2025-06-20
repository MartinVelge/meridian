import os
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np


def fit_linear_regression(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    intercept = float(beta[0])
    coefs = beta[1:]
    y_pred = X_design @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return intercept, coefs, r2


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'meridian-demo-key'

    upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    demo_data = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', 'sample_data_media_and_rf.csv')

    run_counter = {'count': 0}
    results: dict[str, dict] = {}

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            if 'demo' in request.form:
                session['data_path'] = demo_data
            else:
                file = request.files.get('datafile')
                if file and file.filename:
                    path = os.path.join(upload_folder, file.filename)
                    file.save(path)
                    session['data_path'] = path
            return redirect(url_for('setup'))
        return render_template('index.html')

    @app.route('/setup', methods=['GET', 'POST'])
    def setup():
        data_path = session.get('data_path')
        if not data_path:
            return redirect(url_for('index'))
        df = pd.read_csv(data_path)
        columns = df.columns.tolist()
        if request.method == 'POST':
            response = request.form.get('response')
            features = request.form.getlist('features')
            if not response or not features:
                return render_template('setup.html', columns=columns, error='Select response and at least one feature')
            session['response'] = response
            session['features'] = features
            return redirect(url_for('run_model'))
        return render_template('setup.html', columns=columns)

    @app.route('/run', methods=['GET'])
    def run_model():
        data_path = session.get('data_path')
        response = session.get('response')
        features = session.get('features')
        if not data_path or not response or not features:
            return redirect(url_for('index'))
        df = pd.read_csv(data_path)
        X = df[features].values
        y = df[response].values
        intercept, coefs, r2 = fit_linear_regression(X, y)
        run_counter['count'] += 1
        version = f"run_{run_counter['count']}"
        results[version] = {
            'intercept': intercept,
            'coefficients': dict(zip(features, coefs)),
            'r2': r2,
        }
        session['version'] = version
        return redirect(url_for('results_page'))

    @app.route('/results')
    def results_page():
        version = session.get('version')
        if not version:
            return redirect(url_for('index'))
        result = results.get(version)
        return render_template('results.html', version=version, result=result)

    return app
