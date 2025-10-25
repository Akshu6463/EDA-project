from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
LABELS = {0: "Incorrect", 1: "Correct"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = [float(x) for x in request.form.values()]
        features = np.array(form_values).reshape(1, -1)
        prediction_code = model.predict(features)[0]
        prediction_text = LABELS.get(prediction_code, "Unknown")
        confidence = model.predict_proba(features)[0].max() * 100
        return render_template('index.html',
                               prediction_text=f'Prediction: {prediction_text}',
                               confidence_text=f'Confidence: {confidence:.2f}%')
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {e}',
                               confidence_text='')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
