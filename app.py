from flask import Flask, render_template, request
import pickle
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

model = pickle.load(open(os.path.join(BASE_DIR, 'models/model.pkl'), 'rb'))
holiday_encoder = pickle.load(open(os.path.join(BASE_DIR, 'models/holiday_encoder.pkl'), 'rb'))
weather_encoder = pickle.load(open(os.path.join(BASE_DIR, 'models/weather_encoder.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        holiday = request.form['holiday']
        # Handle unseen holidays
        if holiday not in holiday_encoder.classes_:
            holiday = 'Unknown'
        temperature = float(request.form['temperature'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        second = int(request.form['second'])

        # Encode categorical features
        holiday_encoded = holiday_encoder.transform([holiday])[0]
        weather_encoded = weather_encoder.transform([weather])[0]

        features = np.array([[holiday_encoded, temperature, rain, snow, weather_encoded, year, month, day, hour, minute, second]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 