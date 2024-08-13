from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('stroke.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create input array for prediction
        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married,
                                work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return result
        if prediction[0] == 1:
            result = "The patient is likely to have a stroke."
        else:
            result = "The patient is unlikely to have a stroke."

        return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
