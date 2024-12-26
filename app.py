from flask import Flask, request, render_template
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained pipelines and models
my_pipeline = joblib.load('my_pipeline.joblib')
house_price_model = joblib.load('RANDOM_FOREST_REGRESSOR.joblib')

@app.route('/predict', methods=['GET', 'POST'])
def prediction_function():
    prediction = None
    if request.method == 'POST':
        # Retrieve form data
        data = {
            'CRIM': float(request.form['CRIM']),
            'ZN': float(request.form['ZN']),
            'INDUS': float(request.form['INDUS']),
            'CHAS': int(request.form['CHAS']),
            'NOX': float(request.form['NOX']),
            'RM': float(request.form['RM']),
            'AGE': float(request.form['AGE']),
            'DIS': float(request.form['DIS']),
            'RAD': int(request.form['RAD']),
            'TAX': float(request.form['TAX']),
            'PTRATIO': float(request.form['PTRATIO']),
            'B': float(request.form['B']),
            'LSTAT': float(request.form['LSTAT'])
        }

        # Create a DataFrame from the form data
        input_data = pd.DataFrame([data])

        # Preprocess the data using the loaded pipeline
        preprocessed_data = my_pipeline.transform(input_data)

        # Predict using the house price model
        prediction_value = house_price_model.predict(preprocessed_data)[0]
        
        # Convert prediction to price in USD
        prediction_value *= 1000
        prediction = f"$ {prediction_value:,.2f}"

    return render_template('result.html', prediction=prediction)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
