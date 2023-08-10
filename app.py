from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained models from pickle files
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
lr_model = pickle.load(open('lr_model.pkl', 'rb'))

# Render the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    open_price = float(request.form['Open'])
    high_price = float(request.form['High'])
    low_price = float(request.form['Low'])
    volume = float(request.form['Volume'])

    # Create a DataFrame with the input values
    input_df = pd.DataFrame({'Open': [open_price], 'High': [high_price], 'Low': [low_price], 'Volume': [volume]})

    # Make predictions using the trained models
    rf_prediction = rf_model.predict(input_df)
    lr_prediction = lr_model.predict(input_df)

    # Prepare the prediction results
    results = {
        'Random Forest Predicted Price': rf_prediction[0],
        'Linear Regression Predicted Price': lr_prediction[0]
    }

    # Render the results template
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)