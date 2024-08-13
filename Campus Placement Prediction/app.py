from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and label encoders
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        
        # Process categorical variables
        for column in df.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                le = label_encoders[column]
                # Handle unseen labels by assigning a default value (-1)
                if df[column].iloc[0] in le.classes_:
                    df[column] = le.transform(df[column])
                else:
                    df[column] = -1  # Ensure the model can handle -1 or a placeholder for unseen labels
        
        # Convert columns to numeric (handle mixed types and errors)
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    df[column] = df[column].astype(float)
                except ValueError:
                    return f"Invalid value for numerical column '{column}'. Please enter valid data."
        
        # Handle missing values by filling with 0
        df.fillna(0, inplace=True)
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)
        prediction_proba = model.predict_proba(df_scaled)[:, 1]
        
        return render_template('result.html', prediction=prediction[0], proba=round(prediction_proba[0] * 100, 2))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)



