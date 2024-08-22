from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd

# Load the trained model and feature names
with open('best_model_Titanic_dataset.pkl', 'rb') as f:
    model_rfc = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

app = Flask(__name__)

def preprocess_input(age, sex, pclass,  embark_town, familysize):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'pclass': [pclass],
        'embark_town': [embark_town],
        'familysize': [familysize]
    })
    
    # Encode categorical variables with get_dummies
    input_data_encoded = pd.get_dummies(input_data)
    
    # Ensure the same columns as in training data
    for col in feature_names:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[feature_names]
    
    return input_data_encoded

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract input values
    age = data['age']
    sex = data['sex']
    pclass = data['pclass']
    embark_town = data['embark_town']
    familysize = data['familysize']
    
    # Preprocess the input data
    processed_data = preprocess_input(age, sex, pclass, embark_town, familysize)
    
    # Predict survival using the trained model
    prediction = model_rfc.predict(processed_data)
    
    # Return the prediction (0 for not survived, 1 for survived)
    result = 'Survived' if prediction[0] == 1 else 'Not Survived'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

