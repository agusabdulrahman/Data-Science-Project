from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# load the model
model = joblib.load('Computer_vision/Packaging_model/rf_model.pkl')

# define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from POST request
    input_data = np.array(data['input']).reshape(1, -1)  # Reshape input
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)