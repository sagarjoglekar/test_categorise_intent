from flask import Flask, request
import joblib

# Load your model object
model = joblib.load("TfIdfClassifier.joblib")  # Update this line with your actual code

# Create a Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Make predictions using the model
    predictions = model.predict(data)

    # Return the predictions as a response
    return {'predictions': predictions}

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000)
