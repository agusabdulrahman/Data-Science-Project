from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model/iris_rf_model.pkl")

# Define FastAPI app
app = FastAPI(title="Iris Classifier API")

# Define request schema
class InputData(BaseModel):
    input: list  # Example: [5.1, 3.5, 1.4, 0.2]

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array(data.input).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
