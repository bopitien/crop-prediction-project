from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Define the FastAPI app
app = FastAPI()

# Load your trained model
model = joblib.load('random_forest_model.pkl')  # Replace 'your_model.pkl' with your actual model file

# Define the data structure for incoming requests using Pydantic
class InputData(BaseModel):
    N: float
    P: float
    K: float
    ph: float

@app.post('/predict')
def predict(data: InputData):
    features = [[data.N, data.P, data.K, data.ph]]
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
