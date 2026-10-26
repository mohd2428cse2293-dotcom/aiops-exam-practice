from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/iris_random_forest_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Flower Prediction API")

class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health_check():
    return {"status": "healthy"} 
@app.post("/predict")
def predict(features: IrisFeatures):
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    
    prediction = int(model.predict(input_data)[0])
    class_names = ["setosa", "versicolor", "virginica"]
    
    return {
        "prediction": prediction,
        "class_name": class_names[prediction]
    }

 
    

    
