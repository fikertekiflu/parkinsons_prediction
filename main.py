from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS middleware for handling cross-origin requests
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model to define the input structure for Parkinson's disease prediction
class ParkinsonsInput(BaseModel):
    Fo: float  # Fundamental frequency of vocal fold vibrations
    Fhi: float  # Highest vocal fold frequency
    Flo: float  # Lowest vocal fold frequency
    Jitter_Percent: float  # Frequency variation percentage
    Jitter_Abs: float  # Absolute jitter in vocal fold vibration
    RAP: float  # Relative average perturbation
    PPQ: float  # Pitch perturbation quotient
    DDP: float  # Degree of pitch perturbation
    Shimmer: float  # Amplitude variation in vocal fold vibrations
    Shimmer_dB: float  # Amplitude variation in decibels
    APQ3: float  # Three-point amplitude perturbation quotient
    APQ5: float  # Five-point amplitude perturbation quotient
    APQ: float  # Amplitude perturbation quotient
    DDA: float  # Difference of differential amplitudes
    NHR: float  # Noise-to-harmonics ratio
    HNR: float  # Harmonics-to-noise ratio
    RPDE: float  # Recurrence period density entropy
    DFA: float  # Detrended fluctuation analysis
    Spread1: float  # First measure of spread in vocal fold frequencies
    Spread2: float  # Second measure of spread in vocal fold frequencies
    D2: float  # Correlation dimension
    PPE: float  # Pitch period entropy

# Load the saved Parkinson's disease prediction model (assuming a pickled model)
try:
    with open('parkinsons_model.sav', 'rb') as f:
        parkinsons_model = pickle.load(f)
except Exception as e:
    print(f"Error loading the Parkinson's disease model: {e}")

# Endpoint for Parkinson's disease prediction
@app.post('/smart-symptomChecker/parkinsons')
async def parkinsons_predict(input_parameters: ParkinsonsInput):
    try:
        # Convert input data to dictionary and list for prediction
        input_data = input_parameters.dict()
        input_list = [
            input_data['Fo'],
            input_data['Fhi'],
            input_data['Flo'],
            input_data['Jitter_Percent'],
            input_data['Jitter_Abs'],
            input_data['RAP'],
            input_data['PPQ'],
            input_data['DDP'],
            input_data['Shimmer'],
            input_data['Shimmer_dB'],
            input_data['APQ3'],
            input_data['APQ5'],
            input_data['APQ'],
            input_data['DDA'],
            input_data['NHR'],
            input_data['HNR'],
            input_data['RPDE'],
            input_data['DFA'],
            input_data['Spread1'],
            input_data['Spread2'],
            input_data['D2'],
            input_data['PPE']
        ]

        # Perform prediction
        prediction = parkinsons_model.predict([input_list])

        # Generate appropriate response
        if prediction[0] == 0:
            return JSONResponse(content={"result": "The person does not have Parkinson's disease"}, status_code=200)
        else:
            return JSONResponse(content={"result": "The person has Parkinson's disease"}, status_code=200)

    except Exception as e:
        # Error handling in case of prediction failure
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
