from django.shortcuts import render
from django.http import JsonResponse
import pickle
import numpy as np
import json
import requests
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
from .models import FloodPrediction

model_path = "predictor/random_forest_model.pkl"
with open(model_path, "rb") as file:
    model, label_encoder, scaler = pickle.load(file)  
    
# OpenWeather API Key (Replace with your actual API key)
OPENWEATHER_API_KEY = "your_openweather_api_key"

# Risk level mapping
risk_mapping = {
    0: "Very Low",
    1: "Low",
    2: "High",
    3: "Very High"
}

def home(request):
    return render(request, 'main.html')


def get_weather_data(latitude, longitude):

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "rainfall": data.get("rain", {}).get("1h", 0),  # Default to 0 if no rain data
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"]
            }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
    return None  # Return None if API fails


def get_elevation(latitude, longitude):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["results"][0]["elevation"]
    except Exception as e:
        print(f"Error fetching elevation data: {e}")
    return None  # Return None if API fails

@csrf_exempt
def predict_risk(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            latitude = data.get("latitude")
            longitude = data.get("longitude")

            # Fetch environmental data
            # weather_data = get_weather_data(latitude, longitude)
            # elevation = get_elevation(latitude, longitude)

            
            weather_data= None
            elevation = None
            
            if weather_data is None:
                weather_data = {"rainfall": 0.462, "temperature":26.792, "humidity":97.092}
            
            if elevation is None:
                elevation =0.8603  

            # Placeholder values for other features (these should ideally come from APIs or GIS data)
            river_discharge = 0.8833  
            water_level =0.9155 
            population_density = 5413.902  

            prediction_record = FloodPrediction(
                latitude=latitude,
                longitude=longitude,
                water_level=water_level,
                river_discharge=river_discharge,
                rainfall=weather_data['rainfall'],
                elevation=elevation,
                humidity=weather_data['humidity'],
                temperature=weather_data['temperature'],
                population_density=population_density
            )

            # Ensure the feature order matches the training data
            features = np.array([[  
                water_level,               
                river_discharge,           
                weather_data['rainfall'],  
                elevation,                 
                weather_data['humidity'],  
                weather_data['temperature'], 
                population_density         
            ]])

            features = scaler.transform(features)

            # **Predict flood risk**
            prediction_numeric = model.predict(features)[0]

            # **Convert numerical prediction to label**
            prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

            # Add prediction to the object
            prediction_record.predicted_risk = prediction_label

            # Save the object
            prediction_record.save()

            return JsonResponse({'risk_level': prediction_label})
        
        except Exception as e:
            print(f"Error in prediction: {e}")  # Debugging log
            return JsonResponse({'error': str(e)})   



@csrf_exempt
def predict_advanced_risk(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            latitude = data.get("latitude")
            longitude = data.get("longitude")
            rainfall = data.get("rainfall")
            temperature = data.get("temperature")
            humidity = data.get("humidity")

            # Validate inputs
            if None in (latitude, longitude, rainfall, temperature, humidity):
                return JsonResponse({'error': 'Incomplete input data. Please provide latitude, longitude, rainfall, temperature, and humidity.'})

            # Fetch additional features from APIs or fallback
            # elevation = get_elevation(latitude, longitude)
            elevation= None
            if elevation is None:
                elevation = 0.1315  # Default/fallback value

            # Placeholder values (replace with API integration if needed)
            river_discharge = 0.6683
            water_level = 0.258
            population_density = 3630.70

            prediction_record = FloodPrediction(
                latitude=latitude,
                longitude=longitude,
                water_level=water_level,
                river_discharge=river_discharge,
                rainfall=rainfall,
                elevation=elevation,
                humidity=humidity,
                temperature=temperature,
                population_density=population_density
            )

            # Prepare feature array for model input
            features = np.array([[  
                water_level,               
                river_discharge,           
                rainfall,                  
                elevation,                 
                humidity,                  
                temperature,               
                population_density         
            ]])

            features = scaler.transform(features)

            prediction_numeric = model.predict(features)[0]
            prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

            # Store prediction result
            prediction_record.predicted_risk = prediction_label

            # Save record to database
            prediction_record.save()

            return JsonResponse({'risk_level': prediction_label})

        except Exception as e:
            print(f"Error in advanced prediction: {e}")
            return JsonResponse({'error': str(e)})





qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Sample flood-related context for BERT model
CONTEXT = """
Floods are caused by excessive rainfall, river overflow, or storm surges. 
The risk of floods increases in low-lying areas and places with poor drainage systems.
Safety measures include moving to higher ground, having an emergency kit, and staying updated on weather alerts.
A flood is an overflow of water that submerges land that is usually dry. It is one of the most common and destructive natural disasters.

Types of floods:

River Floods: Occur when rivers overflow due to heavy rain or melting snow.

Flash Floods: Happen suddenly due to heavy rainfall, often in urban areas or near mountains.

Coastal Floods: Caused by storm surges, tsunamis, or high tides.

Urban Flooding: Results from poor drainage systems, heavy rain, or rapid urbanization.

Pluvial (Surface Water) Floods: Occur when heavy rain saturates the ground and drainage systems can't keep up.

Glacial Lake Outburst Floods (GLOFs): Happen when glacial lakes break due to natural dam failure.

History of Major Floods
China Floods (1931): Deadliest flood in recorded history, affecting over 50 million people, with an estimated 1-4 million deaths.

Hurricane Katrina (2005, USA): Costliest flood-related disaster; New Orleans was heavily affected.

Pakistan Floods (2010): Covered 20'%' of the country, affecting over 20 million people.

European Floods (2021): Extreme rainfall caused severe flooding in Germany, Belgium, and the Netherlands.

Bangladesh Floods (1998): Affected 75'%' of the country, displacing millions.

Causes of Floods
Natural Causes:

Heavy or prolonged rainfall

Snowmelt from mountains

Storm surges (hurricanes, typhoons)

Tsunamis

Ice jam blockages in rivers

Human-Induced Causes:

Deforestation (reduces water absorption)

Poor urban drainage

Climate change (intensifies storms and sea-level rise)

Dam failures (e.g., Banqiao Dam failure in China, 1975)

Overuse of groundwater leading to land subsidence

Effects of Floods
Economic: Infrastructure damage, loss of crops, destruction of businesses.

Environmental: Soil erosion, water contamination, habitat destruction.

Social & Health: Displacement, waterborne diseases (cholera, dysentery), loss of life.

"""

@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()

            if not question:
                return JsonResponse({"answer": "Please ask a flood-related question."})

            # Use BERT model for question answering
            response = qa_pipeline(question=question, context=CONTEXT)
            answer = response['answer']

            return JsonResponse({"answer": answer})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)