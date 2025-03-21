from django.shortcuts import render
from django.http import JsonResponse
import pickle
import numpy as np
import json
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline

# Load the trained model and scaler
model = pickle.load(open('predictor/flood_risk_model.pkl', 'rb'))
scaler = pickle.load(open('predictor/scaler.pkl', 'rb'))

# Mapping of numeric labels to risk levels
risk_mapping = {
    0: "Very Low",
    1: "Low",
    2: "High",
    3: "Very High"
}

def home(request):
    return render(request, 'index.html')

@csrf_exempt
def predict_risk(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract features in the correct order
            features = np.array([[  
                data['latitude'], data['longitude'], data['rainfall'],
                data['temperature'], data['humidity'], data['river_discharge'],
                data['water_level'], data['elevation'], data['land_cover'],
                data['soil_type'], data['population_density'], data['infrastructure']
            ]])
            
            # Scale input features
            features_scaled = scaler.transform(features)
            
            # Predict risk level
            prediction_numeric = model.predict(features_scaled)[0]
            
            # Convert numerical prediction to category
            prediction_label = risk_mapping.get(int(prediction_numeric), "Unknown")

            return JsonResponse({'risk_level': prediction_label})
        except Exception as e:
            return JsonResponse({'error': str(e)})
        

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
def chatbot_view(request):
    if request.method == "POST":
        import json
        data = json.loads(request.body)
        question = data.get("question", "")

        # Define flood-related knowledge
        flood_context = """
        Floods are caused by excessive rainfall, river overflow, or storm surges. 
        They can be mitigated by proper drainage systems, early warnings, and emergency preparedness.
        """

        if question:
            response = qa_pipeline(question=question, context=flood_context)
            return JsonResponse({"answer": response["answer"]})
        
    return JsonResponse({"answer": "Sorry, I didn't understand that."})