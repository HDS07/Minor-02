from django.shortcuts import render
from django.http import JsonResponse
import pickle
import numpy as np
import json
import requests
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .models import FloodPrediction

model_path = "predictor/svm_model.pkl"
with open(model_path, "rb") as file:
    model, label_encoder, scaler = pickle.load(file)  
    

risk_mapping = {
    0: "Very Low",
    1: "Low",
    2: "High",
    3: "Very High"
}

def home(request):
    return render(request, 'main.html')


OPENWEATHER_API_KEY = '4924b08220e50c3782f66ca37fd050ac'  # make sure your key is set

def get_weather_data(latitude, longitude):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            temperature = data.get("main", {}).get("temp")
            humidity = data.get("main", {}).get("humidity")
            rainfall = data.get("rain", {}).get("1h", 0)  # default to 0 if no rain data

            return {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall
            }
        else:
            print(f"Failed to fetch weather data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching weather data: {e}")

    return None  

@csrf_exempt
def predict_risk(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            latitude = data.get("latitude")
            longitude = data.get("longitude")

            # Fetch environmental data
            weather_data = get_weather_data(latitude, longitude)
            print(weather_data)

            elevation = None
            
            if weather_data is None:
                print("Flag")
                weather_data = {"rainfall": 0.899, "temperature":39.601, "humidity":54.229}
            
            if elevation is None:
                elevation =0.335 

            river_discharge =  0.747 
            water_level =0.397 
            population_density = 457.423  

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
            water_level = data.get("water_level")
            river_discharge = data.get("river_discharge")
            elevation = data.get("elevation")
            population_density = data.get("population_density")

            # Validate inputs
            if None in (latitude, longitude, rainfall, water_level, river_discharge, elevation, population_density):
                return JsonResponse({'error': 'Incomplete input data. Please provide latitude, longitude, rainfall, water_level, river_discharge, elevation and population_density.'})

            weather_data = get_weather_data(latitude, longitude)

            prediction_record = FloodPrediction(
                latitude=latitude,
                longitude=longitude,
                water_level=water_level,
                river_discharge=river_discharge,
                rainfall=rainfall,
                elevation=elevation,
                humidity=weather_data['humidity'],
                temperature=weather_data['temperature'],
                population_density=population_density
            )

            features = np.array([[  
                water_level,               
                river_discharge,           
                rainfall,                  
                elevation,                 
                weather_data['humidity'],
                weather_data['temperature'],               
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




embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')

DOCUMENTS = [
    "Floods are caused by heavy or prolonged rainfall, river overflow, melting snow, or storm surges.",
    "Human activities like deforestation, poor urban drainage, and climate change significantly increase flood risks.",
    "Dam failures and tsunamis are additional causes of severe flooding events.",

    "River floods occur when rivers overflow their banks due to excessive rain or snowmelt.",
    "Flash floods happen suddenly with little warning, typically after intense rainfall in urban or mountainous areas.",
    "Coastal floods are caused by storm surges from cyclones, hurricanes, tsunamis, or unusually high tides.",
    "Urban flooding results from poor drainage systems and rapid urban development with impervious surfaces.",
    "Pluvial floods occur when heavy rain saturates the ground and drainage systems are overwhelmed.",
    "Glacial lake outburst floods happen when natural dams containing glacial lakes fail, releasing large volumes of water suddenly.",

    "Satellite imagery helps monitor rainfall, river levels, and flooding across large areas in real-time.",
    "Hydrological models simulate rainfall-runoff relationships to predict flooding risks.",
    "AI and Machine Learning models are increasingly used to predict flood occurrence based on environmental and meteorological data.",
    "Remote sensing and GIS (Geographical Information Systems) assist in flood mapping and disaster management planning.",
    "IoT devices like water level sensors installed in rivers and drains provide live flood risk updates.",
    
    "During a flood, move to higher ground immediately and avoid walking or driving through floodwaters.",
    "Prepare an emergency kit with essentials like water, food, flashlight, batteries, first-aid supplies, and important documents.",
    "Listen to official alerts and evacuation instructions via radio, TV, or mobile apps.",
    "If trapped inside a building during flooding, go to the highest floor without going into the attic unless necessary.",

    "Floods can cause soil erosion, leading to the loss of fertile agricultural lands.",
    "Contaminated floodwaters spread diseases such as cholera, hepatitis A, and leptospirosis.",
    "Economic impacts of floods include damage to homes, infrastructure, and transportation systems, costing billions globally every year.",
    "Long-term flooding can displace entire communities, causing refugee crises and loss of livelihoods.",
    "Flooding can destroy natural habitats, killing or displacing wildlife species.",
    
    "The 1931 China floods are considered the deadliest in history, affecting millions across the Yangtze, Yellow, and Huai rivers. The disaster claimed between 1 million and 4 million lives, primarily due to drowning and disease outbreaks.",
    "The 1954 North Sea flood in Europe impacted countries such as the Netherlands, UK, and Belgium. It was caused by a storm surge during a North Sea storm and resulted in the deaths of over 1,800 people.",
    "The 1998 Bangladesh floods submerged around 75 percent of the country, displacing over 30 million people and causing widespread damage. Heavy monsoon rains and river overflow led to the disaster.",
    "In 2011, Thailand faced one of its worst flood disasters, resulting in 800 deaths and over $46 billion in damages. The flood affected much of the country, including its capital, Bangkok, disrupting lives and business.",
    "The 2014 Southeast European floods affected Serbia, Bosnia and Herzegovina, and Croatia. The event resulted in over 80 deaths and displaced more than 100,000 people. The floods were caused by a slow-moving weather system bringing heavy rainfall.",
    "In 1927, the Great Mississippi Flood occurred in the USA, one of the most devastating floods in American history. It affected approximately 700,000 people and caused major economic damage to the agriculture sector.",
    "The 2019 India floods caused severe damage in Kerala, Karnataka, Maharashtra, and other states, killing over 1,000 people and displacing more than a million people. The monsoon rains caused landslides and overflowing rivers.",
    "In 2004, the Indian Ocean tsunami caused massive coastal flooding in countries like Sri Lanka, Thailand, Indonesia, and the Maldives. More than 230,000 people perished across 14 countries due to the waves and subsequent flooding.",
    "The 1975 Banqiao Dam failure in China caused one of the worst dam-related floods, killing over 171,000 people. A typhoon caused heavy rainfall, and the dam collapsed, releasing millions of gallons of water, flooding villages below.",
    "The 2010 Pakistan floods affected over 20 million people, making it one of the worst natural disasters in Pakistan's history. Unprecedented monsoon rains caused the Indus River to flood, damaging agriculture and infrastructure.",
    "In 1966, the Florence floods in Italy devastated the city of Florence, causing millions of dollars in damage. The flood submerged major cultural landmarks, including the Uffizi Gallery, and caused significant destruction to art and heritage sites.",
    "The 2008 Indian floods affected Bihar, India, with the Koshi River bursting its banks and submerging much of the state. Over 500 people were killed, and millions were left homeless in one of the most disastrous floods in India in the 21st century.",
    
    "Before floods, create an emergency plan with your family and identify safe evacuation routes.",
    "Install check valves to prevent flood water from backing up into the drains of your home.",
    "Consider purchasing flood insurance, especially if you live in a high-risk flood zone.",
    "Stay informed about weather forecasts and potential flood warnings in your region.",
 
    "After a flood, check your home for structural damage before entering and report broken utilities immediately.",
    "Use protective clothing and boots while cleaning up flood-affected areas to avoid infections.",
    "Document damages with photographs for insurance claims after flood events.",
    "Dispose of food and water that may have been contaminated by floodwaters.",
    "Mental health support is crucial after disasters — survivors should seek counseling if experiencing trauma.",

    "Community-based disaster risk management programs help educate and prepare locals for future floods.",
    "Building homes on stilts and creating elevated platforms for essential infrastructure reduce flood vulnerability.",
    "Flood insurance provides financial protection against property and asset losses from floods.",
    "Zoning laws preventing construction in floodplains help limit potential damage during future floods.",

    "Flood barriers such as levees, embankments, and seawalls help protect areas from rising water.",
    "Retention basins and reservoirs are constructed to store excess rainwater and slowly release it to prevent floods.",
    "Afforestation and reforestation help improve water absorption and reduce runoff that leads to floods.",
    "Urban areas can implement green roofs and permeable pavements to reduce surface runoff and urban flooding.",
    "Early Warning Systems using weather radars and flood sensors allow authorities to alert populations quickly.",
    "River dredging increases a river's capacity to carry water and reduces the risk of overflow.",

    "Flooding has a significant impact on agriculture. Crops like rice, maize, and wheat are vulnerable to submergence in floodwaters, leading to reduced yields. In some cases, entire harvests can be destroyed, leading to food shortages.",
    "Floods can cause soil erosion, washing away valuable topsoil and reducing the soil's fertility. This not only affects current crop production but also the ability to grow crops in the future, especially in flood-prone agricultural regions.",
    "In addition to crop loss, flooding can disrupt the food supply chain, leading to transportation delays and food price hikes. A region that heavily depends on agriculture for its economy can face long-term economic damage.",
    "Post-flood conditions often lead to the contamination of water sources with harmful substances, including chemicals and biological agents. This can severely impact food safety and health, further threatening food security in flood-affected regions.",

    "Flooding increases the risk of waterborne diseases, such as cholera, dysentery, and typhoid fever, due to the contamination of drinking water sources. In the aftermath of floods, it's essential to purify water to prevent outbreaks of disease.",
    "Floods can also lead to mental health issues due to the trauma experienced by survivors. Anxiety, depression, and post-traumatic stress disorder (PTSD) are common among individuals who have lost loved ones, homes, or livelihoods in a flood.",
    "Stagnant floodwaters create breeding grounds for mosquitoes, leading to an increased risk of vector-borne diseases like malaria and dengue fever. Flood-prone areas often see a rise in such diseases following a flood event.",
    "The destruction of health infrastructure, including hospitals and clinics, during floods complicates efforts to provide medical care to the affected population. It also hinders post-disaster recovery and emergency response efforts.",

    "Flooding is one of the major causes of displacement. People forced to evacuate their homes due to flooding often end up in temporary shelters or refugee camps, which can be overcrowded and lack essential services.",
    "Internally displaced people (IDPs) due to floods face many challenges, including limited access to healthcare, education, and employment opportunities. Rebuilding homes and communities after displacement can take years.",
    "Flood-induced displacement can also lead to conflicts over resources such as water, food, and shelter, especially when displaced populations are concentrated in areas with limited capacity to accommodate them.",
    "Climate-induced displacement due to increasing flood risks is becoming a global issue. Many small island nations and low-lying coastal regions are considering migration strategies as a response to the growing threat of flood disasters.",

    "Floods, while often seen as destructive, also play a role in maintaining biodiversity. In floodplains, periodic floods rejuvenate ecosystems by replenishing nutrients in the soil, supporting a wide range of plant and animal species.",
    "However, floods can also threaten biodiversity when they occur too frequently or unpredictably, leading to the destruction of critical habitats. Species that rely on stable environments may face extinction due to rapid or repeated changes.",
    "In some regions, flooding has been shown to lead to the migration of species to higher ground, causing shifts in ecosystem composition. Invasive species can also proliferate in flooded areas, altering the balance of local ecosystems.",

    "Modern technology has significantly improved flood forecasting and monitoring. Remote sensing technologies, such as satellites and drones, provide real-time data on rainfall patterns, water levels, and flood extents.",
    "Flood prediction models now integrate geographic data with weather forecasting systems, allowing for better predictions of when and where floods are likely to occur. These models use machine learning to refine their accuracy over time.",
    "Social media and mobile apps have become crucial tools in flood monitoring and communication. Local authorities and flood-prone communities use platforms like Twitter and Facebook to quickly disseminate flood warnings and alerts.",
    "Artificial intelligence (AI) and machine learning are playing an increasing role in flood prediction and management. AI algorithms analyze vast amounts of historical and real-time data to improve flood risk assessments and response strategies.",

    "Flood resilience refers to the ability of a community to withstand, adapt to, and recover from flooding. Resilient communities invest in flood prevention infrastructure, such as floodwalls and levees, and take proactive measures like land-use planning to avoid flood-prone areas.",
    "Community involvement is critical in building flood resilience. In many flood-prone areas, local communities are engaged in flood management efforts, such as creating flood-ready neighborhoods, restoring wetlands, and organizing evacuation plans.",
    "Public awareness campaigns that educate communities about the risks of flooding and the importance of preparedness can significantly reduce the impacts of floods. In some areas, residents are encouraged to install rainwater harvesting systems to manage water during heavy rainfall.",

    "Flood protection infrastructure, such as levees, dams, and reservoirs, are built to control and manage floodwaters. However, the failure of such infrastructure can lead to catastrophic flooding, as seen with the collapse of the Banqiao Dam in China in 1975.",
    "Some regions are increasingly relying on 'soft' flood protection strategies, which include restoring natural ecosystems like wetlands and forests to help absorb excess water and prevent flooding. These methods are seen as more sustainable compared to hard engineering solutions.",
    "Flood barriers and flood walls are commonly used in urban areas to protect critical infrastructure, such as power stations and transportation networks. However, these solutions need to be regularly maintained to ensure their effectiveness during extreme events.",

    "Machine learning (ML) has revolutionized flood prediction and management in recent years. Traditionally, flood forecasting relied on physical models that simulated hydrological processes and weather patterns. However, these models were often limited by data availability, computational power, and the complexity of real-world conditions.",
    "In the past, flood prediction focused on statistical methods, such as regression analysis and time-series forecasting, which used historical data to predict future floods. While useful, these methods lacked the ability to account for the dynamic, complex nature of flood events and their underlying causes.",
    "With the advent of machine learning, more advanced techniques like neural networks, support vector machines (SVM), and decision trees have been incorporated into flood prediction systems. These models can learn patterns from vast datasets, enabling better predictions based on a combination of historical weather data, geographical information, and real-time sensor inputs.",
    "In recent years, deep learning models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have shown great promise in flood prediction. CNNs, for instance, are used to analyze satellite images and identify flood-prone regions, while RNNs help in predicting rainfall patterns and river discharge over time.",
    "Moreover, machine learning algorithms are now integrated into flood early-warning systems, allowing for real-time monitoring and prediction. This has significantly improved flood risk assessments and response times, providing authorities with the tools to issue timely alerts to communities.",
    "Machine learning is also being applied to optimize flood mitigation strategies. Algorithms can be used to predict the most effective flood protection measures based on various factors, including rainfall intensity, soil types, and river discharge. Additionally, ML helps in evaluating the potential impact of infrastructure failures, such as levee breaches, and their likelihood of causing significant flooding.",
    "Looking ahead, the integration of machine learning with other technologies, such as the Internet of Things (IoT), will provide even more accurate and timely flood predictions. Real-time data from sensors, drones, and weather stations will feed into machine learning models, further enhancing their ability to predict and manage floods. The continuous improvement of these systems promises a future where flood risks are better understood, mitigated, and managed."

]   

document_embeddings = embedder.encode(DOCUMENTS)

@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()

            if not question:
                return JsonResponse({"answer": "Please ask a flood-related question."})

            # Step 1: Embed the user question
            question_embedding = embedder.encode([question])

            # Step 2: Find the most similar document
            similarities = cosine_similarity(question_embedding, document_embeddings)
            best_idx = np.argmax(similarities)
            best_context = DOCUMENTS[best_idx]
            best_score = similarities[0][best_idx]

            # Step 3: Adjust threshold and context usage
            threshold = 0.3  # Lower threshold for better flexibility
            if best_score > threshold:
                # Increase context by combining the best document with others around it
                extended_context = best_context
                if best_idx > 0:  # Add the previous document if available
                    extended_context = DOCUMENTS[best_idx - 1] + " " + extended_context
                if best_idx < len(DOCUMENTS) - 1:  # Add the next document if available
                    extended_context = extended_context + " " + DOCUMENTS[best_idx + 1]
                
                # Step 4: Use the QA model to get a more detailed answer
                response = qa_pipeline(question=question, context=extended_context)
                answer = response['answer']
            else:
                answer = "Sorry, I don't have enough information about that. Please ask something else."

            return JsonResponse({"answer": answer})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)