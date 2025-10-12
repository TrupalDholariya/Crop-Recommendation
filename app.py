# Smart Crop Advisory System - Backend Server (Upgraded with Smarter Regional Filtering)
# FIX: Corrected the datetime import to resolve the AttributeError.
from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
from dotenv import load_dotenv
from datetime import datetime # ⭐ FIX: This line is corrected to import the datetime class directly.
import requests
from groq import Groq

# --- Initialization ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app) 
load_dotenv()

# --- Configuration ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- Securely Initialize Groq Client ---
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    groq_client = None
    print(f"Could not initialize Groq client. Make sure you have set the GROQ_API_KEY environment variable. Error: {e}")

# --- Load Models ---
MODEL_DIR = 'models'
def load_pickle(filename):
    try:
        with open(os.path.join(MODEL_DIR, filename), 'rb') as f: return pickle.load(f)
    except FileNotFoundError:
        return None

crop_rec_model = load_pickle('crop_recommender.pkl')
fert_rec_model = load_pickle('fertilizer_recommender.pkl')
soil_encoder = load_pickle('soil_encoder.pkl')
crop_encoder = load_pickle('crop_encoder.pkl')
known_crops_for_fert = load_pickle('known_crops.pkl')
disease_class_names = load_pickle('disease_class_names.pkl')
try:
    disease_det_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'disease_detector.h5'))
except Exception as e:
    disease_det_model = None
print("All models loaded.")

# --- Multi-Region Knowledge Filter ---
GUJARAT_CROPS = {'rice', 'maize', 'chickpea', 'cotton', 'jute', 'pigeonpeas', 'mungbean', 'groundnut'}
MAHARASHTRA_CROPS = {'rice', 'maize', 'chickpea', 'jowar', 'soyabean', 'cotton', 'sugarcane', 'grapes', 'mango'}

# --- Helper Functions (This is where the error occurred, but the fix is in the import) ---
def get_weather_and_advisory(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        weather = { "city": data.get('name', 'Unknown'), "temperature": data['main']['temp'], "humidity": data['main']['humidity'], "description": data['weather'][0]['description'].capitalize(), "icon": data['weather'][0]['icon']}
    except requests.exceptions.RequestException as e:
        weather = { "city": "Error", "temperature": 0.0, "humidity": 0, "description": "No data", "icon": "01d" }
    
    alerts = []
    current_month = datetime.now().month # This line now works correctly because of the fixed import.
    
    if weather['temperature'] > 38: alerts.append("HEATWAVE ALERT: High temperatures detected. Consider irrigating fields during early morning or late evening to minimize evaporation.")
    if weather['humidity'] > 85: alerts.append("HIGH HUMIDITY ALERT: Conditions are favorable for fungal diseases. Proactively inspect your crops for signs of blight or mildew.")
    if 'rain' in weather['description'].lower(): alerts.append("RAIN ALERT: Rain is expected. Postpone irrigation and ensure proper field drainage to prevent waterlogging.")
    if not alerts: alerts.append("Weather conditions are favorable for farming operations.")

    # Simple seasonal advice for Gujarat
    if current_month in [6, 7, 8, 9]: # Kharif Season
        calendar = ["Currently in the Kharif season (Monsoon). Focus on growth of crops like cotton, groundnut, and paddy. Ensure proper weed and pest management."]
    elif current_month in [10, 11, 12, 1, 2]: # Rabi Season
        calendar = ["Entering the Rabi season (Winter). Prepare fields for wheat, mustard, and cumin. Manage irrigation carefully due to winter conditions."]
    else: # Zaid/Summer Season
        calendar = ["This is the Zaid season (Summer), suitable for crops like moong and vegetables. Also a good time for soil testing and land preparation."]
        
    return {"weather": weather, "advisory": {"alerts": alerts, "calendar": calendar}}

def get_disease_remedy(condition):
    remedies = { "Potato___Early_blight": "Apply a copper-based fungicide.", "Potato___Late_blight": "Serious condition. Apply targeted fungicide immediately.", "Tomato___Late_blight": "Apply fungicide and avoid overhead watering." }
    return remedies.get(condition, "Consult a local agricultural expert.")

# --- Serve Frontend ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- API Endpoints ---
@app.route('/api/advisory', methods=['POST'])
def get_advisory():
    if not all([crop_rec_model, fert_rec_model]): return jsonify({"error": "Models not available."}), 500
    data = request.json
    lat, lon = data['lat'], data['lon']
    response_data = get_weather_and_advisory(lat, lon)
    weather_data = response_data['weather']
    n_val, p_val, k_val = data.get('n', 90), data.get('p', 45), data.get('k', 45)
    crop_input = np.array([[n_val, p_val, k_val, weather_data['temperature'], weather_data['humidity'], 7.0, 100.0]])
    
    if lat > 20.0:
        region_crops = GUJARAT_CROPS
    else:
        region_crops = MAHARASHTRA_CROPS

    all_crop_probabilities = crop_rec_model.predict_proba(crop_input)[0]
    model_classes = crop_rec_model.classes_
    
    region_crop_scores = {}
    for i, crop_name in enumerate(model_classes):
        if crop_name.lower() in region_crops:
            region_crop_scores[crop_name] = all_crop_probabilities[i]
            
    if not region_crop_scores:
        recommended_crop = crop_rec_model.predict(crop_input)[0]
    else:
        recommended_crop = max(region_crop_scores, key=region_crop_scores.get)

    fert_crop = recommended_crop.capitalize()
    if fert_crop not in known_crops_for_fert: fert_crop = 'Maize'
    fert_input = np.array([[weather_data['temperature'], weather_data['humidity'], 50, 'Sandy', fert_crop, n_val, k_val, p_val]])
    fert_input[:, 3] = soil_encoder.transform(fert_input[:, 3])
    fert_input[:, 4] = crop_encoder.transform(fert_input[:, 4])
    recommended_fertilizer = fert_rec_model.predict(fert_input)[0]
    response_data['recommendations'] = {"crop": recommended_crop.capitalize(), "fertilizer": recommended_fertilizer}
    return jsonify(response_data)

@app.route('/api/detect_disease', methods=['POST'])
def detect_disease():
    if not disease_det_model: return jsonify({"error": "Disease model not available."}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((128, 128))
        img_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0) / 255.0
        predictions = disease_det_model.predict(img_array)
        condition = disease_class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]) * 100)
        remedy = get_disease_remedy(condition)
        return jsonify({ "condition": condition.replace("___", " ").replace("_", " "), "confidence": f"{confidence:.2f}%", "remedy": remedy })
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/ask_groq', methods=['POST'])
def ask_groq_question():
    if not groq_client:
        return jsonify({"error": "Groq client is not configured on the server."}), 500
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        prompt = f"You are KrushiMitra, a friendly and helpful AI assistant for Indian farmers. Answer the following question in a simple, concise, and easy-to-understand way. Keep the context of farming in India in mind. Do not use markdown formatting like asterisks or hashtags. Question: {question}"
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful farming assistant for Indian farmers, named KrushiMitra. You always provide answers in plain text without any markdown formatting like * or #."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
        )
        answer = chat_completion.choices[0].message.content
        cleaned_answer = answer.replace('*', '').replace('#', '')
        return jsonify({"answer": cleaned_answer})
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return jsonify({"error": "Sorry, I had a problem getting an answer from the AI assistant."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)