# Smart Crop Advisory System - Integrated & Automated ML Models (Version 4 - Local Dataset)
# This version is optimized to use a pre-downloaded PlantVillage dataset, skipping the download step.
# It also includes the fix for the "unseen label" error.

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from datetime import datetime
import shutil

# Scikit-learn for recommendation models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# TensorFlow/Keras for disease detection model
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

# --- PART 1: CROP RECOMMENDATION MODEL ---

class CropRecommender:
    """
    Builds and uses a model to recommend crops based on environmental factors.
    """
    def __init__(self, model_path='models/crop_recommender.pkl'):
        self.model_path = model_path
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model = None

    def train(self, data_path='data/Crop_recommendation.csv'):
        print("Training Crop Recommendation Model...")
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found at {data_path}. Please place it in the 'data' folder.")
            return False
        df = pd.read_csv(data_path)
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Crop Recommender Model Accuracy: {accuracy * 100:.2f}%")
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Crop Recommender Model saved to {self.model_path}")
        self.model = model
        return True

    def predict(self, input_data):
        if self.model is None:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                print("Model not trained yet. Please call train() first.")
                return None
        return self.model.predict(input_data)[0]

# --- PART 2: FERTILIZER RECOMMENDATION MODEL (WITH FIX) ---

class FertilizerRecommender:
    """
    Builds and uses a model to recommend fertilizers.
    Now saves its list of known crops to prevent "unseen label" errors.
    """
    def __init__(self, model_path='models/fertilizer_recommender.pkl', encoder_paths=None):
        if encoder_paths is None:
            encoder_paths = {
                'soil': 'models/soil_encoder.pkl', 
                'crop': 'models/crop_encoder.pkl',
                'known_crops': 'models/known_crops.pkl'
            }
        self.model_path = model_path
        self.encoder_paths = encoder_paths
        self.model = None
        self.soil_encoder = None
        self.crop_encoder = None
        self.known_crops = None

    def train(self, data_path='data/Fertilizer Prediction.csv'):
        print("\nTraining Fertilizer Recommendation Model...")
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found at {data_path}. Please place it in the 'data' folder.")
            return False
        df = pd.read_csv(data_path)
        self.known_crops = df['Crop Type'].unique().tolist()
        self.soil_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()
        df['Soil Type'] = self.soil_encoder.fit_transform(df['Soil Type'])
        df['Crop Type'] = self.crop_encoder.fit_transform(df['Crop Type'])
        X = df[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
        y = df['Fertilizer Name']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Fertilizer Recommender Model Accuracy: {accuracy * 100:.2f}%")
        with open(self.model_path, 'wb') as f: pickle.dump(model, f)
        with open(self.encoder_paths['soil'], 'wb') as f: pickle.dump(self.soil_encoder, f)
        with open(self.encoder_paths['crop'], 'wb') as f: pickle.dump(self.crop_encoder, f)
        with open(self.encoder_paths['known_crops'], 'wb') as f: pickle.dump(self.known_crops, f)
        print(f"Fertilizer Recommender Model, encoders, and known crop list saved.")
        self.model = model
        return True

    def predict(self, input_data):
        if self.model is None:
            self.load_model_and_encoders()
        if self.model is None:
             print("Model not trained yet. Please call train() first."); return None
        input_df = pd.DataFrame([input_data], columns=['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
        input_df['Soil Type'] = self.soil_encoder.transform(input_df['Soil Type'])
        input_df['Crop Type'] = self.crop_encoder.transform(input_df['Crop Type'])
        return self.model.predict(input_df)[0]

    def load_model_and_encoders(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f: self.model = pickle.load(f)
            with open(self.encoder_paths['soil'], 'rb') as f: self.soil_encoder = pickle.load(f)
            with open(self.encoder_paths['crop'], 'rb') as f: self.crop_encoder = pickle.load(f)
            with open(self.encoder_paths['known_crops'], 'rb') as f: self.known_crops = pickle.load(f)

# --- PART 3: PEST/DISEASE DETECTION MODEL (MODIFIED FOR LOCAL DATA) ---

class DiseaseDetector:
    def __init__(self, model_path='models/disease_detector.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = None

    def _setup_dataset(self, data_dir='data', subset_dir='data/plantvillage_subset'):
        """
        Creates a subset of the PlantVillage dataset from a local folder.
        """
        classes_to_use = ['Tomato___Late_blight', 'Tomato___healthy', 'Potato___Early_blight', 'Potato___healthy']
        
        # Path to your manually downloaded repository
        source_repo_path = os.path.join(data_dir, 'PlantVillage-Dataset')
        
        if not os.path.exists(source_repo_path):
            print("\nERROR: Local PlantVillage-Dataset folder not found!")
            print(f"Please make sure your downloaded repository is inside the '{data_dir}' folder.")
            return False

        if os.path.exists(subset_dir):
            print("\nDataset subset already exists. Skipping creation.")
            return True

        print("\nCreating a smaller, focused dataset from your local repository...")
        os.makedirs(subset_dir, exist_ok=True)
        # The images are in the 'raw/color' subdirectory of the repository
        source_images_dir = os.path.join(source_repo_path, 'raw', 'color')

        for class_name in classes_to_use:
            source_class_path = os.path.join(source_images_dir, class_name)
            destination_class_path = os.path.join(subset_dir, class_name)
            if os.path.exists(source_class_path):
                shutil.copytree(source_class_path, destination_class_path)
            else:
                print(f"Warning: Class folder '{class_name}' not found in the source repository.")

        print("Subset created successfully.")
        return True

    def train(self, subset_dir='data/plantvillage_subset'):
        print("\nTraining Disease Detection Model (This will take time)...")
        if not self._setup_dataset():
            print("Dataset setup failed. Aborting training."); return False
        
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        train_generator = datagen.flow_from_directory(subset_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')
        validation_generator = datagen.flow_from_directory(subset_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')
        self.class_names = list(train_generator.class_indices.keys())
        num_classes = len(self.class_names)
        with open('models/disease_class_names.pkl', 'wb') as f: pickle.dump(self.class_names, f)
        
        model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), MaxPooling2D(2, 2), Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(2, 2), Conv2D(128, (3, 3), activation='relu'), MaxPooling2D(2, 2), Flatten(), Dense(512, activation='relu'), Dropout(0.5), Dense(num_classes, activation='softmax')])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=10, validation_data=validation_generator)
        model.save(self.model_path)
        print(f"Disease Detector Model saved to {self.model_path}")
        self.model = model
        return True

    def predict(self, image_path):
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                with open('models/disease_class_names.pkl', 'rb') as f: self.class_names = pickle.load(f)
            else: return None, 0.0
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) / 255.0
            predictions = self.model.predict(img_array)
            predicted_class = self.class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            return predicted_class, confidence
        except Exception as e:
            return f"Error processing image: {e}", 0.0

# --- MAIN EXECUTION AND DEMONSTRATION ---

def main():
    print("--- SMART CROP ADVISORY SYSTEM V4 (LOCAL DATASET) ---")
    
    # --- Step 1: Training Phase ---
    crop_rec = CropRecommender()
    if not os.path.exists(crop_rec.model_path):
        if not crop_rec.train(): return

    fert_rec = FertilizerRecommender()
    if not os.path.exists(fert_rec.model_path):
        if not fert_rec.train(): return
    else:
        fert_rec.load_model_and_encoders()

    disease_det = DiseaseDetector()
    if not os.path.exists(disease_det.model_path):
        if not disease_det.train(): 
             print("Skipping disease detection due to training failure.")
    
    # --- Step 2: Advisory Simulation ---
    print("\n--- RUNNING ADVISORY SIMULATION ---")
    print(f"Location: Ahmedabad, Gujarat, India | Date: {datetime.now().strftime('%Y-%m-%d')}")
    simulated_N, simulated_P, simulated_K = 90, 45, 45
    simulated_temp, simulated_humidity = 32.0, 65.0
    simulated_ph, simulated_rainfall = 7.5, 80.0
    simulated_moisture, simulated_soil_type = 55.0, 'Sandy'

    print("\n[1] CROP RECOMMENDATION")
    crop_input = [[simulated_N, simulated_P, simulated_K, simulated_temp, simulated_humidity, simulated_ph, simulated_rainfall]]
    recommended_crop = crop_rec.predict(crop_input)
    print(f"-> Based on your soil and local weather, the recommended crop is: {recommended_crop.upper()}")

    print("\n[2] FERTILIZER RECOMMENDATION")
    
    final_crop_for_fertilizer = recommended_crop
    if recommended_crop.capitalize() not in fert_rec.known_crops:
        print(f"   - WARNING: '{recommended_crop}' is not in the fertilizer database.")
        final_crop_for_fertilizer = 'Maize' 
        print(f"   - Providing a general recommendation for a common crop instead: '{final_crop_for_fertilizer}'.")

    fert_input = [simulated_temp, simulated_humidity, simulated_moisture, simulated_soil_type, final_crop_for_fertilizer.capitalize(), simulated_N, simulated_K, simulated_P]
    recommended_fertilizer = fert_rec.predict(fert_input)
    print(f"-> For a '{final_crop_for_fertilizer}' crop, the recommended fertilizer is: {recommended_fertilizer}")
    
    print("\n[3] DISEASE DETECTION")
    test_image_path = 'test_images/test_potato_leaf.JPG'
    if os.path.exists(disease_det.model_path) and os.path.exists(test_image_path):
        disease, confidence = disease_det.predict(test_image_path)
        if disease:
            print(f"-> Analysis of '{test_image_path}':")
            print(f"   Detected Condition: {disease.replace('___', ' ')}")
            print(f"   Confidence: {confidence*100:.2f}%")
    else:
        print("-> Disease detection skipped (model not trained or test image not found).")

if __name__ == '__main__':
    main()