import os
import json
import numpy as np
import time
import tensorflow as tf
from PIL import Image
from django.conf import settings
import gdown
import gc
from datetime import datetime, timedelta

class CropModelLoader:
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'models')
        self.recommendations_path = os.path.join(settings.BASE_DIR, 'crop_diseases_detailed.json')
        os.makedirs(self.model_dir, exist_ok=True)

        self.crop_models = {
            "maize": {
                "filename": "maize_model.h5",
                "classes": [
                    "abiotics_disease_d", "aphids_p", "curvulariosis_d",
                    "healthy_leaf", "helminthosporiosis_d", "rust_d",
                    "spodoptera_frugiperda_p", "stripe_d", "virosis_d"
                ],
                "gdrive_id": "1nGh0rM1_hHM_8lQpo4M4J1URsl4a53Aw",
                "input_size": (224, 224)
            },
            "onion": {
                "filename": "onion_model.h5",
                "classes": [
                    "alternaria_d", "bulb_blight_d", "fusarium_d",
                    "healthy_leaf", "virosis_d", "caterpillar_p"
                ],
                "gdrive_id": "1MJ6pSTNCCcwJ0WLkf4hvnu1mivoRd9CY",
                "input_size": (299, 299)
            },
            "tomato": {
                "filename": "tomato_model.h5",
                "classes": [
                    "alternaria_d", "alternaria_mite_d", "bacterial_floundering_d",
                    "blossom_end_rot_d", "exces_nitrogen_d", "fusarium_d",
                    "healthy_leaf", "mite_d", "sunburn_d", "tomato_late_blight_d",
                    "virosis_d", "helicoverpa_armigera_p", "tuta_absoluta_p"
                ],
                "gdrive_id": "131dXFO87Y4w-eUiwrU5OtBUa5S3bqfFq",
                "input_size": (299, 299) 
            }
        }

        # Changed from dict to track model usage and last access time
        self.models = {}
        self.model_last_used = {}
        self.max_models_in_memory = 1  # Only keep one model in memory at a time
        self.model_timeout = 300  # Unload models after 5 minutes of inactivity

        # Load recommendations once at startup
        try:
            with open(self.recommendations_path, 'r') as f:
                self.recommendations = json.load(f)
        except Exception as e:
            print(f"Error loading recommendations: {e}")
            self.recommendations = {"crops": {}}

    def download_model_from_gdrive(self, file_id, dest_path, max_retries=3):
        """
        Downloads a model file from Google Drive with retry logic and better error handling.
        
        Args:
            file_id (str): The Google Drive file ID
            dest_path (str): Path where the file should be saved
            max_retries (int): Maximum number of download attempts
        
        Returns:
            bool: True if download was successful
        """
        print(f"Downloading model from Google Drive to: {dest_path}")
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        
        # Ensure any partial downloads are removed
        if os.path.exists(dest_path):
            os.remove(dest_path)
        
        # Try to download with retries
        for attempt in range(max_retries):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                
                # Use cookies to avoid Google Drive download quota issues
                gdown.download(url, dest_path, quiet=False, fuzzy=True)
                
                # Verify download was successful
                if not os.path.exists(dest_path):
                    print(f"Attempt {attempt+1}/{max_retries}: File does not exist after download")
                    continue
                    
                file_size = os.path.getsize(dest_path)
                if file_size < 10000:  # 10KB minimum as a sanity check
                    print(f"Attempt {attempt+1}/{max_retries}: File too small ({file_size} bytes), likely corrupted")
                    os.remove(dest_path)
                    continue
                
                print(f"Model downloaded successfully: {dest_path} ({file_size:,} bytes)")
                return True
                
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries}: Download failed with error: {str(e)}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                
                # Wait before retrying (with exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4, 8... seconds
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        # If we get here, all attempts failed
        raise ValueError(f"Failed to download file after {max_retries} attempts. Please check file ID and permissions.")

    def unload_inactive_models(self):
        """Unload models that haven't been used recently"""
        current_time = datetime.now()
        models_to_remove = []
        
        for crop, last_used in self.model_last_used.items():
            if (current_time - last_used).total_seconds() > self.model_timeout:
                models_to_remove.append(crop)
        
        for crop in models_to_remove:
            if crop in self.models:
                print(f"Unloading inactive model for '{crop}'")
                del self.models[crop]
                tf.keras.backend.clear_session()
                gc.collect()  # Force garbage collection
    
    def manage_model_memory(self, current_crop):
        """Ensure we don't exceed the maximum number of models in memory"""
        # First mark this model as recently used
        self.model_last_used[current_crop] = datetime.now()
        
        # Check if we need to unload any models
        if len(self.models) > self.max_models_in_memory:
            # Find the least recently used model that's not the current one
            least_recent = None
            oldest_time = datetime.now()
            
            for crop, last_used in self.model_last_used.items():
                if crop != current_crop and crop in self.models and last_used < oldest_time:
                    least_recent = crop
                    oldest_time = last_used
            
            # Unload the least recently used model
            if least_recent:
                print(f"Unloading model '{least_recent}' to free memory")
                del self.models[least_recent]
                tf.keras.backend.clear_session()
                gc.collect()  # Force garbage collection

    def load_model(self, crop):
        """Load model with memory management"""
        if crop not in self.crop_models:
            raise ValueError(f"No model found for crop: {crop}")

        # Unload any inactive models first
        self.unload_inactive_models()
        
        model_info = self.crop_models[crop]
        model_path = os.path.join(self.model_dir, model_info["filename"])

        if not os.path.exists(model_path):
            self.download_model_from_gdrive(model_info["gdrive_id"], model_path)

        if crop not in self.models:
            try:
                # Clear session before loading new model to free memory
                tf.keras.backend.clear_session()
                
                # Try to load a quantized/optimized version first if available
                optimized_path = model_path.replace('.h5', '_quantized.tflite')
                if os.path.exists(optimized_path):
                    print(f"Loading optimized TFLite model for '{crop}'")
                    interpreter = tf.lite.Interpreter(model_path=optimized_path)
                    interpreter.allocate_tensors()
                    self.models[crop] = {
                        'type': 'tflite',
                        'interpreter': interpreter,
                        'input_details': interpreter.get_input_details(),
                        'output_details': interpreter.get_output_details()
                    }
                else:
                    # Load regular model with reduced precision if possible
                    print(f"Loading regular model for '{crop}'")
                    model = tf.keras.models.load_model(model_path)
                    self.models[crop] = {
                        'type': 'keras',
                        'model': model
                    }
                
                print(f"Model for '{crop}' loaded successfully.")
                
                # Manage memory to ensure we don't exceed limits
                self.manage_model_memory(crop)
                
            except Exception as e:
                print(f"Failed to load model for '{crop}' from {model_path}")
                raise e

        # Update last used timestamp
        self.model_last_used[crop] = datetime.now()
        
        return self.models[crop], model_info["classes"], model_info["input_size"]

    def preprocess_image(self, image_path, target_size):
        """Preprocess image with the correct target size for the specific model"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        
        # Return as float32 to reduce memory usage (vs float64)
        return np.expand_dims(img_array, axis=0).astype('float32')

    def predict(self, image_path, selected_crop):
        if selected_crop not in self.crop_models:
            return {'error': f"Unsupported crop: {selected_crop}"}

        try:
            model_data, classes, input_size = self.load_model(selected_crop)
            image = self.preprocess_image(image_path, input_size)

            # Handle different model types (regular Keras or TFLite)
            if model_data['type'] == 'tflite':
                interpreter = model_data['interpreter']
                input_details = model_data['input_details']
                output_details = model_data['output_details']
                
                interpreter.set_tensor(input_details[0]['index'], image)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            else:
                # Regular Keras model
                predictions = model_data['model'].predict(image)[0]

            idx = np.argmax(predictions)
            confidence = float(predictions[idx]) * 100
            label = classes[idx]

            # Get recommendation data with better error handling
            crop_data = self.recommendations.get("crops", {}).get(selected_crop, {})
            if not crop_data:
                print(f"Warning: No crop data found for {selected_crop}")
                return {
                    'disease': label.replace("_", " ").title(),
                    'subclass': selected_crop.title(),
                    'confidence': confidence,
                    'cause': 'Crop data not available',
                    'remedy_organic': 'Information not available',
                    'remedy_chemical': 'Information not available',
                    'selected_crop': selected_crop
                }
            
            # Try exact match first
            recommendation = crop_data.get(label)
            
            # Fallback: try fuzzy match (e.g., prefix match)
            if recommendation is None:
                for key in crop_data:
                    if key.lower() == label.lower():
                        recommendation = crop_data[key]
                        break
            
            # Handle case when recommendation is still None
            if recommendation is None:
                print(f"Warning: No recommendation found for {selected_crop}/{label}")
                return {
                    'disease': label.replace("_", " ").title(),
                    'subclass': selected_crop.title(),
                    'confidence': confidence,
                    'cause': 'Recommendation not available',
                    'remedy_organic': 'Information not available',
                    'remedy_chemical': 'Information not available',
                    'selected_crop': selected_crop
                }

            disease = label.replace("_", " ").title()

            # Safe access to nested data with defaults
            treatments = recommendation.get('treatment', {}) or {}
            organic_treatments = treatments.get('organic', 'Information not available')
            chemical_treatments = treatments.get('chemical', 'Information not available')

            if isinstance(organic_treatments, list):
                organic_treatments = "\n".join(organic_treatments)
            if isinstance(chemical_treatments, list):
                chemical_treatments = "\n".join(chemical_treatments)

            return {
                'disease': disease,
                'subclass': selected_crop.title(),
                'confidence': confidence,
                'cause': recommendation.get('cause', 'Information not available'),
                'remedy_organic': organic_treatments,
                'remedy_chemical': chemical_treatments,
                'selected_crop': selected_crop
            }

        except Exception as e:
            import traceback
            print("Prediction error:", e)
            traceback.print_exc()
            return {
                'disease': 'Error in prediction',
                'subclass': selected_crop.title(), 
                'confidence': 0,
                'cause': f'Error: {str(e)}',
                'remedy_organic': 'Please try again with a different image',
                'remedy_chemical': 'Please try again with a different image',
                'selected_crop': selected_crop
            }
        # finally:
            # Consider unloading the model immediately if memory is critical
            # Uncomment if you're still having memory issues
            # if selected_crop in self.models:
            #    del self.models[selected_crop]
            #    tf.keras.backend.clear_session()
            #    gc.collect()

    def get_crop_options(self):
        return list(self.crop_models.keys())


# Use a model unloading timer to ensure models don't stay in memory forever
def create_model_unloader(predictor, interval=60):
    """Create a background thread to unload inactive models periodically"""
    import threading
    
    def unload_timer():
        while True:
            try:
                predictor.unload_inactive_models()
            except Exception as e:
                print(f"Error in model unloader: {e}")
            time.sleep(interval)
    
    # Start the thread
    thread = threading.Thread(target=unload_timer, daemon=True)
    thread.start()
    return thread


# Singleton with unloader
predictor = None
unloader_thread = None

def get_predictor():
    global predictor, unloader_thread
    if predictor is None:
        predictor = CropModelLoader()
        # Start the model unloader thread
        unloader_thread = create_model_unloader(predictor)
    return predictor