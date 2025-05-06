import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from django.conf import settings
from pathlib import Path
from .cnn_model import SimpleCNN
import urllib.request

def download_model_if_needed():
    model_path = 'simple_CNN_weights.pth'
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        urllib.request.urlretrieve(
            'YOUR_CLOUD_STORAGE_URL/simple_CNN_weights.pth',
            model_path
        )
        print("Model downloaded successfully")

# Call this function before loading your model
download_model_if_needed()

class DiseasePredictor:
    def __init__(self):
        model_path = os.path.join(settings.BASE_DIR, 'simple_CNN_weights.pth')
        recommendations_path = os.path.join(settings.BASE_DIR, 'agriculture_recommendations.json')

        with open(recommendations_path, 'r') as f:
            self.recommendations = json.load(f)

        # Define the main categories (7 classes) that the model was trained on
        self.main_categories = [
            "maize_diseases",
            "maize_pests",
            "maize_pests_activities",
            "onion_diseases",
            "onion_pests",
            "tomato_diseases",
            "tomato_pests"
        ]
        
        # Define crop outer classes
        self.crop_outer_classes = {
            "maize": ["maize_diseases", "maize_pests", "maize_pests_activities"],
            "onion": ["onion_diseases", "onion_pests"],
            "tomato": ["tomato_diseases", "tomato_pests"]
        }

        # Define the hierarchical structure mapping
        self.category_to_subclasses = {
            "maize_diseases": ["abiotic_disease_d", "curvularia_d", "healthy_leaf", 
                              "helminthosporiosis_d", "rust_d", "stripe_d", "virosis_d"],
            "maize_pests": ["aphids_p", "spodoptera_frugiperda_p"],
            "maize_pests_activities": ["spodotera_frugiperda_a"],
            "onion_diseases": ["alternaria_d", "bulb_blight_d", "fusarium_d", 
                              "healthy_leaf", "virosis_d"],
            "onion_pests": ["caterpillar_p"],
            "tomato_diseases": ["alternaria_d", "alternaria_mite_d", "bacterial_floundering_d", 
                               "blossom_end_rot_d", "exces_nitrogen_d", "fusarium_d", 
                               "healthy_fruit", "healthy_leaf", "mite_d", "sunburn_d", 
                               "tomato_late_blight_d", "virosis_d"],
            "tomato_pests": ["helicoverpa_armigera_p", "tuta_absoluta_p"]
        }

        # Create a flattened list of all class names in format "category___subclass"
        self.flattened_classes = []
        for category, subclasses in self.category_to_subclasses.items():
            for subclass in subclasses:
                self.flattened_classes.append(f"{category}___{subclass}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the model with the number of main categories (7)
        num_classes = len(self.main_categories)
        self.model = SimpleCNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded successfully on {self.device}!")
        print(f"Using {num_classes} main categories with {len(self.flattened_classes)} total subclasses")

    def preprocess_image(self, image_path):
        """Preprocess the image for model input"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor.to(self.device)

    def predict(self, image_path, selected_crop=None):
        """
        Make prediction on the input image with optional crop selection
        
        Args:
            image_path: Path to the image file
            selected_crop: Optional crop selection ('maize', 'onion', or 'tomato')
                           to filter prediction categories
        """
        try:
            img = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.model(img)
                
                # Get probabilities for all main categories
                category_probs = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # If a crop is selected, filter prediction to only include relevant categories
                if selected_crop and selected_crop in self.crop_outer_classes:
                    allowed_categories = self.crop_outer_classes[selected_crop]
                    
                    # Create a mask for allowed categories
                    mask = torch.zeros_like(category_probs)
                    for i, category in enumerate(self.main_categories):
                        if category in allowed_categories:
                            mask[i] = 1.0
                    
                    # Apply mask to zero out probabilities for non-relevant categories
                    masked_probs = category_probs * mask
                    
                    # Get the highest probability among allowed categories
                    if torch.max(masked_probs) > 0:
                        _, category_index = torch.max(masked_probs, 0)
                        category_confidence = float(masked_probs[category_index.item()]) * 100
                    else:
                        # Fallback if no allowed category has probability > 0
                        _, category_index = torch.max(category_probs, 0)
                        category_confidence = float(category_probs[category_index.item()]) * 100
                        print(f"Warning: No strong match found for {selected_crop}, using best overall prediction")
                else:
                    # If no crop filter, use the highest probability category
                    _, category_index = torch.max(category_probs, 0)
                    category_confidence = float(category_probs[category_index.item()]) * 100
                
                # Get the predicted main category
                main_category = self.main_categories[category_index.item()]
                
                # For now, we'll use the main category prediction
                # In a more advanced implementation, you might want to use a second model
                # to predict the specific subclass within the main category
                
                # For demonstration, we'll select the first subclass in the category
                # You should replace this with an actual subclass prediction logic
                # or implement a user interface to select the specific subclass
                possible_subclasses = self.category_to_subclasses[main_category]
                subclass = possible_subclasses[0]  # Default to first subclass
                
                # Format the result
                category = main_category.replace("_", " ").title()
                disease = subclass.replace("_", " ").title()
                
                # Get recommendation
                recommendation = self.recommendations.get(main_category, {}).get(subclass, {})

            return {
                'disease': disease,
                'subclass': category,  # Note: In your system, the "subclass" field shows the main category
                'confidence': category_confidence,
                'cause': recommendation.get('description', 'Information not available'),
                'remedy': "\n".join(recommendation.get('management', ['Information not available'])),
                'healthy_image': recommendation.get('healthy_image', None),
                'possible_subclasses': possible_subclasses,
                'selected_crop': selected_crop
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'disease': 'Error in prediction',
                'subclass': 'unknown',
                'confidence': 0,
                'cause': f'Error: {str(e)}',
                'remedy': 'Please try again with a different image',
                'healthy_image': None,
                'possible_subclasses': [],
                'selected_crop': selected_crop
            }

    def predict_with_subclass(self, image_path, selected_crop=None, selected_subclass=None):
        """
        Make prediction with optional crop selection and specific subclass selection.
        
        Args:
            image_path: Path to the image file
            selected_crop: Optional crop selection ('maize', 'onion', or 'tomato')
            selected_subclass: If provided, use this subclass for detailed results
        """
        result = self.predict(image_path, selected_crop)
        
        if selected_subclass is not None and 'possible_subclasses' in result:
            if selected_subclass in result['possible_subclasses']:
                main_category = result['subclass'].lower().replace(" ", "_")
                
                # Update disease name
                result['disease'] = selected_subclass.replace("_", " ").title()
                
                # Update recommendation
                recommendation = self.recommendations.get(main_category, {}).get(selected_subclass, {})
                result['cause'] = recommendation.get('description', 'Information not available')
                result['remedy'] = "\n".join(recommendation.get('management', ['Information not available']))
                result['healthy_image'] = recommendation.get('healthy_image', None)
        
        return result
    
    def get_crop_options(self):
        """Return the available crop options for the user interface"""
        return list(self.crop_outer_classes.keys())

# Create a singleton instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = DiseasePredictor()
    return predictor