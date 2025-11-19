import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

class AnimeClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = [
            "A Place Further Than The Universe",
            "A Silent Voice",
            "Angel Beats!",
            "Attack on Titan",
            "Bakemonogatari",
            "Chihayafuru",
            "Clannad",
            "Code Geass",
            "Death Note",
            "Death Parade",
            "Haikyu!!",
            "Hunter x Hunter",
            "Hyouka",
            "Kamisama Kiss",
            "Laid-Back Camp",
            "Maid Sama!",
            "My Teen Romantic Comedy",
            "Nana",
            "Neon Genesis Evangelion",
            "Nichijou",
            "No Game No Life",
            "Ping Pong the Animation",
            "ReLife",
            "ReZero",
            "Sound! Euphonium",
            "Steins Gate",
            "Tsukigakirei",
            "Violet Evergarden",
            "Yona of the Dawn",
            "Your Lie in April"
        ]
        self.num_classes = len(self.class_names)
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        # Initialize model architecture
        # We use weights=None because we are loading our own weights, 
        # but we need the structure. 
        # Note: In the notebook, weights="IMAGENET1K_V2" was used for initialization before modification.
        # Here we just need the structure to load the state_dict.
        model = models.resnet50(weights=None) 
        
        # Modify the final layer as per the notebook
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes),
        )

        # Load the trained weights
        # map_location ensures it loads on CPU if CUDA is not available
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

    def predict(self, image_file):
        image = Image.open(image_file).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
