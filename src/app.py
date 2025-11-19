import os
import base64
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from model import AnimeClassifier

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_resnet50_final.pt')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'anime_cleaned.csv')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the classifier
try:
    classifier = AnimeClassifier(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

# Load Anime Metadata
ANIME_METADATA = {}
try:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        
        if classifier:
            for name in classifier.class_names:

                row = df[df['title'] == name]
                if row.empty:
                    row = df[df['title_english'] == name]
                
                if not row.empty:
                    data = row.iloc[0]
                    ANIME_METADATA[name] = {
                        'anime_id': data.get('anime_id'),
                        'title_synonyms': data.get('title_synonyms'),
                        'type': data.get('type'),
                        'aired_from_year': data.get('aired_from_year'),
                        'source': data.get('source'),
                        'episodes': data.get('episodes'),
                        'status': data.get('status'),
                        'rating': data.get('rating'),
                        'score': data.get('score'),
                        'studio': data.get('studio'),
                        'genre': data.get('genre')
                    }
                else:
                    print(f"Warning: No metadata found for class '{name}'")
    else:
        print(f"Warning: Data file not found at {DATA_PATH}")
except Exception as e:
    print(f"Error loading metadata: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if classifier:
                try:
                    # read file for prediction, encode for display, reset stream for PIL
                    file_bytes = file.read()
                    
                    image_data = base64.b64encode(file_bytes).decode('utf-8')
                    
                    from io import BytesIO
                    file_stream = BytesIO(file_bytes)
                    
                    predictions = classifier.predict(file_stream)
                    

                    predicted_class, confidence = predictions[0]
                    other_predictions = predictions[1:]
                    
                    anime_info = ANIME_METADATA.get(predicted_class, {})
                    
                    return render_template('result.html', 
                                           anime_title=predicted_class,
                                           confidence=f"{confidence*100:.2f}%",
                                           anime_info=anime_info,
                                           image_data=image_data,
                                           other_predictions=other_predictions)
                except Exception as e:
                    return f"An error occurred during prediction: {e}"
            else:
                return "Model not loaded. Please check server logs."
    return render_template('index.html', allowed_extensions=ALLOWED_EXTENSIONS)

if __name__ == '__main__':
    app.run(debug=True)
