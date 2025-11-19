import os
from flask import Flask, request, render_template, redirect, url_for
from model import AnimeClassifier

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained_resnet50_final.pt')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the classifier
# We initialize it once when the app starts
try:
    classifier = AnimeClassifier(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if classifier:
                try:
                    predicted_class, confidence = classifier.predict(file)
                    return render_template('result.html', 
                                           anime_title=predicted_class, 
                                           confidence=f"{confidence*100:.2f}%")
                except Exception as e:
                    return f"An error occurred during prediction: {e}"
            else:
                return "Model not loaded. Please check server logs."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
