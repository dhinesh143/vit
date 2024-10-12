import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Instantiate the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4)
model_path = r'C:\Users\Mazveen\Documents\vision_dks\model\vit_multiple_sclerosis_final.pth'

# Load your custom-trained model weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Class mapping
class_mapping = {0: 'Control-Axial', 1: 'Control-Sagittal', 2: 'MS-Axial', 3: 'MS-Sagittal'}

# Ensure the 'static/uploads' directory exists
os.makedirs('static/uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if a file has been uploaded
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', 'uploads', filename)
        file.save(filepath)

        # Load and preprocess the image
        image = Image.open(filepath).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        # Perform inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = class_mapping[predicted_class_idx]

        # Generate the image URL
        image_url = url_for('static', filename='uploads/' + filename)

        # Pass the predicted class and uploaded image URL to the result page
        return render_template('result.html', prediction=predicted_class, uploaded_image=image_url)

    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
