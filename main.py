import os
import torch
from flask import Flask, request, render_template, url_for
from transformers import ViTForImageClassification, ViTConfig
from PIL import Image
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load ViT model with 4 output labels (adjust if needed)
config = ViTConfig(num_labels=4)  # Match number of output labels
model = ViTForImageClassification(config)

# Load the saved model weights
model_path = r'C:\Users\Mazveen\Documents\vision_dks\model\vit_multiple_sclerosis_final.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Adjust path
model.eval()  # Set model to evaluation mode

# Load class mapping (assuming 4 classes)
class_mapping = {0: 'Control-Axial', 1: 'Control-Sagittal', 2: 'MS-Axial', 3: 'MS-Sagittal'}

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Load and preprocess the image
    image = Image.open(file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor).logits
        _, predicted = torch.max(outputs, 1)

    # Get the predicted class name
    predicted_class = class_mapping[predicted.item()]

    # Save the uploaded image to the static folder for display
    uploaded_image_path = os.path.join('static', 'uploads', file.filename)
    image.save(uploaded_image_path)

    # Pass the predicted class and uploaded image path to the result page
    return render_template('index.html', prediction=predicted_class, uploaded_image=url_for('static', filename=f'uploads/{file.filename}'))

if __name__ == '__main__':
    # Ensure the static/uploads directory exists
    os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)
    app.run(debug=True)
