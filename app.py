import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid
import traceback

app = Flask(__name__)

# =========================
# CREATE UPLOADS FOLDER
# =========================

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = load_model('model/tomato_leaf_disease_model.h5')

print("Model loaded successfully!")

# =========================
# CLASS NAMES
# =========================

classes = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Yellow Leaf Curl Virus',
    'Mosaic Virus',
    'Healthy'
]

# =========================
# DISEASE INFORMATION
# =========================

disease_info = {

    'Bacterial Spot': {
        'description': 'A bacterial disease causing dark water-soaked spots on leaves.',
        'treatment': 'Remove infected leaves and apply copper-based bactericides.'
    },

    'Early Blight': {
        'description': 'A fungal disease causing brown concentric spots on leaves.',
        'treatment': 'Apply fungicide and remove infected leaves.'
    },

    'Late Blight': {
        'description': 'A severe fungal disease causing dark lesions and rapid decay.',
        'treatment': 'Use copper fungicides and avoid excessive moisture.'
    },

    'Leaf Mold': {
        'description': 'A fungal disease producing yellow spots and mold growth.',
        'treatment': 'Improve ventilation and apply fungicide.'
    },

    'Septoria Leaf Spot': {
        'description': 'A fungal infection causing small circular spots on leaves.',
        'treatment': 'Remove infected leaves and avoid overhead watering.'
    },

    'Spider Mites': {
        'description': 'Tiny pests that damage leaves by sucking plant fluids.',
        'treatment': 'Apply insecticidal soap or miticides.'
    },

    'Target Spot': {
        'description': 'A fungal disease causing circular target-like spots.',
        'treatment': 'Use fungicides and maintain proper spacing.'
    },

    'Yellow Leaf Curl Virus': {
        'description': 'A viral disease causing yellowing and curling leaves.',
        'treatment': 'Control whiteflies and remove infected plants.'
    },

    'Mosaic Virus': {
        'description': 'A viral disease causing mottled leaf discoloration.',
        'treatment': 'Remove infected plants and sanitize tools.'
    },

    'Healthy': {
        'description': 'The tomato leaf appears healthy.',
        'treatment': 'No treatment needed.'
    }
}

# =========================
# ALLOWED FILE TYPES
# =========================

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():
    return render_template('index.html')

# =========================
# PREDICTION ROUTE
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        print("Prediction started...")

        # CHECK FILE EXISTENCE
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']

        # CHECK EMPTY FILE
        if file.filename == '':
            return "No selected file"

        # CHECK FILE TYPE
        if not allowed_file(file.filename):
            return "Invalid file type"

        # GENERATE UNIQUE FILENAME
        filename = str(uuid.uuid4()) + ".jpg"

        # CREATE FILE PATH
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # SAVE IMAGE
        print("Saving uploaded image...")
        file.save(filepath)

        # LOAD IMAGE
        print("Loading image...")

        img = image.load_img(filepath, target_size=(128, 128))

        # PREPROCESS IMAGE
        img_array = image.img_to_array(img)

        img_array = img_array.astype("float32") / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        print("Image preprocessing complete.")

        # PREDICT
        print("Running model prediction...")

        prediction = model.predict(img_array)

        print("Prediction complete!")

        # GET RESULTS
        confidence = round(np.max(prediction) * 100, 2)

        predicted_class = classes[np.argmax(prediction)]

        print("Predicted Class:", predicted_class)

        # DELETE IMAGE AFTER PREDICTION
        if os.path.exists(filepath):
            os.remove(filepath)

        # GET DISEASE INFO
        description = disease_info[predicted_class]['description']

        treatment = disease_info[predicted_class]['treatment']

        # RETURN RESULTS
        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence=confidence,
            description=description,
            treatment=treatment
        )

    except Exception as e:

        traceback.print_exc()

        print("ERROR OCCURRED:")
        print(str(e))

        return f"""
        <h1>Error Occurred</h1>
        <p>{str(e)}</p>
        """

# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)