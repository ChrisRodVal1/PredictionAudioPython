import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import librosa.display

app = Flask(__name__)

# Define the directory paths
audio_directory = 'C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionVoz/audio/'
upload_directory = 'C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionVoz/static/uploads'

# Load the trained model
model = load_model('C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionVoz/audio_classification_model.h5')
print('Model loaded successfully.')

# Function to extract MFCC features from audio file
def extract_features(file_path):
    sample, sample_rate = librosa.load(file_path, sr=None)  # Load audio with original sample rate
    sample = librosa.resample(sample, orig_sr=sample_rate, target_sr=8000)  # Resample audio to 8000 Hz
    
    # Trim or pad the sample to ensure it's exactly 1 second long
    target_length = 8000  # 1 second at 8000 Hz
    if len(sample) < target_length:
        sample = np.pad(sample, (0, target_length - len(sample)))
    elif len(sample) > target_length:
        sample = sample[:target_length]
    
    # Check if the sample is exactly 1 second long
    if len(sample) == target_length: 
        mfcc = librosa.feature.mfcc(y=sample, sr=8000, n_mfcc=20)
        mfcc = np.transpose(mfcc)
        return np.expand_dims(mfcc, axis=0)
    else:
        return None



# Function to predict class for a given audio sample
def predict_audio(file_path):
    features = extract_features(file_path)
    if features is not None:
        prob = model.predict(features)
        index = np.argmax(prob[0])
        return classes[index]
    else:
        return "Audio sample length is not 1 second (8000 samples)"

# Route to render upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle file upload and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('upload.html', message='No file selected')  # Render upload form with message
        filename = secure_filename(f.filename)
        file_path = os.path.join(upload_directory, filename)
        f.save(file_path)
        predicted_class = predict_audio(file_path)

        # Generate plots
        sample, sample_rate = librosa.load(file_path, sr=32000)
        duration = len(sample) / sample_rate
        time_axis = np.linspace(0, duration, len(sample))
        
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        ax1.set_title('Audio Sample ' + filename)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.plot(time_axis, sample)
        plt.savefig(os.path.join(upload_directory, 'audio_sample_plot.png'))  # Save plot as image

        mfcc = librosa.feature.mfcc(y=sample, hop_length=512, n_mfcc=20)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(mfcc, x_axis='time', sr=32000)
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.savefig(os.path.join(upload_directory, 'mfcc_plot.png'))  # Save plot as image

        return render_template('predict.html', filename=filename, predicted_class=predicted_class)

if __name__ == '__main__':
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    classes = ["bird", "cat", "dog"]
    app.run(debug=True)
