import os
import librosa   # pip install librosa 
import IPython.display as ipd # library to play audio
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# First, let's load an audio sample into the Jupyter notebook
directory = 'C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionVoz/audio/bird/'
sample, sample_rate = librosa.load(directory+'00b01445_nohash_0.wav', sr=16000) # loads the specified audio with a sample rate of 16,000 frames per second. In 'sample', we'll have the magnitude of each point in space, and in 'sample_rate', we'll have the specified sample rate.
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(2,1,1)
ax1.set_title('Audio Sample ' + '00b01445_nohash_0.wav')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, 1, sample_rate), sample) 

# np.linspace generates equally spaced numbers within an interval. For example: np.linspace(2.0, 3.0, 5) will generate 5 equally spaced numbers
# located between 2.0 and 3.0, which would be: [2., 2.25, 2.5, 2.75, 3.].
# In this case, I'm creating on the x-axis sample_rate points equally spaced between zero and 1, plotting the magnitude values
# of the sample in question.

len(sample)

# Creating and visualizing the MFCC spectrum:
mfcc = librosa.feature.mfcc(y=sample, hop_length=512, n_mfcc=20) # hop_length is how many times I will subdivide the sample to create the MFCC features
# The calculation is (seconds x frames/seconds)/hop_length. For example, if hop_length=512 (default), I have 1 x 16000/512 = 32

mfcc.shape

# Showing the MFCC:
import librosa.display
plt.figure(figsize=(8, 5))
librosa.display.specshow(mfcc, x_axis='time', sr=16000)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# Listening to a sample within the Jupyter notebook:
ipd.Audio(sample, rate=16000)

# Visualizing how many samples are in each class:
directory = 'C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionVoz/audio/'
classes = os.listdir(directory)
number_audios = []
for class_ in classes: # for each class bird, cat, dog
    list_names = [name for name in os.listdir(directory + class_) if name.endswith('.wav')] # loads all file names from each folder into the list
    number_audios.append(len(list_names)) # saves the size of each folder, i.e., the number of files present in it
    
print('Class bird:', number_audios[0], '\nClass cat:', number_audios[1], '\nClass dog:', number_audios[2],)


# Showing a histogram of the duration of each file
classes = ["bird", "cat", "dog"]
recordings_duration = []
for class_ in classes:
    list_names = [name for name in os.listdir(directory + class_) if name.endswith('.wav')]
    for name in list_names: # for each file
        sample, sample_rate = librosa.load(directory + '/' + class_ + '/' + name, sr=16000)
        recordings_duration.append(float(len(sample)/sample_rate)) # calculates the duration of each file (result of the calculation 'number_of_samples_in_the_file'/'number_of_samples_per_second')
    
plt.hist(np.array(recordings_duration))


# Loading all files using the librosa library and fixing the sample_rate at 8,000 (to reduce the amount of data)
# We'll exclude samples that have a duration of less than 1 second
all_audios = []
all_labels = []
count = 0
for class_ in classes:
    print('Processing', class_)
    list_names = [name for name in os.listdir(directory + '/'+ class_) if name.endswith('.wav')]
    for name in list_names:
        count += 1
        sample, sample_rate = librosa.load(directory + '/' + class_ + '/' + name, sr=16000)
        sample = librosa.resample(sample, orig_sr=sample_rate, target_sr=8000)
        if(len(sample) == 8000): 
            all_audios.append(sample)
            all_labels.append(class_)
            
            
# Showing the total number of samples we ended up with and how many there were
print('We ended up with', len(all_audios), 'out of a total of', count)


# Note: with this data, we could create a classifier using dense neural networks, for example. But as we're going to use LSTM, 
# 8,000 timesteps is a very large number, so working with MFCC will be useful.

# Creating the MFCC spectrograms
mfccs = []
for audio in all_audios:
    mfcc = librosa.feature.mfcc(y=audio, sr=8000, n_mfcc=20)
    mfcc = np.transpose(mfcc) # reversing the position of features and timesteps, as LSTM expects to receive them in this order: (timesteps, features)
    mfccs.append(mfcc)
    
    
# mfccs is a list of two-dimensional arrays. We need to turn this into a numpy array of 3 dimensions:
mfccs


x = np.stack(mfccs, axis=0)

# Visualizing the result of this:
x

x.shape

# Now we will work with the classes. We already created a class for each sample, let's check:
len(all_labels)

all_labels

# Performing label encoding with these classes:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(all_labels)
classes = list(le.classes_)

# Applying one hot encoding:
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
#y = np_utils.to_categorical(y, num_classes=len(classes))
y = to_categorical(y, num_classes=len(classes))

y.shape

# Now that we have our data x and y as numpy arrays in the correct dimensions, we can split the data between training and testing:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2) # stratify maintains the class ratio

x_train.shape

x_test.shape

# Creating the LSTM
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
import warnings
warnings.filterwarnings('ignore')

# Creating the LSTM model
model = Sequential()
model.add(LSTM(150, dropout=0.3, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=500, validation_data=(x_test, y_test), verbose=2)
print('model trained')

# Making predictions:
from numpy import expand_dims
sample = expand_dims(x_test[0], 0) # creating an extra dimension to be (n_samples, timesteps, features), because when taking only x_test[0] the first dimension dies
prob = model.predict(sample) # getting the probabilities of each class
index = np.argmax(prob[0]) # getting the index of the column with the highest probability
classes[index] # showing the respective class

# Verifying the answer
index = np.argmax(y_test[0])
classes[index]

# Save the trained model
model.save('C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionVoz/audio_classification_model.h5')
print('Model saved successfully.')
