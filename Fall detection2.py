import tensorflow as tf
from keras import layers, models
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical

# Function to create the RNN model
def create_rnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.SimpleRNN(64, input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Function to preprocess the data
def preprocess_data(data):
    # Drop timestamp and gyroscope columns
    data = data.drop(columns=['timestamp', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ'])
    return data

# Load data from a folder
def load_data(folder_path):
    X = []
    y = []
    label_encoder = LabelEncoder()
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Load the CSV file
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            
            # Preprocess the data
            data = preprocess_data(data)
            
            label = filename.split('.')[0] # Extract label from filename
            
            # Append to X and y
            X.append(data[['accX', 'accY', 'accZ']].values)
            y.append(label)  
    
    y = label_encoder.fit_transform(y)
    y = to_categorical(y, num_classes=6)
            
    return np.array(X), np.array(y)

# Set paths to training and testing data folders
train_data_folder = r'C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\wm90-project-1-export\training_csv'
test_data_folder = r'C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\wm90-project-1-export\testing_csv'

# Load training and testing data
X_train, y_train = load_data(train_data_folder)
X_test, y_test = load_data(test_data_folder)

# Determine input shape and number of classes
input_shape = X_train.shape[1:]  # Shape of one sample
num_classes = len(np.unique(np.concatenate((y_train, y_test))))  # Number of unique classes

# Create the RNN model
model = create_rnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))