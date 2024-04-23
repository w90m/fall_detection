
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam

# Step 1: Load and preprocess the data

# Define data folders
train_data_folder = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\fall3_rnn\training_csv"
test_data_folder = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\fall3_rnn\testing_csv"

# Define class labels and repetitions
class_repetitions = {
    "walking": 30,
    "idle": 30,
    "fall": 50
}

# Load training data
train_dataframes = []
y_train_labels = []
for filename in os.listdir(train_data_folder):
    if filename.endswith(".csv"):
        label = filename.split(".")[0]  # Extract label from filename
        if label in class_repetitions:
            df = pd.read_csv(os.path.join(train_data_folder, filename))
            train_dataframes.append(df)
            y_train_labels.extend([label] * len(df))  # Assign label to each sample


# Concatenate all training dataframes
train_data = pd.concat(train_dataframes, ignore_index=True)


# Separate features and labels
X_train = train_data.drop(columns=['gyrX','gyrY', 'gyrZ', 'magX', 'magY', 'magZ'])
y_train = pd.Series(y_train_labels)

# Load testing data
test_dataframes = []
y_test_labels = []
for filename in os.listdir(test_data_folder):
    if filename.endswith(".csv"):
        label = filename.split(".")[0]  # Extract label from filename
        if label in class_repetitions:
            df = pd.read_csv(os.path.join(test_data_folder, filename))
            test_dataframes.append(df)
            y_test_labels.extend([label] * len(df))  # Assign label to each sample


# Concatenate all testing dataframes
test_data = pd.concat(test_dataframes, ignore_index=True)

# Separate features and labels
X_test = test_data.drop(columns=['gyrX','gyrY', 'gyrZ', 'magX', 'magY', 'magZ'])
y_test = pd.Series(y_test_labels)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Reshape data for RNN input
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Step 2: Define the model architecture
model = Sequential([
    SimpleRNN(128, input_shape=(X_train.shape[1], X_train.shape[2])),
    #SimpleRNN(64, return_sequences=True),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(class_repetitions), activation='softmax')
])



print("Number of samples in X_train:", X_train.shape[0])
print("Number of samples in y_train:", len(y_train))
print("Number of samples in X_test:", X_test.shape[0])
print("Number of samples in y_test:", len(y_test))


# Step 3: Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

model.save('rnn_model.h5')
# Step 6: Save the model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\rnn_model.h5")
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False  # Disable experimental lowering of tensor list ops

tflite_model = converter.convert()

with open('rnn_model.tflite', 'wb') as f:
    f.write(tflite_model)



