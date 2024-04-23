#Fall detection

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam

#---------------------------------------------------
# Load and preprocess data
#---------------------------------------------------
#train_data = pd.read_csv("training_csv.csv")
#test_data = pd.read_csv("testing_csv.csv")


train_data_folder = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\wm90-project-1-export\training_csv"
test_data_folder = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\wm90-project-1-export\testing_csv"

#list to store the dataframe
train_dataframes = []
test_dataframes = []
labels = []

#number of label repetitions for each class
class_repetitions = {
    "walking": 3751,
    "right-side_fall": 626,
    "left-side_fall": 626,
    "idle": 626,
    "forward_fall": 626,
    "backwards_fall": 626}

#counter for each label
label_counts = {label:0 for label in class_repetitions.keys()}


#----------------------------------------------
# Training Dataframe
#----------------------------------------------
#iterate over each file in the folder
for filename in os.listdir(train_data_folder):
    if filename.endswith(".csv"): #assuming all files are csv files
        file_path = os.path.join(train_data_folder, filename)


        #extract label from filename
        train_label = None
        for class_name in class_repetitions:
            if class_name in filename:
                train_label = class_name
                break
        
        if train_label is None:
            print(f"Warning, unable to determine label for {filename}")
            continue

        try:

            #load data from the filename into a Dataframe
            df = pd.read_csv(file_path)

            #append the lable column to the Dataframe
            df["label"] = train_label

            #update label count
            label_counts[train_label]+=len(df)

            #append the dataframe to the list
            train_dataframes.append(df)


            #debuging - checking length of dataframe and associated label
            print(f"filnemane: {filename}, Dataframe length: {len(df)}, Label repetitions: {len(df)}")

        except ValueError as ve:
            print(f"Value error processing file {filename}: {ve}")

        except pd.errors.ParserError as pe:
            print(f"Parser error processing {filename}: {pe}")
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")



#----------------------------------------------
# Testing Dataframe
#----------------------------------------------
#iterate over each file in the folder
for filename in os.listdir(test_data_folder):
    if filename.endswith(".csv"): #assuming all files are csv files
        file_path = os.path.join(test_data_folder, filename)


        #extract label from filename
        test_label = None
        for class_name in class_repetitions:
            if class_name in filename:
                test_label = class_name
                break
        
        if test_label is None:
            print(f"Warning, unable to determine label for {filename}")
            continue

        try:

            #load data from the filename into a Dataframe
            df = pd.read_csv(file_path)

            #append the lable column to the Dataframe
            df["label"] = test_label

            #update label count
            label_counts[test_label]+=len(df)

            #append the dataframe to the list
            test_dataframes.append(df)


            #debuging - checking length of dataframe and associated label
            print(f"filnemane: {filename}, Dataframe length: {len(df)}, Label repetitions: {len(df)}")

        except ValueError as ve:
            print(f"Value error processing file {filename}: {ve}")

        except pd.errors.ParserError as pe:
            print(f"Parser error processing {filename}: {pe}")
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")





#--------------------------------------------------
# Concatenate train and test dataframes
#--------------------------------------------------
#concatenate all Dataframes into a single Dataframe
train_data = pd.concat(train_dataframes, ignore_index = True) 
test_data = pd.concat(test_dataframes, ignore_index = True)


print("length of train labels:", len(train_label))
print("length of train_data index:", len(train_data.index))

print("length of test labels:", len(test_label))
print("length of test_data index:", len(test_data.index))


#print summary pf label distribution
for label, count in label_counts.items():
    print(f"{label}: {count}samples")

print("Complete.")

columns_to_drop = ["timestamp", "gyrX", "gyrY", "gyrZ", "magX","magY", "magZ"]
train_data.drop(columns = columns_to_drop, inplace = True)

print("Train data--------------------------------------")
print(train_data.head(187800))


test_data.drop(columns = columns_to_drop, inplace = True)

print("Test data--------------------------------------")
print(test_data.head(187800))


#-------------------------------------------
#More data preprocessing
#-------------------------------------------
X_train = train_data.drop(columns = ['label']) # keep only the features for training
y_train = train_data['label'] #training labels

X_test = test_data.drop(columns = ['label'])
y_test = test_data['label']

#Encoding labels to numerical form

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

#--------------------------------------------
#Build, compile and train the neural network
#--------------------------------------------


model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # 6 output classes
])


X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))

'''
# Define the RNN model
model = Sequential([
    LSTM(512, activation='relu', return_sequences=True, input_shape=(1,3)),
    Dropout(0.5),
    LSTM(256, activation='relu', return_sequences=True),
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # 6 output classes
])
'''




optimizer = Adam(learning_rate=0.001)

model.compile(optimizer= optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)

