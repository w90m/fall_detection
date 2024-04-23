#Fall detection rough

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from sklearn.model_selection import train_test_split

#---------------------------------------------------
# Load and preprocess data
#---------------------------------------------------
#train_data = pd.read_csv("training_csv.csv")
#test_data = pd.read_csv("testing_csv.csv")


train_data_folder = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\wm90-project-1-export\training_csv"
#train_data_folder = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\wm90-project-1-export\testing_csv"

#list to store the dataframe
dataframes = []
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

#iterate over each file in the folder
for filename in os.listdir(train_data_folder):
    if filename.endswith(".csv"): #assuming all files are csv files
        file_path = os.path.join(train_data_folder, filename)


        #extract label from filename
        label = None
        for class_name in class_repetitions:
            if class_name in filename:
                label = class_name
                break
        
        if label is None:
            print(f"Warning, unable to determine label for {filename}")
            continue

        try:

            #load data from the filename into a Dataframe
            df = pd.read_csv(file_path)

            #append the lable column to the Dataframe
            df["label"] = label

            #update label count
            label_counts[label]+=len(df)

            #append the dataframe to the list
            dataframes.append(df)


            '''
            #extract label from the filename
            #label = filename.split('.')[0] 

            #append the Dataframe to the list
            dataframes.append(df)
            labels.append(label)

            #determine number of repetitions for the label based on the class
            repetitions = class_repetitions[label]

            #repeat the label for number of rows in the dataframe
            labels.extend([label] * len(df))
            '''

            #debuging - checking length of dataframe and associated label
            print(f"filnemane: {filename}, Dataframe length: {len(df)}, Label repetitions: {len(df)}")

        except ValueError as ve:
            print(f"Value error processing file {filename}: {ve}")

        except pd.errors.ParserError as pe:
            print(f"Parser error processing {filename}: {pe}")
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

#concatenate all Dataframes into a single Dataframe
train_data = pd.concat(dataframes, ignore_index = True)



print("length of lables:", len(labels))
print("length of train_data index:", len(train_data.index))

#if len(labels) !=len(train_data.index):
#    print("Error: lenghts of labels and train_data do not match")
#    print(f"filename {filename}")

#train_data['label'] = labels

#print summary pf label distribution
for label, count in label_counts.items():
    print(f"{label}: {count}samples")

print("Complete.")

columns_to_drop = ["timestamp", "gyrX", "gyrY", "gyrZ", "magX","magY", "magZ"]
train_data.drop(columns = columns_to_drop, inplace = True)

print(train_data.head(187800))

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
    SimpleRNN(20, input_shape=(X_train.shape[1],)),
    SimpleRNN(10),
    Dense(3, activation='softmax')  # 3 output classes
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer= optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)


