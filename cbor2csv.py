import cbor2
import sys
import csv
import os

'''
with open(sys.argv[1], 'rb') as f:
    # Load the cbor file
    obj = cbor2.load(f)

    # Get the filename without extension
    label = os.path.splitext(sys.argv[1])[0]
    filename = str(label) + ".csv"

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        # uncomment to add a fixed header
        # writer.writerow(['x','temp [degC]','press [kPa]','hum [%]','gas res [MOhm]'])
        
        # or use this to read the values from edge impulse exported cbor files
        header = []
        header.insert(0, "timestamp")
        for sensor in obj['payload']['sensors']:
            header.append(sensor['name'])
        writer.writerow(header)

        timestamp = 0
        for val in obj['payload']['values']:
            val.insert(0, timestamp)
            writer.writerow(val)
            timestamp += int(obj['payload']['interval_ms'])

'''

# Check if the folder path is provided as command line argument
if len(sys.argv) != 3:
    print("Usage: python script.py input_folder_path output_folder_path")
    sys.exit(1)

input_folder_path = sys.argv[1]
output_folder_path = sys.argv[2]

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over each file in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith(".cbor"):
        cbor_file_path = os.path.join(input_folder_path, filename)
        csv_file_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + ".csv")

        with open(cbor_file_path, 'rb') as f:
            # Load the cbor file
            obj = cbor2.load(f)

            with open(csv_file_path, 'w') as csv_file:
                writer = csv.writer(csv_file)

                header = ["timestamp"]
                for sensor in obj['payload']['sensors']:
                    header.append(sensor['name'])
                writer.writerow(header)

                timestamp = 0
                for val in obj['payload']['values']:
                    val.insert(0, timestamp)
                    writer.writerow(val)
                    timestamp += int(obj['payload']['interval_ms'])

print("Conversion complete. CSV files are saved in", output_folder_path)