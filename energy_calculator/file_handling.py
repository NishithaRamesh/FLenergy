import os
import csv
import shutil
import string
import random
import time
from datetime import datetime
def write_to_csv(data, file_path, lock=None, mode="standalone"):

    file_exists = os.path.exists(file_path)
    if file_exists:
            mode = 'a+'
    else:
            mode = 'w'
    
    header = ["Project Name", "Learning Rate", "Batch Size",
              "CPU Model", "CPU Cores", "CPU Utilization(%)", "CPU power(KW)", "CPU Energy(KWH)",
              "RAM Power(KW)", "RAM Energy(KWH)",
              "GPU Model", "GPU Power(KW)", "GPU Energy(KWH)", 
              "Total Energy(KWH)", "Total Power(KW)",
              "Country", "Carbon Intensity(gCO2eq/kWh)", "Carbon Emission(gCO2eq)", "Epoch"]
    if mode == 'FL':
        header.append("Round")
    
    transposed_data = list(map(list, zip(*data)))
    # with lock:        
    with open(file_path, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(header)
        csv_writer.writerows(transposed_data)

def copy_file(source_path, destination_path):
    try:
        shutil.copyfile(source_path, destination_path)
        # print(f"File copied successfully from {source_path} to {destination_path}")
    except FileNotFoundError:
        print("Copy Error: One of the specified paths does not exist.")
    except PermissionError:
        print("Copy Error: Permission denied.")
    except Exception as e:
        print(f"An error occurred: {e}")

def del_file(file_path):
    try:
        os.remove(file_path)
        # print(f"File '{file_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Deletion Error: File '{file_path}' not found.")
    except PermissionError:
        print(f"Deletion Error: Permission denied to delete file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_random_string(length=5):
    
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    
    return random_string

def get_csv_data(file_path, lock, last_line_read):
    data = []
    text_values = []
    cpu_util = []
    cpu_power = []
    cpu_energy = []
    gpu_power = []
    gpu_energy = []
    total_emissions = []
    total_power = []
    total_energy = []
    ci = []

    
    file_exists = os.path.exists(file_path) 
    attempts = 0
    while not file_exists and attempts < 5:
        print(f"Averaging Error: File '{file_path}' not found. Waiting for a while.")
        time.sleep(30)
        file_exists = os.path.exists(file_path)
        attempts += 1
        # print(f"Error: File '{file_path}' not found.")
    if attempts == 5:
        print(f"Averaging Error: File '{file_path}' not found.")
        return data
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('./temp'):
        os.makedirs('./temp')
    temp_file = "./temp/temp" + generate_random_string() + "_" + timestamp +".csv"
    copy_file(file_path, temp_file)

    new_last_line_read = last_line_read
    with lock:
        with open(temp_file, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader, None)  # Skip the header if present
            current_line = 0

            # Skip lines up to last_line_read
            for _ in range(last_line_read):
                next(csv_reader, None)
                current_line += 1

            # Process rows
            for row in csv_reader:
                current_line += 1

                # Capture text values only once from the first row
                if current_line == last_line_read + 1:
                    text_values = [row[0], row[1], row[2], row[3], row[4], row[8], row[13]]
                
                cpu_util.append(float(row[5]))
                cpu_power.append(float(row[6]))
                cpu_energy.append(float(row[7]))
                gpu_power.append(float(row[9]))
                gpu_energy.append(float(row[10]))
                total_energy.append(float(row[11]))
                total_power.append(float(row[12]))
                ci.append(float(row[14]))
                total_emissions.append(float(row[15]))

            new_last_line_read = current_line

    # Combine text values with the numeric lists
    data = text_values + [cpu_util, cpu_power, cpu_energy, gpu_power, gpu_energy, total_energy, total_power, ci, total_emissions]               
    # print(data)
    # del_file(temp_file)  # Uncomment if necessary
    return data, new_last_line_read