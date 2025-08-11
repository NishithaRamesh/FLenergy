import requests
import platform
import cpuinfo
import GPUtil
import psutil
import csv
import time
import os
import subprocess
from datetime import datetime
import multiprocessing

url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest?lat=49.43&lon=7.77&zone=DE"
headers = {
    "auth-token": "beupWDmvL1DJkKbmjcbnQEGDEqWILClf"
    }

def busy_function():
    while True:
        # Perform some CPU-intensive computations
        result = 0
        for _ in range(10**6):
            result += 1

def write_to_csv(data):
    
    header = ["Time stamp", "Project Name", "CPU Cores", "CPU Model", "CPU power", "CPU Energy",
    "GPU Model", "GPU Power", "GPU Energy", "Total Energy", "Total Power",
    "Country", "Carbon Intensity", "Carbon Emmision"]

    file_path = '../Final_result_data/Data_Emmision_csv.csv'
    file_exists = os.path.exists(file_path)

    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert the string timestamp to a list
    # timestamp_list = [timestamp]    
    # print(type(timestamp_list))
    # data = [timestamp_list + row for row in data]

    transposed_data = list(map(list, zip(*data)))
    
    with open(file_path, 'a+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(header)
        csv_writer.writerows(transposed_data)

def carbon_intensity():
    country = 'DE'
    carbon_intensity = 138
    max_retries = 2
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            country = data.get("zone")
            carbon_intensity = data.get("carbonIntensity")
            return country, carbon_intensity
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except Exception as e:
            print(f"Error getting carbon intensity: {e}")
        retries += 1
        delay = 2 ** retries
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)
    print(f"Unable to access geographical location. Using 'Germany' as the default value - url={url}")
    return country, carbon_intensity

def get_system_info(cpu_info = True):

    if cpu_info:

        cpu_info = {
            "CPU Model": cpuinfo.get_cpu_info()["brand_raw"],
            "CPU Architecture": platform.architecture(),
            "Physical Cores": psutil.cpu_count(logical=False),
            "Logical Cores": psutil.cpu_count(logical=True),
        }
        return cpu_info
    else:

        try:
            gpus = GPUtil.getGPUs()
            gpu_info = {
                f"GPU {i + 1} Model": gpu.name for i, gpu in enumerate(gpus)
            }
        except Exception as e:
            gpu_info = {"GPU Information": f"Error retrieving GPU information: {e}"}
        return gpu_info

def get_gpu_power_nvidia():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_info_lines = result.stdout.strip().split('\n')
            gpu_info_reader = csv.reader(gpu_info_lines)
            gpu_power_values = {line[1]: float(line[2]) for line in gpu_info_reader}
            
            return gpu_power_values
        else:
            print(f"Error running nvidia-smi: {result.stderr}")
    except Exception as e:
        print(f"Error getting GPU information: {e}")

def read_tdp_values_from_csv(file_path):
    tdp_values = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cpu_model = row["Name"]
            tdp = int(row["TDP"])
            tdp_values[cpu_model] = tdp
    return tdp_values

def power_consumption(time_elapsed, project_name):

    csv_data = []
    power_w_CPU = 0
    power_w_GPU = 0
    cpu_usage_percent = 0
    total_cpu_usage_percent = 0
    
    zone, intensity = carbon_intensity()
    # print(zone)
    # print(intensity)  

    cpu_info = get_system_info(cpu_info=True)
    print("\nCPU Information:")
    
    cpu_model = cpu_info.get("CPU Model", "")
    print(f"CPU Model: {cpu_model}")

    cpu_cores = cpu_info.get("Physical Cores", 0)
    print(f"CPU Cores: {cpu_cores}")

    gpu_info = get_system_info(cpu_info=False)
    print("\nGPU Information:")
    print(gpu_info)

    tdp_values = read_tdp_values_from_csv("../data/cpu_power.csv")
    

    if cpu_model in tdp_values:
        tdp = tdp_values[cpu_model]
        cpu_usage_percent = psutil.cpu_percent(interval=1)
        power_w_CPU = tdp * cpu_cores
        power_w_CPU = power_w_CPU * (cpu_usage_percent / 100.0)
    else:
        print(f"\nWarning: TDP not available for {cpu_model}. Power estimation may not be accurate.\n")
        tdp = 85
        cpu_usage_percent = psutil.cpu_percent(interval=0.1)
        total_cpu_usage_percent =+ cpu_usage_percent
        power_w_CPU = tdp * cpu_cores
        power_w_CPU = power_w_CPU * (cpu_usage_percent / 100.0)
   
    # gpu_model = list(cpu_info.get("GPU Information", {}).values())[0]
    # gpu_power_values = read_power_values_from_csv("gpu_power.csv", "Power")

    # if gpu_model in gpu_power_values:
    #     gpu_power = gpu_power_values[gpu_model]
    #     power += gpu_power
    # else:
    nvidia_gpu_info = get_gpu_power_nvidia()
    for gpu, power_draw in nvidia_gpu_info.items():
        gpu_model = gpu
        power_w_GPU += power_draw

    
    power_kw_CPU = power_w_CPU / 1000.0
    energy_kwh_CPU = power_kw_CPU * time_elapsed

    power_kw_GPU = power_w_GPU / 1000.0
    energy_kwh_GPU = power_kw_GPU * time_elapsed
    
    total_power = power_kw_CPU + power_kw_GPU
    total_energy = energy_kwh_CPU + energy_kwh_GPU

    carbon_emission = intensity * total_energy

    csv_data = [[project_name], [cpu_model], [cpu_cores], [total_cpu_usage_percent], [power_kw_CPU], [energy_kwh_CPU], 
                [gpu_model], [power_kw_GPU], [energy_kwh_GPU], [total_energy], [total_power],
                [zone], [intensity], [carbon_emission]]
    
    write_to_csv(csv_data)
    
    return total_power, total_energy


if __name__ == "__main__":  
    
    project_name = input("Enter project name: ")
    
    total_power, total_energy = power_consumption(project_name)
     
    print("Total energy (kwh): ")
    print(total_energy)
    print("Total Power(Watts): ")
    print(total_power)


    