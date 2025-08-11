import psutil
import cpuinfo
import platform
import subprocess
import os 
import shutil
# import get_tdp_values
import pandas as pd
import re
# path to store powervalues into a csv from Intel Power Gadget
cpu_windows_path = os.getcwd() + "\\cpu_values.csv"
cpu_linux_path = os.getcwd() + "\\cpu_output.txt"

def get_power_from_RAM():
    # Get the power consumed by the system
    POWER_PER_GB = 3  # Estimated power in watts per 8GB

    # Get RAM usage in GB
    ram_info = psutil.virtual_memory()
    used_gb = ram_info.used / (1024 ** 3)

    # Estimate power usage
    power_used = (used_gb / 8) * POWER_PER_GB
    return power_used

def get_cpu_details():
    cpu_info = {
        "CPU Model": cpuinfo.get_cpu_info()["brand_raw"],
        "CPU Architecture": platform.architecture(),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Logical Cores": psutil.cpu_count(logical=True),
    }
    # print(cpu_info)
    return cpu_info

def get_curr_process_cpu_utilization(process_id):
    # process = psutil.Process(process_id)
    # cpu_percent = process.cpu_percent(interval=1)  # specify the interval in seconds
    cpu_percent = psutil.cpu_percent(interval=1)
    total_cores = psutil.cpu_count(logical=True)
    # cpu_percent = cpu_percent / total_cores

    return cpu_percent

def parse_cpu_model_name(model_name):
    if "intel" in model_name.lower():
        cleaned = re.sub(r'\(R\)', '', model_name)
        cleaned = re.sub(r'CPU', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        # Remove everything after '@' (e.g., the clock speed)
        cleaned = cleaned.split('@')[0].strip()
        return cleaned
    else:
        cleaned = re.sub(r'(\d+)(.*)', r'\1', model_name) 
        return cleaned.strip()
    
def calculate_from_tdp(cpu_model, tdp_values):

    if cpu_model in tdp_values.keys():
        # print('CPU found in TDP sheet')
        tdp = tdp_values[cpu_model]
        # cpu_usage_percent = psutil.cpu_percent(interval=1)
        power_w_CPU = tdp 

    else: 
        power_w_CPU = 85

    # power_w_CPU = 0.5 * power_w_CPU # 50% of TDP value is used if power metric isnot available
    return power_w_CPU

# Checks if intel powergadget is installed
def check_if_powerlog(powergadget_path):
    return shutil.which(powergadget_path) is not None
    # return os.path.exists(powergadget_path)

def calculate_from_ipg(csv_file_path):
    df = pd.read_csv(csv_file_path)
    search_string = "Average IA Power_0"
    row  = df[df['System Time'].str.contains(search_string, na=False)]
    power_str = str(row['System Time']).split('=')[1].split('\n')[0]
    power = float(power_str)
    
    return power

def calculate_from_pwst(csv_file_path):
    #write the code to read the powerstatt file
    # with open(filename, "r") as f:
    #     lines = f.readlines()

    # # Find the "Average" line
    # for line in lines:
    #     if "Average" in line:
    #         parts = line.split()
    #         power_avg = float(parts[1])  # Extract the average power (Watts)
    #         break

    # # Calculate energy in Joules
    # energy = power_avg * duration
    # print("energy: ")
    # print(energy)
    # return energy
    return 45

def calculate_cpu_power(os_name, cpu_model, tdp_values):
    if os_name == "Windows":
        if os.path.exists(cpu_windows_path):
            power = calculate_from_ipg(cpu_windows_path)
        else:
            power = calculate_from_tdp(cpu_model, tdp_values)
    elif os_name == "Linux":
        # if os.path.exists(cpu_linux_path):
        #     power = calculate_from_pwst(cpu_linux_path)
        # else:
        power = calculate_from_tdp(cpu_model, tdp_values)
        return power
    
    return power

# Start running intel power_gadget
def start_energy_power_gadget(powergadget_path,duration, resolution):
    
    # duration in seconds 
    duration = str(duration)

    # resolution in milliseconds
    resolution = str(resolution)

    # Command to run powergadget
    command = f'"{powergadget_path}" -duration {duration} -resolution {resolution} -file "{cpu_windows_path}"'

    # Execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Get the output and errors
    output, errors = process.communicate()

    return 

# def powerstatt_start(duration, resolution):
    # resolution = str(resolution)
    # output_file = "./cpu_output.txt"
    # command = ["powerstat", "-d", str(resolution), "-c", "-z"]

    # with open(output_file, "w") as f:
    #     # Start powerstat
    #     process = subprocess.Popen(command, stdout=f)

    #     # Let it run for the specified duration
    #     # time.sleep(duration)

    #     # # Stop powerstat
    #     # process.terminate()


# def powerstat_start(duration, resolution):
#     resolution = str(resolution)
#     output_file = "./cpu_output.txt"
#     # command = ["powerstat", "-d", resolution, "-c", "-z", "-r"]
#     command = ["sudo", "powerstat", "-R"]
#     with open(output_file, "w") as f:
#         # Start powerstat
#         process = subprocess.Popen(command, stdout=f)

#         # Let it run for the specified duration
#         time.sleep(duration)

#         # Stop powerstat
#         process.terminate()
        
def get_platform():
    os_name = platform.system()
    print("Operating System:", os_name)
    return os_name