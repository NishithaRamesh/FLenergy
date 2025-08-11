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
rapl_path = "/sys/class/powercap/intel-rapl"

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

def calculate_from_rapl():
    """
    Calculate CPU power consumption using Linux RAPL interface.
    """
    try:
        energy_uj_path = os.path.join(rapl_path, "intel-rapl:0", "energy_uj")
        if os.path.exists(energy_uj_path):
            with open(energy_uj_path, "r") as file:
                energy_uj = int(file.read().strip())
                power_watts = energy_uj / 1e6  # Convert microjoules to watts
                return power_watts
        else:
            print("RAPL interface not found.")
            return 0
    except Exception as e:
        print(f"Error reading RAPL data: {e}")
        return 0

def calculate_cpu_power(os_name, cpu_model, tdp_values):
    """
    Calculate CPU power consumption based on the operating system.
    """
    if os_name == "Windows":
        if os.path.exists(cpu_windows_path):
            return calculate_from_ipg(cpu_windows_path)
        return calculate_from_tdp(cpu_model, tdp_values)
    elif os_name == "Linux":
        if os.path.exists(rapl_path):
            return calculate_from_rapl()
        return calculate_from_tdp(cpu_model, tdp_values)
    return 0

def start_energy_power_gadget(powergadget_path, duration, resolution):
    """
    Start Intel Power Gadget to log power consumption.
    """
    command = f'"{powergadget_path}" -duration {duration} -resolution {resolution} -file "{cpu_windows_path}"'
    subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

def get_platform():
    """
    Get the operating system name.
    """
    os_name = platform.system()
    print("Operating System:", os_name)
    return os_name