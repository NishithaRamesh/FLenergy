import csv
import subprocess
import gpustat

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

def get_gpu_utilization():
    try:
        stats = gpustat.new_query()
        for gpu in stats.gpus:
            print(f"GPU {gpu} - {gpu.utilization} utilization")
            for proc in gpu.processes:
                print(f"  Process {proc['pid']} - {proc['command']} - GPU usage: {proc['gpu_memory_usage']} MB")
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    get_gpu_utilization()

if __name__ == '__main__':
    main()