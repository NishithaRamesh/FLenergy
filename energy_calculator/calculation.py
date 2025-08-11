import psutil
from energy_calculator import get_carbon_intensity
from energy_calculator import get_cpu_details
from energy_calculator import get_gpu_details
from energy_calculator import file_handling
from energy_calculator import get_cpu_details
from energy_calculator import get_tdp_values
import csv
import time
import threading
import pandas as pd 
import os 


class EnergyMonitoring:
    def __init__(self, project_path, project_name, parameters, frequency, pid, mode='standalone'):
       
        # Energy monitoring variables
        self.project_name = project_name
        self.mode = mode
        self.parameters = parameters
        self.frequency = frequency
        self.pid = pid
        self.epoch = 0
        self.round = 0
        self.output_comm_path = ''

        # File path for saving the energy results
        self.powergadget_path = 'C:\\Program Files\\Intel\\Power Gadget 3.6\\PowerLog3.0.exe' # path where powergadget is located
        self.file_path = './Energy_Results/csv_files/' + project_path + '/'
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        self.create_empty_csv()

        # Thread variables
        self.energythread = None
        self.thread_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.last_line_read = 0
        print("TDP values are being read from the csv file")
        self.tdp_path = os.getcwd() + "/cpu_power.csv"
        self.tdp_values = get_tdp_values.read_tdp_values_from_csv(self.tdp_path)
        self.cpu_info = get_cpu_details.get_cpu_details()
        self.cpu_model = self.cpu_info.get("CPU Model", "")
        self.cpu_model = get_cpu_details.parse_cpu_model_name(self.cpu_model)
        self.cpu_cores = self.cpu_info.get("Physical Cores", 0)

    def create_empty_csv(self):
        output_energy_file = self.file_path + 'Energy_Tracking_' + self.project_name + '.csv'
        header = ["Project Name", "Learning Rate", "Batch Size",
                  "CPU Model", "CPU Cores", "CPU Utilization(%)", "CPU Power(KW)", "CPU Energy(KWH)",
                  "RAM Power(KW)", "RAM Energy(KWH)",
                  "GPU Model", "GPU Power(KW)", "GPU Energy(KWH)", 
                  "Total Energy(KWH)", "Total Power(KW)",
                  "Country", "Carbon Intensity(gCO2eq/kWh)", "Carbon Emission(gCO2eq)", "Epoch"]
        if self.mode == 'FL':
            header.append("Round")
        if not os.path.exists(output_energy_file):
            try:
                with open(output_energy_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                print(f"Created Energy CSV file: {output_energy_file}")
            except Exception as e:
                print(f"Error creating Energy CSV file: {e}")
        
    
    def power_consumption(self, time_elapsed, os_name):
        csv_data = []
        power_w_CPU = 0
        power_w_GPU = 0
        cpu_usage_percent = 0
        total_cpu_usage_percent = 0
        
        zone, intensity = get_carbon_intensity.get_carbon_intensity()
    
        power_w_CPU = get_cpu_details.calculate_cpu_power(os_name, self.cpu_model, self.tdp_values)
        total_cpu_usage_percent = get_cpu_details.get_curr_process_cpu_utilization(self.pid) 
        power_w_CPU = power_w_CPU * total_cpu_usage_percent / 100

        power_w_RAM = get_cpu_details.get_power_from_RAM()

        nvidia_gpu_info = get_gpu_details.get_gpu_power_nvidia()
        for gpu, power_draw in nvidia_gpu_info.items():
            gpu_model = gpu
            power_w_GPU += power_draw

        power_kw_CPU = power_w_CPU / 1000.0
        energy_kwh_CPU = (power_kw_CPU * time_elapsed) / 3600.0

        power_kw_RAM = power_w_RAM / 1000.0
        energy_kwh_RAM = (power_kw_RAM * time_elapsed) / 3600.0

        power_kw_GPU = power_w_GPU / 1000.0
        energy_kwh_GPU = (power_kw_GPU * time_elapsed) / 3600.0
        
        total_power = power_kw_CPU + power_kw_GPU + power_kw_RAM
        total_energy = energy_kwh_CPU + energy_kwh_GPU + energy_kwh_RAM

        carbon_emission = intensity * total_energy

        csv_data = [[self.project_name], [self.parameters[0]], [self.parameters[1]], 
                    [self.cpu_model], [self.cpu_cores], [total_cpu_usage_percent], [power_kw_CPU], [energy_kwh_CPU], 
                    [power_kw_RAM], [energy_kwh_RAM], 
                    [gpu_model], [power_kw_GPU], [energy_kwh_GPU], 
                    [total_energy], [total_power],
                    [zone], [intensity], [carbon_emission], [self.epoch]]
        
        if self.mode == 'FL':
            csv_data.append([self.round])
        
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        output_file = self.file_path + 'Energy_Tracking_' + self.project_name + '.csv'
        file_handling.write_to_csv(csv_data, output_file, self.thread_lock, self.mode)

    def calculate_averages(self, cpu_util, cpu_power, cpu_energy, gpu_power, gpu_energy, total_energy, total_power, ci, total_emissions):
        # # Calculate average for each column
        averages = []
        avg_cpu_util = float(sum(cpu_util)) / len(cpu_util)
        avg_cpu_power = float(sum(cpu_power)) #/ len(cpu_power)
        avg_cpu_energy = float(sum(cpu_energy)) #/ len(cpu_energy)
        avg_gpu_power = float(sum(gpu_power)) #/ len(gpu_power)
        avg_gpu_energy = float(sum(gpu_energy)) #/ len(gpu_energy)
        avg_total_energy = float(sum(total_energy)) #/ len(total_energy)
        avg_total_power = float(sum(total_power)/ len(total_power)) #/ len(total_power)
        avg_ci = float(sum(ci)) / len(ci)
        total_emissions = float(sum(total_emissions)) #/ len(total_emissions)
        averages = [avg_cpu_util, avg_cpu_power, avg_cpu_energy, avg_gpu_power, avg_gpu_energy, avg_total_power, avg_total_energy, avg_ci, total_emissions]
        return averages

    def calculate_total_energy(self, epoch, per_epoch=False):

        ot_string = 'Average_emmissions'
        input_file = self.file_path + 'Energy_Tracking_' + self.project_name + '.csv'
        output_file = self.file_path + ot_string + '_' + self.project_name +'.csv'
        data, last_line_read = file_handling.get_csv_data(input_file, self.thread_lock, self.last_line_read)
        self.last_line_read = last_line_read

        for i in range(self.num_epochs):
            if 'Epoch' == i:
                cpu_util, cpu_power, cpu_energy, gpu_power, gpu_energy, total_energy, total_power, ci, total_emissions = data[7:16]
                # calculate_averages(data)
                averages = self.calculate_averages(cpu_util, cpu_power, cpu_energy, gpu_power, gpu_energy, 
                total_energy, total_power, ci, total_emissions)

                csv_data = [[data[0]], [data[1]], [data[2]], [data[3]], [data[4]],
                            [averages[0]], [averages[1]],  [averages[2]], [data[5]], [averages[3]], [averages[4]], [averages[6]], [averages[5]],
                            [data[6]], [averages[7]], [averages[8]],[epoch]]
                    
        # averages.to_csv(output_file, index=False)
        
        file_handling.write_to_csv(csv_data, output_file,  self.thread_lock, True)
        return

    def energy_calculation(self):
        os_name = get_cpu_details.get_platform()
        while True:
            if os_name == "Windows":
                check_power_log = get_cpu_details.check_if_powerlog(self.powergadget_path)
                if check_power_log:
                    # Resolution in milliseconds
                    resolution = 1000
                    thread = threading.Thread(target=get_cpu_details.start_energy_power_gadget, args=(self.powergadget_path, self.frequency, resolution))
                    thread.daemon = True
                    thread.start()
                    # get_cpu_details.start_energy_power_gadget(self.powergadget_path, frequency, resolution)
            elif os_name == "Linux":
                    # Resolution in seconds
                    resolution = 5
                    thread = threading.Thread(target=get_cpu_details.powerstat_start, args=(self.frequency, resolution))
                    thread.daemon = True
                    thread.start()
                    pass 

            time_elapsed = self.frequency
            start_time = time.time()
            self.power_consumption(time_elapsed, os_name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_time = max(0, self.frequency - elapsed_time)
            # print(f"Sleeping for {self.frequency} seconds")
            # print(f"Time taken for measuring and writing power consumption: {elapsed_time} seconds")
            time.sleep(sleep_time)
            if self.stop_flag.is_set():
                time.sleep(30)
                return
    
    def update_epoch(self, epoch):
        self.epoch = epoch
    
    def update_round(self, round):
        self.round = round

    def final_calculation(self):#,input_file, output_file):
        ot_string = 'Epochwise_emissions'
        input_file = self.file_path + 'Energy_Tracking_' + self.project_name + '.csv'
        df = pd.read_csv(input_file)
        output_file = self.file_path + ot_string + '_' + self.project_name +'.csv'
        
        if self.mode == 'standalone':
            # Ensure "Epoch" exists
            if "Epoch" not in df.columns:
                raise ValueError("Error: 'Epoch' column not found in the dataset.")
        
            # Identify non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

            # Define custom aggregation rules
            sum_cols = {"CPU Power(KW)", "CPU Energy(KWH)", "RAM Power(KW)", "RAM Energy(KWH)","GPU Power(KW)", 
                        "GPU Energy(KWH)", "Total Power(KW)", "Total Energy(KWH)", 
                        "Carbon Emission(gCO2eq)" 
                        }  
            avg_cols = {"CPU Utilization(%)", "Carbon Intensity(gCO2eq/kWh)"}  
            first_cols = set(non_numeric_cols)  

            # Create aggregation rules
            agg_rules = {
                col: "first" if col in first_cols else
                    "mean" if col in avg_cols else
                    "sum" if col in sum_cols else "first"
                for col in df.columns if col != "Epoch"  
            }

            # Group by Epoch (handling decimal values properly)
            df_grouped = df.groupby("Epoch", as_index=False).agg(agg_rules)

            # Round numerical values for better readability
            df_grouped = df_grouped.round(8)
            df_grouped["Epoch"] = df_grouped["Epoch"].round(0)  # Rounds to the nearest integer
            df_grouped["Epoch"] = df_grouped["Epoch"].astype(int)  # Converts to integer
            df_grouped.to_csv(output_file, index=False)

        elif self.mode == 'FL':
            # Ensure "Epoch" and "Round" exist
            if "Epoch" not in df.columns or "Round" not in df.columns:
                raise ValueError("Error: 'Epoch' or 'Round' column not found in the dataset.")

            # Identify non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

            # Define custom aggregation rules
            sum_cols = {"CPU Power(KW)", "CPU Energy(KWH)", "RAM Power(KW)", "RAM Energy(KWH)", "GPU Power(KW)",
                        "GPU Energy(KWH)", "Total Power(KW)", "Total Energy(KWH)", "Carbon Emission(gCO2eq)"
                        }
            
            avg_cols = {"CPU Utilization(%)", "Carbon Intensity(gCO2eq/kWh)"}
            first_cols = set(non_numeric_cols)

            # Create aggregation rules
            agg_rules = {
                col: "first" if col in first_cols else
                    "mean" if col in avg_cols else
                    "sum" if col in sum_cols else "first"
                for col in df.columns if col not in ["Epoch", "Round"]
            }

            # Group by Epoch and Round (handling decimal values properly)
            df_grouped = df.groupby(["Round", "Epoch"], as_index=False).agg(agg_rules)

            # Round numerical values for better readability
            df_grouped = df_grouped.round(8)
            df_grouped["Epoch"] = df_grouped["Epoch"].round(0)  # Rounds to the nearest integer
            df_grouped["Epoch"] = df_grouped["Epoch"].astype(int)  # Converts to integer
            df_grouped["Round"] = df_grouped["Round"].astype(int)  # Converts to integer


            output_file = self.file_path + ot_string + '_' + self.project_name + '_All_Rounds.csv'
            file_exists = os.path.exists(output_file)
            df_grouped.to_csv(output_file, mode='a', header=not file_exists, index=False)

        df_epochwise = pd.read_csv(output_file)
        total_aggregates = df_epochwise.agg({
            "CPU Power(KW)": "sum",
            "CPU Energy(KWH)": "sum",
            "RAM Power(KW)": "sum",
            "RAM Energy(KWH)": "sum",
            "GPU Power(KW)": "sum",
            "GPU Energy(KWH)": "sum",
            "Total Power(KW)": "sum",
            "Total Energy(KWH)": "sum",
            "Carbon Emission(gCO2eq)": "sum",
            "CPU Utilization(%)": "mean",
            "Carbon Intensity(gCO2eq/kWh)": "mean"
        })
        file_exists = os.path.exists(output_file)
        total_aggregates.to_csv(output_file, mode='a', header=not file_exists, index=False)

    def start_calculation_epochwise(self, epoch, per_epoch):
        thread = threading.Thread(target=self.calculate_total_energy, args=(epoch, per_epoch))
        thread.start()

    def start_energy_calculation(self):
        self.stop_flag.clear()
        self.energythread = threading.Thread(target=self.energy_calculation)
        self.energythread.daemon = True
        self.energythread.start()
        return self.energythread

    def stop_energy_calculation(self):
        self.stop_flag.set()
        self.energythread.join()
        print("Energy Calculation Stopped")

    def calculate_communication_energy(self, client_name, direction):
        columns = ["Round", "Client", "Direction", "Model_size", "Energy (KWH)"]
        df = pd.DataFrame(columns=columns)
        df.loc[0] = [self.round, client_name, direction, None, None]
        
        output_file = self.file_path + 'Communication_Energy.csv'
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode='a', header=not file_exists, index=False)
        self.output_comm_path = output_file
        
        print(f"Communication event logged to: {output_file}")

    def aggregate_communication_energy(self, model_paths):
        file_path = self.file_path + 'Communication_Energy.csv'
        consolidated_file_path = self.file_path + 'Consolidated_Communication_Energy.csv'
        
        if not os.path.exists(file_path):
            print("No communication energy log found.")
            return
        
        df = pd.read_csv(file_path)
        energy_consumed_per_mb = 0.0000065  # kWh per MB transferred over the network
        
        # Map model paths to directions
        direction_mapping = {
            'server_to_client': model_paths[0],
            'client_to_server': model_paths[1]
        }
        
        # Fill in model size and energy
        for direction, model_path in direction_mapping.items():
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            df.loc[df['Direction'] == direction, 'Model_size'] = file_size_mb
            df.loc[df['Direction'] == direction, 'Energy (KWH)'] = energy_consumed_per_mb * file_size_mb
        
        df.to_csv(file_path, index=False)
        # Aggregate only total energy per round
        consolidated_df = df.groupby('Round', as_index=False)['Energy (KWH)'].sum()
        
        # Save to Excel
        consolidated_df.to_csv(consolidated_file_path, index=False)
        print(f"Total communication energy per round saved to: {consolidated_file_path}")