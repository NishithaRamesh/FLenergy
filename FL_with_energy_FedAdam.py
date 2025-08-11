from collections import OrderedDict
from typing import List, Tuple

import pandas as pd
import os
import torch
from ultralytics import YOLO
from energy_calculator import calculation
from datetime import datetime
import numpy as np
import re
from energy_calculator.calculation import EnergyMonitoring
import time
import torch.distributed as dist
# import yolov8
import shutil
import yaml

import flwr
from flwr.client import ClientApp, NumPyClient
from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation
import threading
# from flwr_datasets import FederatedDataset
# from torch.utils.data import DataLoader
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))  # Bind to a free port chosen by the OS
        _, port = s.getsockname()
        return port

port_num = find_free_port()
print(f"Selected free port: {port_num}")
os.environ["MASTER_ADDR"] = "localhost"  # Change if running on a multi-node cluster
os.environ["MASTER_PORT"] = str(port_num)  # Change this to an unused port
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used is {device}")


# Define the hyperparameters
learning_rate = 0.001
lrf = 0.001
batch_size = 32
parameters = [learning_rate, batch_size]

num_rounds = 10
val_per_epoch = True
model_parameters = None

epoch_frequency = 1
energy_calc_frequency = 2 

log_path = './final_result_data_FL/logs/'
filename = ''
filepath = ''
dt_string = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
pid = os.getpid()

project_name = 'FOD_adam'
project_path =  f'FL/{project_name}'
server_project_name = f"{project_name}_server"
server_energy = EnergyMonitoring(
    project_path, 
    server_project_name, 
    parameters, 
    energy_calc_frequency, 
    pid
    )

is_training_flag = False
flag = True

def del_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # Deletes the entire directory and its contents
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Directory does not exist: {dir_path}")

def save_round_number(round_num, file_path="./temp/round_number.txt"):
    dir_path = os.path.dirname(file_path)

    # Create directory if it does not exist (but only if there's a directory to create)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, "w") as f:
        f.write(str(round_num))

def load_round_number(file_path="./temp/round_number.txt"):
    try:
        with open(file_path, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def write_flag_to_file(flag_value, file_path="./temp/training_flag.txt"):
    dir_path = os.path.dirname(file_path)

    # Create directory if it does not exist (but only if there's a directory to create)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, "w") as f:
        f.write(str(flag_value))  # Convert Boolean to string

def read_flag_from_file(file_path="./temp/training_flag.txt"):
    try:
        with open(file_path, "r") as f:
            return f.read().strip() == "True"  # Convert back to Boolean
    except FileNotFoundError:
        return False  # Default to False if file doesn't exist
 
def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict()

    for k, v in params_dict:
        try:
            # Ensure proper dtype, clone, and detach
            state_dict[k] = torch.tensor(v, dtype=torch.float32).clone().detach() #changed to detach function
        except Exception as e:
            print(f"Error setting parameter {k}: {e}, Expected shape: {model.state_dict()[k].shape}, Given shape: {v.shape}")
            raise
    try:
        model.load_state_dict(state_dict, strict=True)  # Use strict=False for debugging
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        raise

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def parse_yaml_and_count_images(yaml_path):
    # Load YAML config from file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    train_path = config.get('train')
    val_path = config.get('val')

    def count_images(path):
        if not os.path.isdir(path):
            return f"Path not found: {path}"
        return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    train_count = count_images(train_path)
    val_count = count_images(val_path)

    return train_count, val_count

class FlowerClient(NumPyClient):
    def __init__(self, model, yaml_file, client_project_name, client_name, num_rounds):
        self.model = model
        self.model.to(device)
        self.yaml_file = yaml_file
        self.client_project_name = client_project_name
        self.client_energy = EnergyMonitoring(project_path, self.client_project_name, parameters, energy_calc_frequency, pid, mode='FL')
        self.client_name = client_name
        self.num_rounds = num_rounds
        self.logpath = f'./results/FL/{project_name}/'
        self.model_path = f'./models_weights/{project_path}'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.port = find_free_port()
        print(f"Selected free port for {self.client_name}: {port_num}")
        os.environ["MASTER_PORT"] = str(self.port)
        self.mode = None
        self.rounds_file = f"./temp/{project_name}/rounds.txt"
        self.rounds = 0
        self.flag_file = f"./temp/{project_name}/training_flag.txt"
        self.num_train_images, self.num_val_images = parse_yaml_and_count_images(self.yaml_file)

    def get_parameters(self, config):
        self.get_save_path = os.path.join(self.model_path, f'{self.client_project_name}_{self.rounds}_get_weights.pth')
        if self.rounds == self.num_rounds:
            if self.client_name == "client0":
                torch.save(self.model.state_dict(),self.get_save_path)
        self.client_energy.calculate_communication_energy(self.client_name, "client_to_server")
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        # Update model parameters from the server
        set_parameters(self.model, parameters)
        if self.mode == "fit":
            self.set_save_path = os.path.join(self.model_path, f'{self.client_project_name}_{self.rounds}_set_weights.pth')
            if self.rounds == self.num_rounds:
                if self.client_name == "client0":
                    torch.save(self.model.state_dict(),self.set_save_path)
            self.client_energy.calculate_communication_energy(self.client_name, "server_to_client")

    def fit(self, parameters, config):
        self.mode = "fit"
        self.rounds = config["server_round"]
        # save_round_number(self.rounds, self.rounds_file)
        self.client_energy.update_round(self.rounds)
        print(f"Round : {self.rounds}")

        self.set_parameters(parameters)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        
        train_results_folder = f"{self.client_name}_round{self.rounds}_training_results"
        self.model.add_callback("on_train_epoch_start", lambda trainer: on_train_epoch_start(trainer, self))
        self.model.add_callback("on_train_end", lambda trainer: on_train_end(trainer, self))
        self.model.add_callback("on_train_start", lambda trainer: on_train_start(trainer, self))

        is_training_flag = True
        write_flag_to_file(is_training_flag, self.flag_file)
        save_round_number(self.rounds, self.rounds_file)

        self.model.train(
            data=self.yaml_file, epochs=10, imgsz=640, freeze=10,
            workers=4, batch=batch_size, device=device, lr0=learning_rate,
            lrf=lrf, optimizer='SGD', save = False,
            project=self.logpath, name = train_results_folder, val = val_per_epoch
        )

        is_training_flag = False
        write_flag_to_file(is_training_flag, self.flag_file)

        params = self.get_parameters(config)
        
        if self.rounds == self.num_rounds:
            self.client_energy.final_calculation()
            if self.client_name == "client0":               
                self.client_energy.aggregate_communication_energy([self.set_save_path, self.get_save_path])
        return params, self.num_train_images, {}

    def evaluate(self, parameters, config):
        self.mode = "evaluate"

        is_training_flag = True
        write_flag_to_file(is_training_flag, self.flag_file)

        self.set_parameters(parameters)
        # Perform evaluation using YOLOv8 validation
        val_results_folder = f"{self.client_name}_round{self.rounds}_evaluation_results"
        results = self.model.val(data=self.yaml_file, batch=batch_size, device=device, 
                                 project=self.logpath, name = val_results_folder)

        map50 = results.box.map50  # Example metric, adjust as needed

        is_training_flag = False
        write_flag_to_file(is_training_flag, self.flag_file)

        print(map50)
        return 0.0, self.num_val_images, {"map50": float(map50)}

def on_train_epoch_start(trainer, client):
    # time.sleep(5)
    current_epoch = trainer.epoch
    if current_epoch % epoch_frequency == 0:
        # epoch_calc = client.client_energy.start_calculation_epochwise(current_epoch, True)
        client.client_energy.update_epoch(current_epoch+1)
        # print(current_epoch)
        
def on_train_end(trainer, client):
    # client.client_energy.calculate_total_energy(num_epochs)
    client.client_energy.stop_energy_calculation()
    
def on_train_start(trainer, client):
    # global energy_thread
    client.client_energy.start_energy_calculation()

    # print('start')

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["map50"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"map50": sum(accuracies) / sum(examples)}

def client_fn(context: Context):
    global num_rounds
    partition_id = context.node_config["partition-id"]
    client_yaml_path = f"./cient_data.yaml" 
    yolo_yaml_path = f"./yolov8n_cus.yaml"

    # model = YOLO("yolo_base.pt").to(device)
    model = YOLO(yolo_yaml_path, task='detect')
    model.load("yolo_base.pt")
    client_project_name = f"{project_name}_client{partition_id}"
    # model.to(device)

    return FlowerClient(
        model=model,
        yaml_file=client_yaml_path,
        client_project_name=client_project_name,
        client_name=f"client{partition_id}",
        num_rounds = num_rounds
    ).to_client()

def fit_config(server_round: int):
    config = {
        "server_round": server_round,
    }
    return config

def server_fn(context: Context) -> ServerAppComponents:
    global model_parameters 
    strategy = FedAdam(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        initial_parameters=ndarrays_to_parameters(model_parameters),
        eta = 0.001, 
        eta_l = 0.1,
        beta_1 = 0.9,
        beta_2 = 0.99,    
        tau = 1e-09
    )
    config = ServerConfig(num_rounds=num_rounds)
    server_app_components = ServerAppComponents(strategy=strategy, config=config)
    return server_app_components

def server_update():
    prev = None  # Track the previous state to prevent redundant function calls
    rounds_file = f"./temp/{project_name}/rounds.txt"
    flag_file = f"./temp/{project_name}/training_flag.txt"
    server_round=0
    while True:  # Run indefinitely, but handle exit conditions
        server_round = load_round_number(rounds_file)
        is_training_flag = read_flag_from_file(flag_file)
        server_energy.update_round(server_round)
        if is_training_flag:
            if prev != "training":  # Only stop if there's a state change
                print("Client is training. Stopping the server for now.")
                server_energy.stop_energy_calculation()
                prev = "training"
        else:
            if prev != "idle":  # Only start if there's a state change
                print("Running Server Energy Calculation")
                server_energy.start_energy_calculation()
                prev = "idle"
        
        time.sleep(3)  # Small delay to prevent excessive CPU usage

        # Break condition (Add if there's a need to stop the loop)
        if flag is False:
            print("Server update loop stopped.")
            break

def simulate():
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    
    backend_config = {"client_resources": {"num_cpus": 16, "num_gpus": 1.0}} if device.type == "cuda" else {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    
    server_thread = server_energy.start_energy_calculation()
    checkserver = threading.Thread(target=server_update)
    checkserver.daemon = True
    checkserver.start()
    
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=5,
        backend_config=backend_config,
    )
    global flag
    flag = False
    checkserver.join()
    server_energy.stop_energy_calculation()
    # server_energy.calculate_total_energy(num_epochs)

def main():
    
    # Clean old outputs (optional)
    del_dir(f"./models_weights/{project_path}")
    del_dir(f"./results/FL/{project_name}")
    del_dir(f"./temp/{project_name}")

    # Train base model
    model = YOLO('yolov8n.pt')
    data_yaml_path = f"./data.yaml"
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    model.train(
        data=data_yaml_path,
        epochs=1,
        imgsz=640,
        freeze=10,
        workers=4,
        batch=batch_size,
        device=device,
        lr0=learning_rate,
        lrf=lrf,
        val=False,
        optimizer='SGD',
    )

    model.save(f'yolo_base.pt')
    model_parameters = get_parameters(model)

    start_time = time.time()
    simulate()
    end_time = time.time()

    print(f"✅ Finished {project_name} in {end_time - start_time:.2f}s")
    server_energy.final_calculation()


if __name__ == '__main__':
    main()

