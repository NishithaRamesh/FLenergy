import pandas as pd

def read_tdp_values_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
    
    tdp_values = {}
    # Assuming 'Name' and 'TDP' are the columns in your CSV
    for index, row in df.iterrows():
        cpu_model = row['Name']
        tdp_str = row['TDP']
        try:
            tdp = float(tdp_str)  # Or you can round/convert it as needed
        except ValueError:
            tdp_str = tdp_str.split('.')[0] # For using with examples only when values lik 27.2.5
            try:
                tdp = int(tdp_str)  # Convert the first part to an integer
            except ValueError:
                tdp = 0  # Handle the case where conversion fails
        tdp_values[cpu_model] = tdp

    return tdp_values