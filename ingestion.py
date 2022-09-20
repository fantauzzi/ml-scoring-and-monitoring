import pandas as pd
import csv
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime


#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # check for datasets, compile them together, and write to an output file
    input_files = Path(input_folder_path).glob('*.csv')
    input_files = [str(file_name) for file_name in input_files if file_name.is_file()]
    df = None
    for file_name in input_files:
        df_from_file = pd.read_csv(file_name)
        df = df_from_file.copy() if df is None else pd.concat([df, df_from_file], ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv(f'{output_folder_path}/finaldata.csv', index=False)

    with open(f'{output_folder_path}/ingestedfiles.txt', 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(input_files)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    Path(output_folder_path).mkdir(exist_ok=True)
    merge_multiple_dataframe(input_folder_path, output_folder_path)
