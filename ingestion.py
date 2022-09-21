import pandas as pd
import csv
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger()


#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # check for datasets, compile them together, and write to an output file
    input_files = Path(input_folder_path).glob('*.csv')
    input_files = [str(file_name) for file_name in input_files if file_name.is_file()]
    logging.info(f'Ingesting {len(input_files)} file(s) from directory {input_folder_path}')
    df = None
    for file_name in input_files:
        df_from_file = pd.read_csv(file_name)
        df = df_from_file.copy() if df is None else pd.concat([df, df_from_file], ignore_index=True)
    df = df.drop_duplicates()
    final_data_filename = f'{output_folder_path}/finaldata.csv'
    logging.info(f'Saving ingested datasets into file {final_data_filename}')
    df.to_csv(final_data_filename, index=False)

    ingested_files_filename = f'{output_folder_path}/ingestedfiles.txt'
    logging.info(f'Saving the list of ingested files into {ingested_files_filename}')
    with open(ingested_files_filename, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(input_files)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    Path(output_folder_path).mkdir(exist_ok=True)
    merge_multiple_dataframe(input_folder_path, output_folder_path)
