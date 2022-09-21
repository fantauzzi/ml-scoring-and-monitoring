from scoring import score_model
from pathlib import Path
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger()

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

trained_model_file = f'{output_model_path}/trainedmodel.pkl'
latest_score_file = f'{prod_deployment_path}/latestscore.txt'
new_score_file = f'{output_model_path}/latestscore.txt'
test_data_file = f'{test_data_path}/testdata.csv'
final_data_file = f'{output_folder_path}/finaldata.csv'

##################Check and read new data

logging.info(f'Current directory is {os.getcwd()}')
os.system('python ingestion.py')
# first, read ingestedfiles.txt
ingestedfiles_filename = f'{prod_deployment_path}/ingestedfiles.txt'
logging.info(f'Reading list of ingested files from {ingestedfiles_filename}')
with open(ingestedfiles_filename) as f:
    line_from_file = f.readline()
    if line_from_file[-1] == '\n':
        line_from_file = line_from_file[:-1]
    ingested_files = line_from_file.split(',')

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_files = Path(input_folder_path).glob('*.csv')
input_files = [str(file_name) for file_name in input_files if file_name.is_file()]

ingested_files = set(ingested_files)
for file_name in input_files:
    if not file_name in ingested_files:
        break
else:  ###################Deciding whether to proceed, part 1
    logging.info('No new ingested files, exiting')
    exit(1)  # if you found new data, you should proceed. otherwise, do end the process here

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
os.system('python training.py')
os.system('python scoring.py')

with open(latest_score_file, 'rt') as f:
    line_from_file = f.readline()
    if line_from_file[-1] == '\n':
        line_from_file = line_from_file[:-1]
    previous_score = float(line_from_file)

new_score = score_model(trained_model_file, final_data_file, new_score_file)
logging.info(f'Previous model score is {previous_score} -  New model score is {new_score}')
##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if new_score >= previous_score:
    logging.info('New model score indicates no drift, exiting')
    exit(1)

##################Re-deployment
logging.info('New model score indicates drift, proceeding with re-deployment')
# if you found evidence for model drift, re-run the deployment.py script

os.system('python deployment.py')

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
os.system('python diagnostics.py')
os.system('python reporting.py')
os.system('python apicalls.py')
