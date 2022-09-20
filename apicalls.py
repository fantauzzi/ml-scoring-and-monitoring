import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

test_dataset_file = f'{test_data_path}/testdata.csv'

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction', json={'filepath': test_dataset_file}).text
print(response1)
response2 = requests.get(f'{URL}/scoring').text
print(response2)
response3 = requests.get(f'{URL}/summarystats').text
print(response3)

"""response2 = #put an API call here
response3 = #put an API call here
response4 = #put an API call here

#combine all API responses
responses = #combine reponses here

#write the responses to your workspace
"""


