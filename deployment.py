from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import json
import shutil


####################function for deployment
def deploy(output_folder_path, output_model_path, prod_deployment_path):
    Path(prod_deployment_path).mkdir(exist_ok=True)
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    ingested_file = f'{output_folder_path}/ingestedfiles.txt'
    shutil.copy(ingested_file, prod_deployment_path)
    model_file = f'{output_model_path}/trainedmodel.pkl'
    shutil.copy(model_file, prod_deployment_path)
    score_file = f'{output_model_path}/latestscore.txt'
    shutil.copy(score_file, prod_deployment_path)


def main():
    ##################Load config.json and correct path variable
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_folder_path = os.path.join(config['output_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    deploy(output_folder_path, output_model_path, prod_deployment_path)


if __name__ == '__main__':
    main()
