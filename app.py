from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, count_na_percentage, execution_time, \
    outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

trained_model_file = f'{output_model_path}/trainedmodel.pkl'
latest_score_file = f'{output_model_path}/latestscore.txt'
test_data_file = f'{test_data_path}/testdata.csv'
final_data_file = f'{output_folder_path}/finaldata.csv'


# prediction_model = None

@app.route('/')
def home():
    return 'Howdy! This is the home page for the customer churn prediction API.'


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    dataset_file = request.get_json()['filepath']

    df = pd.read_csv(dataset_file)
    df = df.drop(['corporation', 'exited'], axis=1)
    y_pred = model_predictions(df, trained_model_file)
    return jsonify(y_pred)  # add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    # check the score of the deployed model
    f1 = score_model(trained_model_file, test_data_file, latest_score_file)

    return str(f1)  # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    df = pd.read_csv(final_data_file)
    X = df.drop(['corporation'], axis=1)
    summary = dataframe_summary(X)
    return jsonify(summary)  # return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    df = pd.read_csv(test_data_file)
    X = df.drop(['corporation'], axis=1)
    na_percentages = count_na_percentage(X)
    ingestion_time, training_time = execution_time()
    outdated_report = outdated_packages_list()
    res = {'na_percentages': na_percentages,
           'ingestion_time': ingestion_time,
           'training_time': training_time,
           'outdated_packages_report': outdated_report}

    # check timing and percent NA values
    return jsonify(res)  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
