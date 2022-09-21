import subprocess

import pandas as pd
import timeit
import os
import json
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger()


##################Function to get model predictions
def model_predictions(X, model_file):
    # read the deployed model and a test dataset, calculate predictions
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    y_pred_as_list = [int(item) for item in y_pred]

    return y_pred_as_list  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary(df):
    cols = df.columns
    stats = ('mean', 'median', 'std')
    required_stats = {key: [value for value in stats] for key in cols}
    res = df.agg(required_stats)

    res_list = []
    for row, s in res.iterrows():
        for col, item in s.items():
            res_list.append(f'{row}({col})={item}')

    # calculate summary statistics here
    return res_list  # return value should be a list containing all summary statistics


##################Function to count proportion of missing data by variable

def count_na_percentage(df):
    nas = df.isna()
    count = nas.sum(axis=0).to_numpy()
    percentage = 100. * count / len(df)
    percentage = [float(item) for item in percentage]
    return percentage


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py

    start = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start

    start = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start

    return [ingestion_time, training_time]  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    outdated_report = subprocess.check_output(['pip', 'list', '--outdated'])
    return outdated_report.decode('ascii')


if __name__ == '__main__':
    ##################Load config.json and get environment variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_folder_path = os.path.join(config['output_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])
    test_data_path = os.path.join(config['test_data_path'])

    df_test = pd.read_csv(f'{test_data_path}/testdata.csv')
    X_test = df_test.drop(['exited', 'corporation'], axis=1)
    y_pred = model_predictions(X_test, f'{output_model_path}/trainedmodel.pkl')
    logging.info(f'Obtained {len(y_pred)} prediciton(s) from batch of {len(X_test)} sample(s).')

    df = pd.read_csv(f'{output_folder_path}/finaldata.csv')
    X = df.drop(['corporation'], axis=1)
    summary = dataframe_summary(X)
    logging.info(f'Dataset summary: {summary}')

    na_percentage = count_na_percentage(X)
    logging.info(f'NA percentage by variable (column): {na_percentage}')

    input_folder_path = config['input_folder_path']
    exec_times = execution_time()
    logging.info(f'Execution time for ingestion: {exec_times[0]}s   Execution time for training: {exec_times[1]}s')
    print('\nOutdated python packages')
    report = outdated_packages_list()
    print(report)
    # print(report.decode('ascii'))
