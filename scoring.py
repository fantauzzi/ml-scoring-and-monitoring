from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json


#################Function for model scoring
def score_model(model_file, test_data_file, score_file):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    df = pd.read_csv(test_data_file)
    y = df['exited']
    X = df.drop(['exited', 'corporation'], axis=1)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred)
    with open(score_file, 'wt') as f:
        f.write(str(f1))
    return float(f1)


def main():
    #################Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])
    test_data_path = os.path.join(config['test_data_path'])

    score_model(f'{model_path}/trainedmodel.pkl', f'{test_data_path}/testdata.csv', f'{model_path}/latestscore.txt')


if __name__ == '__main__':
    main()
