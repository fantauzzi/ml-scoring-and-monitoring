import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
from pathlib import Path


#################Function for training the model
def train_model(dataset_csv_path, model_path):
    df = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')
    df.drop(['corporation'], axis=1, inplace=True)
    y = df['exited']
    X = df.drop(['exited'], axis=1, inplace=False)

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl

    with open(f'{model_path}/trainedmodel.pkl', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def main():
    ###################Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    Path(model_path).mkdir(exist_ok=True)

    train_model(dataset_csv_path, model_path)


if __name__ == '__main__':
    main()
