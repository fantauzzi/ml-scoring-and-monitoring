import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from diagnostics import model_predictions


##############Function for reporting
def score_model(X, output_model_path):
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    y_test = X['exited']
    X_test = X.drop(['exited'], axis=1)
    y_pred = model_predictions(X_test, f'{output_model_path}/trainedmodel.pkl')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'{output_model_path}/confusionmatrix.png')
    # plt.show()


if __name__ == '__main__':
    ###############Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)
    test_data_path = os.path.join(config['test_data_path'])
    output_folder_path = os.path.join(config['output_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])

    df_test = pd.read_csv(f'{test_data_path}/testdata.csv')
    X_test = df_test.drop(['corporation'], axis=1)
    score_model(X_test, output_model_path)
