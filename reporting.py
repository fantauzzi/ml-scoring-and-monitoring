import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from diagnostics import model_predictions
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger()

##############Function for reporting
def score_model(X, output_model_path):
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    y_test = X['exited']
    X_test = X.drop(['exited'], axis=1)
    model_filename = f'{output_model_path}/trainedmodel.pkl'
    y_pred = model_predictions(X_test, model_filename)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    output_filename = f'{output_model_path}/confusionmatrix.png'
    plt.savefig(output_filename)
    logging.info(f'Saved confusion matrix for model {model_filename} into file {output_filename}')


if __name__ == '__main__':
    ###############Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)
    test_data_path = os.path.join(config['test_data_path'])
    output_folder_path = os.path.join(config['output_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])

    test_data_filename = f'{test_data_path}/testdata.csv'
    df_test = pd.read_csv(test_data_filename)
    X_test = df_test.drop(['corporation'], axis=1)
    logging.info(f'Preparing the confusion matrix based on test set {test_data_filename}')
    score_model(X_test, output_model_path)
