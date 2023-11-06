import json
import math
import os
import pickle
import sys

import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model, matrix, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        matrix (scipy.sparse.csr_matrix): Input matrix.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    labels = matrix.iloc[:,0]
    X = matrix.iloc[:,[1,2,3]].to_numpy(dtype='float32')
    predictions = model.predict(X, check_input=False)

    # Use dvclive to log a few simple metrics...
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1_score = metrics.f1_score(labels, predictions)
    
    if not live.summary:
        live.summary = {"accuracy": {}, "precision": {}, "recall": {}, "f1_score": {}}
    
    live.summary["accuracy"][split] = accuracy
    live.summary["precision"][split] = precision
    live.summary["recall"][split] = recall
    live.summary["f1_score"][split] = f1_score


def main():
    EVAL_PATH = "eval"

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)

    model_file = sys.argv[2]
    
    test_file = os.path.join(sys.argv[1])

    # Load model and data.
    with open(model_file, "rb") as fd:
        model = pickle.load(fd)
    
    test = pd.read_csv(test_file, header=None)

    # Evaluate train and test datasets.
    with Live(EVAL_PATH, dvcyaml=False) as live:
        #evaluate(model, train, "train", live, save_path=EVAL_PATH)
        evaluate(model, test, "test", live, save_path=EVAL_PATH)


if __name__ == "__main__":
    main()
