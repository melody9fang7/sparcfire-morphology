import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import xgboost as xgb
import matplotlib.pyplot as plt


from data import firstload, filter_merge

def model(xdata, ydata):
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2, random_state=42)
    
    pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
    ])
    
    pipe_rf.fit(xtrain, ytrain)
    predictions = pipe_rf.predict(xtest)
    evaluate(predictions, ytest, 'pipeline_rf_results.png')
    classify(predictions, ytest)
    
    return xtrain, xtest, ytrain, ytest, pipe_rf

def accuracy(ytest, ypred, margin):
    within = np.abs(ytest - ypred) <= margin
    mock_acc = np.mean(within)
    print(f"{within.sum()}/{len(ytest)} predictions within ±{margin}. mock accuracy: {mock_acc:.3f}")

def evaluate(ypred, ytest, path):
    print(f"mae: {mean_absolute_error(ytest, ypred)}")
    print(f"rmse: {root_mean_squared_error(ytest, ypred)}")
    print(f"r2: {r2_score(ytest, ypred)}")

    plt.figure(figsize=(6,6))
    plt.scatter(ytest, ypred, alpha=0.6, edgecolor='k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.savefig(path)
    plt.show()

    accuracy(ytest, ypred, 2)
    accuracy(ytest, ypred, 1.5)
    accuracy(ytest, ypred, 1)
    accuracy(ytest, ypred, 0.5)

def classify(ypred, ytest):
    ypred = np.round(ypred)
    print(f"fake classifier accuracy: {accuracy_score(ytest, ypred)}")
    evaluate(ypred, ytest, "rf_reg_to_class_results.png")

def main(): 
    # run ↓ if you haven't loaded the txt files in or if you want to re-laod them 
    # firstload("data\efigi-1.6\EFIGI_attributes.txt", "data\efigi-1.6\EFIGI_SDSS.txt",  "data\SDSS+SpArcFiRe_r+SFR.tsv")

    # loading pkl files
    x, y = filter_merge("data/efigi_final.pkl", "data/sparcfire_outputs.pkl")
    
    '''
    # plotting distribution of T
    y.hist()
    plt.title('Histogram of T')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig("t_distribution.png")
    plt.show()
    '''

    # masking out rows where y is na
    mask = y.notna()
    xdata = x[mask]
    ydata = y[mask]
    xdata = xdata.fillna(0)

    xtrain, xtest, ytrain, ytest, rfmodel = model(xdata, ydata)

if __name__ == "__main__":
    main()