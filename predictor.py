import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt

from data import firstload, filter_merge

def model(xdata, ydata):
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2, random_state=42)
    print(len(xtrain))
    print(len(xtest))

    model = RandomForestRegressor(max_depth=20, max_features='sqrt', n_estimators=500, random_state=42)
    model.fit(xtrain, ytrain)

    evaluate(xtrain, xtest, ytrain, ytest, model, "rf_reg_results.png")
    
    return xtrain, xtest, ytrain, ytest, model

def accuracy(ytest, ypred, margin):
    within = np.abs(ytest - ypred) <= margin
    mock_acc = np.mean(within)
    print(f"{within.sum()}/{len(ytest)} predictions within ±{margin}. mock accuracy: {mock_acc:.3f}")

def evaluate(xtrain, xtest, ytrain, ytest, model, path):
    ypred = model.predict(xtest)
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

def main(): 
    # run ↓ if you haven't loaded the txt files in or if you want to re-laod them 
    # firstload("data\efigi-1.6\EFIGI_attributes.txt", "data\efigi-1.6\EFIGI_SDSS.txt",  "data\SDSS+SpArcFiRe_r+SFR.tsv")

    # loading pkl files and dropping unnecessary columns
    final = filter_merge("data/efigi_final.pkl", "data/sparcfire_outputs.pkl")
    # print(final.columns.tolist())
    x = final.drop(columns=['T', 'T_inf', 'T_sup', 'objId', 'SpecObjId', 'SDSS_dr8objid', 'GZ_dr8objid', 'objID', 'badBulgeFitFlag'])
    x = x.select_dtypes(include=np.number)
    y = final["T"]
    
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

    print(len(xdata))

    xtrain, xtest, ytrain, ytest, rfmodel = model(xdata, ydata)

    '''
    # grid tuning of model parametrs
    grid = GridSearchCV(rfmodel, {'n_estimators': [100, 200, 300],
                                    'max_depth': [10, 20, None],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4],
                                    'max_features': ['sqrt', 'log2']},verbose=3)
    grid.fit(xdata, ydata)
    print(grid.best_params_) 
    '''

if __name__ == "__main__":
    main()