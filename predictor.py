import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import xgboost as xgb


import matplotlib.pyplot as plt

def firstload():
    efigi_1 = pd.concat([chunk for chunk in tqdm(pd.read_csv('efigi-1.6/EFIGI_attributes.txt',  dtype={"objId": str}, comment="#", sep=r'\s+', na_values=["-1"], chunksize=100), desc='loading')])
    efigi_1.rename(columns={"PGCname": "PGC_name"}, inplace=True)
    
    efigi_2 = pd.concat([chunk for chunk in tqdm(pd.read_csv('efigi-1.6/EFIGI_SDSS.txt',  dtype={"objId": str}, comment="#", sep=r'\s+', na_values=["none", "-99.99"], chunksize=100), desc='loading')])

    efigi_final = pd.merge(efigi_1, efigi_2, on="PGC_name", how="inner")
    efigi_final.to_pickle("efigi_final.pkl")
      
    sparcfire_outputs = pd.concat([chunk for chunk in tqdm(pd.read_csv('SDSS+SpArcFiRe_r+SFR.tsv', sep=r'\t', chunksize=100) , desc='loading')])
    sparcfire_outputs.to_pickle("sparcfire_outputs.pkl")

def filter_merge():
    with open("efigi_final.pkl", 'rb') as pickle_file:
        efigi = pickle.load(pickle_file)
    with open("sparcfire_outputs.pkl", 'rb') as pickle_file:
        sparcfire = pickle.load(pickle_file)

    efigi["objId"] = efigi["objId"].dropna().astype(str)
    sparcfire["dr7objid"] = sparcfire["dr7objid"].astype(str)
    sparcfire.rename(columns={'dr7objid': 'objId'}, inplace=True)
    sparcfire = sparcfire[sparcfire["objId"].isin(efigi["objId"])]
    final = pd.merge(efigi, sparcfire, on="objId", how="inner")

    return final

def model(xdata, ydata):
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=20, max_features='sqrt', n_estimators=300, random_state=42)
    model.fit(xtrain, ytrain)

    evaluate(xtrain, xtest, ytrain, ytest, model, "rf_reg_results.png")
    
    return xtrain, xtest, ytrain, ytest, model

def evaluate(xtrain, xtest, ytrain, ytest, model, path):
    ypred = model.predict(xtest)
    plt.figure(figsize=(6,6))
    plt.scatter(ytest, ypred, alpha=0.6, edgecolor='k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.savefig(path)
    plt.show()
    print(f"mae: {mean_absolute_error(ypred, ytest)}")
    print(f"rmse: {np.sqrt(mean_absolute_error(ypred, ytest))}")
    print(f"r2: {r2_score(ypred, ytest)}")

def main():    
    final = filter_merge()

    x = final.drop(columns=['T', 'T_inf', 'T_sup'])
    x = x.select_dtypes(include=np.number)
    y = final["T"]
    '''
    y.hist()
    plt.title('Histogram of T')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig("t_distribution.png")
    plt.show()
    '''

    
    mask = y.notna()
    xdata = x[mask]
    ydata = y[mask]
    xdata = xdata.fillna(0)

    xtrain, xtest, ytrain, ytest, rfmodel = model(xdata, ydata)
    '''
    grid = GridSearchCV(rfmodel, {'n_estimators': [100, 200, 300],
                                    'max_depth': [10, 20, None],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4],
                                    'max_features': ['sqrt', 'log2']},verbose=3)
    grid.fit(xdata, ydata)
    print(grid.best_params_) 
    '''

    xgbmodel = xgb.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)
    xgbmodel.fit(xtrain, ytrain)
    evaluate(xtrain, xtest, ytrain, ytest, xgbmodel, "xgb_reg_results.png")

if __name__ == "__main__":
    main()