import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

def firstload(efigi1path, efigi2path, sdsspath):
    efigi_1 = pd.concat([chunk for chunk in tqdm(pd.read_csv(efigi1path,  dtype={"objId": str}, comment="#", sep=r'\s+', na_values=["-1"], chunksize=100), desc='loading')])
    efigi_1.rename(columns={"PGCname": "PGC_name"}, inplace=True)
    
    efigi_2 = pd.concat([chunk for chunk in tqdm(pd.read_csv(efigi2path,  dtype={"objId": str}, comment="#", sep=r'\s+', na_values=["none", "-99.99"], chunksize=100), desc='loading')])

    # merging the 2 efigi files based on the PGC name so that we have both T score and the sdss7 id together
    efigi_final = pd.merge(efigi_1, efigi_2, on="PGC_name", how="inner")
    efigi_final.to_pickle("data/efigi_final.pkl")
      
    sparcfire_outputs = pd.concat([chunk for chunk in tqdm(pd.read_csv(sdsspath, sep=r'\t', chunksize=100) , desc='loading')])
    sparcfire_outputs.to_pickle("data/sparcfire_outputs.pkl")

def filter_merge(efigipklpath, sparcfirepklpath):
    with open(efigipklpath, 'rb') as file:
        efigi = pickle.load(file)
    with open(sparcfirepklpath, 'rb') as file:
        sparcfire = pickle.load(file)

    efigi["objId"] = efigi["objId"].dropna().astype(str)
    sparcfire["dr7objid"] = sparcfire["dr7objid"].astype(str)
    sparcfire.rename(columns={'dr7objid': 'objId'}, inplace=True)
    sparcfire = sparcfire[sparcfire["objId"].isin(efigi["objId"])]
    final = pd.merge(efigi, sparcfire, on="objId", how="inner")

    x = final.drop(columns=['T', 'T_inf', 'T_sup', 'objId', 'SpecObjId', 'SDSS_dr8objid', 'GZ_dr8objid', 'objID', 'badBulgeFitFlag', 
                            'PGC_name', 'name', 'warnings', 'fit_state', 'numArcsGE000', 'numArcsGE010', 'numArcsGE020', 'numArcsGE040', 
                            'numArcsGE050', 'numArcsGE055', 'numArcsGE060', 'numArcsGE065', 'numArcsGE070', 'numArcsGE075', 'numArcsGE080', 
                            'numArcsGE085', 'numArcsGE090', 'numArcsGE095', 'numArcsGE100', 'numArcsGE120', 'numArcsGE140', 'numArcsGE160', 
                            'numArcsGE180', 'numArcsGE200', 'numArcsGE220', 'numArcsGE240', 'numArcsGE260', 'numArcsGE280', 'numArcsGE300', 
                            'numArcsGE350', 'numArcsGE400', 'numArcsGE450', 'numArcsGE500', 'numArcsGE550', 'numArcsGE600', 'numDcoArcsGE000', 
                            'numDcoArcsGE010', 'numDcoArcsGE020', 'numDcoArcsGE040', 'numDcoArcsGE050', 'numDcoArcsGE055', 'numDcoArcsGE060', 
                            'numDcoArcsGE065', 'numDcoArcsGE070', 'numDcoArcsGE075', 'numDcoArcsGE080', 'numDcoArcsGE085', 'numDcoArcsGE090', 
                            'numDcoArcsGE095', 'numDcoArcsGE100', 'numDcoArcsGE120', 'numDcoArcsGE140', 'numDcoArcsGE160', 'numDcoArcsGE180', 
                            'numDcoArcsGE200', 'numDcoArcsGE220', 'numDcoArcsGE240', 'numDcoArcsGE260', 'numDcoArcsGE280', 'numDcoArcsGE300', 
                            'numDcoArcsGE350', 'numDcoArcsGE400', 'numDcoArcsGE450', 'numDcoArcsGE500', 'numDcoArcsGE550', 'numDcoArcsGE600', 
                            'badBulgeFitFlag', 'hasDeletedCtrClus', 'failed2revDuringMergeCheck', 'failed2revDuringSecondaryMerging', 
                            'failed2revInOutput', 'star_mask_used', 'noise_mask_used'])
    x = x.select_dtypes(include=np.number)
    y = final["T"]

    return x, y