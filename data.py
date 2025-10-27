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
    print(len(final))

    return final