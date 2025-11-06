## Overview
This project is currently being developed to use features extracted from SpArcFiRe outputs to predict Hubble T-type classifications of galaxies with open-source machine learning algorithms.

## Data
This project currently uses the EFIGI dataset to find T-Scores, and SpArcFiRe outputs on SDSS galaxies also available in EFIGI. 

From EFIGI (https://www.astromatic.net/projects/efigi/) you need EFIGI_attributes.txt and EFIGI_SDSS.txt.
The TSV file for the SpArcFiRe outputs is called SDSS+SpArcFiRe_r+SFR.tsv, and must be found through UCI openlab at /extra/wayne1/research/drdavis/SDSS/SpArcFiRe/2016-09/SDSS+SpArcFiRe_r+SFR.tsv
None of this data is included in this github repository. You can also find all the data through UCI openlab at /extra/wayne2/preserve/mkfang/sparcfire-morphology/data

To load the data, you need to run **main()** in **predictor.py** with **firstload()** and the correct file paths for the data as parameters. If you download EFIGI on your own, you may need to delete the leading comma. **firstload()** will automatically merge the EFIGI files together, and **main()** automatically calls **filter_merge()** to merge the EFIGI and SpArcFiRe data for common galaxies only. You only need to call **firstload()** once, so comment it out once it's successfully run once.

The final xdata and ydata are also available as xdata.csv and ydata.csv in /extra/wayne2/preserve/mkfang/sparcfire-morphology/data

## Requirements
- Python
- Dependencies:
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - XGBoost
    - TQDM
    - Pickle
