import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt

def extract_data(file = "./Data/osisaf_nh_sie_monthly.nc", dtype = "dict"):
    """ Extract september Sea Ice extent (sie) from 1979 to 2022. can return the data in two different parts depending on dtype parameter:
    dtype = "dict" return a dictionnary {'1979': 7.556, '1980': 8.244,...} 
    dtype = "array" return a np.array [7.556, 8.244, ...]
    """
    ds = xr.open_dataset(file)
    if dtype == "dict":
        september_sie = {}
        for year in range(1979,2022):
            september_sie[f"{year}"] = float(ds['sie'].sel(time = datetime(year,9,16)))
        
    elif dtype == "array":
        september_sie = []
        for year in range(1979,2022):
            september_sie.append(float(ds['sie'].sel(time = datetime(year,9,16))))
        september_sie = np.array(september_sie)
    else:
        print("! Incorrect data type !")
        ds.close()
        return 0
    ds.close()
    return september_sie

def plot(sept_sie):
    plt.plot([year for year in range(1979,2022)], sept_sie)
    plt.grid()
    plt.title("Sea ice extent in September")
    plt.show()

plot(extract_data(dtype ="array"))
