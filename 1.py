import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_data(file = "./Data/osisaf_nh_sie_monthly.nc", dtype = "array"):
    """ Extract september Sea Ice extent (sie) from 1979 to 2022. can return the data in two different parts depending on dtype parameter:
    dtype = "dict" return a dictionnary {1979: 7.556, 1980: 8.244,...} 
    dtype = "array" return a np.array [7.556, 8.244, ...]
    """
    ds = xr.open_dataset(file)
    if dtype == "dict":
        september_sie = {}
        for year in range(1979,2022):
            september_sie[year] = float(ds['sie'].sel(time = datetime(year,9,16)))
        
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

def plot(sept_sie, a, b):
    """ sept_sie must be an array
    """
    def trend_line(year,a,b):
        return b + a*year

    plt.plot([year for year in range(1979,2022)], sept_sie)
    plt.plot([year for year in range(1979,2022)], [trend_line(year,a,b) for year in range(1979,2022)])
    plt.grid()
    plt.ylabel('SIE [1e6 km^2]')
    plt.title("Sea ice extent in September")
    plt.show()

def trend_line_coeff(sept_sie):
    """ sept_sie must be a dict. Return the two coefficient a and b of the linear trend.
    Return also a simple forecast of the sept_2023 SIE based on this trend
    """
    mean_sie = np.mean([sie for sie in sept_sie.values()])
    mean_year = 2000.5
    num = 0
    denum = 0
    for year in sept_sie:
        num += (year - mean_year) * (sept_sie[year] - mean_sie)
        denum += (year - mean_year)**2
    a = num/denum
    b = mean_sie - a * mean_year
    forecast = b + a * 2023
    return a,b, forecast


    
a,b, forecast = trend_line_coeff(extract_data(dtype="dict"))
print(forecast)
plot(extract_data(dtype = "array"),a,b)