import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

def extract_data(file = "./Data/osisaf_nh_sie_monthly.nc", dtype = "array", month = 9):
    """ Extract Sea Ice extent (sie) from 1979 to 2022 for the month "month".
    Can return the data in two different parts depending on dtype parameter:
    dtype = "dict" return a dictionnary {1979: 7.556, 1980: 8.244,...} 
    dtype = "array" return a np.array [7.556, 8.244, ...]
    """
    ds = xr.open_dataset(file)
    if dtype == "dict":
        sie = {}
        for year in range(1979,2022):
            sie[year] = float(ds['sie'].sel(time = datetime(year,month,16)))
        
    elif dtype == "array":
        sie = []
        for year in range(1979,2022):
            sie.append(float(ds['sie'].sel(time = datetime(year,month,16))))
        sie = np.array(sie)
    else:
        print("! Incorrect data type !")
        ds.close()
        return 0
    ds.close()
    return sie

def plot_data(sept_sie, a, b):
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

def APF(year, may_sie, sept_sie):
    """ Anomaly Persistence forecast
    ma.masked_invalid([array]) is used because some data are missing (NaN). masking them allowed the var and mean operations to handle it.
    """

    nb_year = year - 1979
    obs_sept_mean = np.mean(ma.masked_invalid([sept_sie[year] for year in range(1979,year)]))
    obs_may_mean = np.mean(ma.masked_invalid([may_sie[year] for year in range(1979,year)]))
    Forecast_mean = obs_sept_mean + (may_sie[year] - obs_may_mean)     #Eq (2) in instruction
    Forecast_var = (1/nb_year) * (np.std(ma.masked_invalid([may_sie[year] for year in range(1979,year)])) + np.std(ma.masked_invalid([sept_sie[year] for year in range(1979,year)])))    #Eq(3) in instruction
    return [Forecast_mean, Forecast_var]

def Plot_APF(sept_sie, may_sie):
    #APF_mean = [APF(year, may_sie, sept_sie)[0] for year in range(1981,2022)]
    #print(APF(2000, may_sie, sept_sie))
    #print([APF(year, may_sie, sept_sie)[1] for year in range(1981,2022)])
    plt.plot([year for year in range(1979,2022)], [sept_sie[year] for year in range(1979,2022)], label = 'Observed')
    plt.errorbar([year for year in range(1981,2022)], [APF(year, may_sie, sept_sie)[0] for year in range(1981,2022)], [2*APF(year, may_sie, sept_sie)[1] for year in range(1981,2022)],linestyle='None', marker='^', label = 'Forecasted')
    plt.grid()
    plt.legend()
    plt.ylabel('SIE [1e6 km^2]')
    plt.title("Sea ice extent in September. Forecasted values with 2 times std.")
    plt.show()
sept_sie = extract_data(dtype="dict", month=9)
may_sie = extract_data(dtype="dict", month=5)
a,b, forecast = trend_line_coeff(extract_data(dtype="dict"))
Plot_APF(sept_sie,may_sie)
#plot_data(may_sie,a,b)