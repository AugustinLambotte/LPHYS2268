import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


a = 6371 #Earth radius [km]
def grid_cell_area(phi):
    return float(a * np.cos(np.radians(phi)) * np.radians(5.625) * a * np.radians(5.625))

def SIE(year, month):
    """ Returns the Sea Ice Extent (SIE) for a given year (yyyy) between 0001 and 5272 and a given month (mm) btwn 01 and 12.
        Data used come from Mr. Massonet PlaSim run.
    """
    print(year)
    file = '/cofast/fmasson/LPHYS2268/CTRL/CTRL.0'+year+'.nc'
    ds = xr.open_dataset(file)
    SIE = 0 #Initializing SIE 
    for lat in ds['lat']:
        for lon in ds['lon']:
            SIE += grid_cell_area(lat) * float(ds.sic.sel(lat = lat,lon = lon,time = int(year + month + '15')))
    ds.close()
    return SIE

def SIE_plotting(time_range):
    sie = [SIE(str(f"{year:04d}"), '09') for year in range(time_range[0], time_range[1])]
    plt.plot(range(time_range[0],time_range[1]), sie)
    plt.savefig(f"SIE_{time_range[0]}-{time_range[1]}")

SIE_plotting([1,20])



