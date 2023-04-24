import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.stats import norm
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def extract_data(file = "./Data/osisaf_nh_sie_monthly.nc", dtype = "dict", month = 9):
    """ Extract Sea Ice extent (sie) from 1979 to 2022 for the month "month".
    Can return the data in two different parts depending on dtype parameter:
    dtype = "dict" return a dictionnary {1979: 7.556, 1980: 8.244,...} 
    dtype = "array" return a np.array [7.556, 8.244, ...]
    """
    ds = xr.open_dataset(file)
    if dtype == "dict":
        sie = {}
        for year in range(1979,2023):
            sie[year] = float(ds['sie'].sel(time = datetime(year,month,16)))
        
    elif dtype == "array":
        sie = []
        for year in range(1979,2023):
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

def APF_mean_std(year, may_sie, sept_sie):
    """ Anomaly Persistence Forecast. For a given year, compute the mean and std forcasted lying on the previous years sie at may and sept.
    ma.masked_invalid([array]) is used because some data are missing (NaN). masking them allowed the var and mean operations to handle it.
    """

    nb_year = year - 1979
    obs_sept_mean = np.mean(ma.masked_invalid([sept_sie[year] for year in range(1979,year)]))
    obs_may_mean = np.mean(ma.masked_invalid([may_sie[year] for year in range(1979,year)]))
    Forecast_mean = obs_sept_mean + (may_sie[year] - obs_may_mean)     #Eq (2) in instruction
    Forecast_var = (1/nb_year) * (np.var(ma.masked_invalid([may_sie[year] for year in range(1979,year)])) + np.var(ma.masked_invalid([sept_sie[year] for year in range(1979,year)])))    #Eq(3) in instruction
    
    Forecast_std = np.sqrt(Forecast_var)
    return [Forecast_mean, Forecast_std]

def Plot_APF(sept_sie, may_sie):
    plt.plot([year for year in range(1979,2022)], [sept_sie[year] for year in range(1979,2022)], label = 'Observed')
    plt.errorbar([year for year in range(1981,2022)], [APF_mean_std(year, may_sie, sept_sie)[0] for year in range(1981,2022)], [2*APF_mean_std(year, may_sie, sept_sie)[1] for year in range(1981,2022)],linestyle='None', marker='^', label = 'Forecasted')
    plt.grid()
    plt.legend()
    plt.ylabel('SIE [1e6 km^2]')
    plt.title("Sea ice extent in September. Forecasted values with 2 times std.")
    plt.show()

def proba_BTL(year,sept_sie, may_sie):
    """ Return the probability (between 0 and 1) that the September sie (for year = 'year') will be Below the Trend Line (BTL) 
    """
    #Compute the mean and std forecast for the year using APF_mean_std function
    Forecast = APF_mean_std(year,may_sie,sept_sie)
    Forecast_mean = Forecast[0]
    Forecast_std = Forecast[1]

    #Compute the trend line
    a, b, useless = trend_line_coeff(sept_sie)
    trend_line_forecast = b + a*year    # trend_line_forecast is the sie forecasted by simple extension of the trend line
    #Compute the probability of the BTL event
    proba_BTL = norm.cdf((trend_line_forecast - Forecast_mean)/Forecast_std)
    return proba_BTL

def proba_LPY(year, may_sie, sept_sie):
    """ Return the Probability (btwn 0 and 1) that sie will be Less than Previous Year (LPY)
    """
    #Compute the mean and std forecast for the year using APF_mean_std function
    Forecast = APF_mean_std(year,may_sie,sept_sie)
    Forecast_mean = Forecast[0]
    Forecast_std = Forecast[1]

    Last_year_sept_sie = sept_sie[year-1]
    #Compute the probability of the LPY event
    proba_LPY = norm.cdf((Last_year_sept_sie - Forecast_mean)/Forecast_std)
    return proba_LPY

def Forecast_LPY():
    fig, axs = plt.subplots(nrows = 2, ncols = 1)
    axs[0].bar([year for year in range(1981,2022)],[proba_LPY(year, may_sie,sept_sie) -1/2 for year in range(1981,2022)])
    axs[0].title.set_text('Forecasted probability that the sie will be less than previous year.')
    axs[0].set_ylabel('P(LPY) - 1/2')
    axs[0].grid()

    axs[1].plot([year for year in range(1979,2022)], [sept_sie[year] for year in range(1979,2022)], label = 'Observed')
    axs[1].errorbar([year for year in range(1981,2022)], [APF_mean_std(year, may_sie, sept_sie)[0] for year in range(1981,2022)], [2*APF_mean_std(year, may_sie, sept_sie)[1] for year in range(1981,2022)],linestyle='None', marker='^', label = 'Forecasted')
    axs[1].grid()
    axs[1].legend()
    axs[1].set_ylabel('SIE [1e6 km^2]')
    axs[1].title.set_text("Sea ice extent in September. Forecasted values with 2 times std.")
    plt.show()

def Brier_Score_LPY():

    # For each year btwn 1981 and 2023 (included) compute the probability of LPY event and store it in "Forecast" array.
    Forecast = []


    for year in range(1981,2023):
        Forecast.append(proba_LPY(year,sept_sie,may_sie))
   
    observation = []
    for year in range(1981,2023):
        last_year_sie = sept_sie[year-1]
        current_year_sie = sept_sie[year]
        
        #LPY = 1 if event occurs and LPY = 0 if not
        if last_year_sie > current_year_sie:
            LPY = 1
        else: 
            LPY = 0
        observation.append(LPY)
    Forecast = np.array(Forecast)
    observation = np.array(observation)

    BS = 1/(len(Forecast)) * np.sum(ma.masked_invalid([(Forecast[i] - observation[i])**2 for i in range(len(Forecast))])) # First equation in "Question8"
    
    return BS

def proba_NN_LPY():
    """
        Return prediction: a list of 43 [0< p- <1,0< p+ <1] elements, one per year btwn 1979 and 2022 (both included). 
        For each year, p+ gives the probability that next sept SIE will be bigger than last one and conversly for p-.

        Example: prediction[0] = [p-,p+], p- is the probability of observe a LPY event in 1980
                 prediction[-1] = [p-,p+], p- is the probability of observe a LPY event in 2022
    """
    def data_arange(SIE):
        """
            Returns:
                -x: Array with the SIE of the month from last september to current may (normalized)
                -y: Array with whether or not the current sept SIE is bigger or smaller than the last one.
        """
        sept_to_dec_last_year = SIE[:-1,8:]
        jan_to_may_current_year = SIE[1:,:5]

        x = np.concatenate((sept_to_dec_last_year,jan_to_may_current_year),axis = 1)
        y = SIE[1:,8:9]
        # On rempli y de 0 et de 1. 0 si la SIE sera plus petite que l'année passée et 1 si plus grande.
        y = [int(np.modf(y[year]/x[year,0])[1]) for year in range(len(y))] 
        y = np.array(y)

        # Next tree following lines turns the output data in the form [.,.]: [1,0] if it was equals to 0 (smaller) and [0,1] if it was 1 (bigger)
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y.reshape(-1, 1)).toarray()
        # Input, We normalize the input datas.
        sc = StandardScaler()
        x = sc.fit_transform(x)
        return x,y
    
    file = "./Data/osisaf_nh_sie_monthly.nc"
    ds = xr.open_dataset(file)
    SIE = np.zeros((44,12))
    for year in range(1979,2022):
        for month in range(1,13):
            SIE[year-1979,month-1] = float(ds['sie'].sel(time = datetime(year,month,16)))
    ds.close()
    x,y = data_arange(SIE)
    

    model = keras.models.load_model('./Machine_Learning/NN_model')

     
    prediction = model.predict(x)
    return prediction

def Forecast_NN_LPY():
    prediction = proba_NN_LPY()
    fig, axs = plt.subplots(nrows = 2, ncols = 1)
    axs[0].bar([year for year in range(1980,2023)],[prediction[line,1] - prediction[line,0] for line in range(len(prediction))])
    axs[0].title.set_text('Forecasted probability that the sie will be more than previous year.')
    axs[0].set_ylabel('P(More SIE) - P(Less SIE)')
    axs[0].grid()

    axs[1].plot([year for year in range(1979,2022)], [sept_sie[year] for year in range(1979,2022)], label = 'Observed')
    axs[1].grid()
    axs[1].legend()
    axs[1].set_ylabel('SIE [1e6 km^2]')
    axs[1].title.set_text("Sea ice extent in September. ")
    plt.show()

def Brier_score_NN():
    prediction = proba_NN_LPY()
    #Proba of LPY for each year from 1980 to 2022 (both included)
    Forecast = [prediction[year, 0] for year in range(len(prediction))]
    observation = []
    for year in range(1980,2023):
        last_year_sie = sept_sie[year-1]
        current_year_sie = sept_sie[year]
        
        #LPY = 1 if event occurs and LPY = 0 if not
        if last_year_sie > current_year_sie:
            LPY = 1
        else: 
            LPY = 0
        observation.append(LPY)
    Forecast = np.array(Forecast)
    observation = np.array(observation)
    print(Forecast)
    print(observation)
    BS = 1/(len(Forecast)) * np.sum(ma.masked_invalid([(Forecast[i] - observation[i])**2 for i in range(len(Forecast))])) # First equation in "Question8"
    
    return BS

def forecast_NN_septSIE(sie_range = 0.5):
    def data_arange(SIE,sie_range = sie_range):
        """
            Turns x and y in the good format for a SIE sept extend forecast.
        """
        sept_to_dec_last_year = SIE[:-1,8:]
        jan_to_may_current_year = SIE[1:,:5]
        x = np.concatenate((sept_to_dec_last_year,jan_to_may_current_year),axis = 1)
        y = SIE[1:,8:9]
        # The smallest data are always over 4*1e6 km^2 so we put the set 'to the ground'.
        y -= 4 
        # Subdivision of SEPT_SIE in range spanning sie_range each.
        y = np.array([np.modf(y[year]/(sie_range)) for year in range(len(y))])
        # ohe_y is init.
        ohe_y = np.zeros((len(y),int(np.max(y[:,1]))+2))
        """ ohe_y is fill. the correspondance btwn ohe_y and y is the following:
        e.g. ohe_y = [0,0,0,0.8,0.2,0] --> y = 3*0.8*sie_range + 4*0.2*sie_range """
        for line in range(len(ohe_y)):
            integer_part = int(y[line][1])
            float_part = float(y[line][0])
            ohe_y[line,integer_part] = 1-float_part
            ohe_y[line,integer_part+1] = float_part
        # Normalizaton of input datas
        sc = StandardScaler()
        x = sc.fit_transform(x)
        return x,y

    file = "./Data/osisaf_nh_sie_monthly.nc"
    ds = xr.open_dataset(file)
    SIE = np.zeros((43,12))
    for year in range(1979,2022):
        for month in range(1,13):
            SIE[year-1979,month-1] = float(ds['sie'].sel(time = datetime(year,month,16)))
    ds.close()
    x,y = data_arange(SIE)
    model = keras.models.load_model('./Machine_Learning/NN_mod') 
    pred_NN = model.predict(x)
    # Recovering the september predicted SIE in km^2 (pred_km) from the neural state (pred_NN)
    pred_km = np.zeros(len(pred_NN))
    for sample_pred in range(len(pred_NN)):
        predicted_val = 0
        for neuron in range(len(pred_NN[0])):
            predicted_val += sie_range * pred_NN[sample_pred][neuron] * neuron
        pred_km[sample_pred] = predicted_val
    pred_km += 4
    return pred_km

def Forecast_NN_septSIE():
    forecast = forecast_NN_septSIE()

    fig,axs = plt.subplots(nrows = 1, ncols = 2)
    def post_process():
        """
            Return a new forecast with biais correction.
        """
        true_mean = np.mean([sept_sie[year] for year in range(1980,2022)])
        Forecasted_mean = np.mean(ma.masked_invalid(forecast))
        biais =  Forecasted_mean - true_mean
        new_forecast = forecast - biais
        return new_forecast,biais 
    
    new_forecast,biais = post_process()
    axs[0].scatter(new_forecast,[sept_sie[year] for year in range(1980,2022)], label =f'post-processed: Biais correction = {round(biais,3)} 1e6km^2')
    axs[0].scatter(forecast,[sept_sie[year] for year in range(1980,2022)], label ='Forecast')
    # Display the diagonal
    axs[0].plot([0,100],[0,100],color = 'grey', linestyle='dashed')
    axs[0].set_xlim(3.5,10)
    axs[0].set_ylim(3.5,9)
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel('September SIE forecasted [1e6km^2]')
    axs[0].set_ylabel('September SIE from data [1e6km^2]')
    axs[0].title.set_text('Comparison Forecasted and real value. \n with and without post-processing.')

    axs[1].plot([year for year in range(1979,2022)], [sept_sie[year] for year in range(1979,2022)], label = 'Observed')
    axs[1].plot([year for year in range(1980,2022)], [new_forecast[year] for year in range(len(new_forecast))], label = 'NN Forcasted post processed')
    axs[1].plot([year for year in range(1980,2022)], [forecast[year] for year in range(len(new_forecast))], label = 'NN Forcasted')
    axs[1].grid()
    axs[1].legend()
    axs[1].set_ylabel('SIE [1e6 km^2]')
    axs[1].title.set_text("Sea ice extent in September. ")
    plt.show()

sept_sie = extract_data(dtype="dict", month=9)
may_sie = extract_data(dtype="dict", month=5)
Forecast_NN_septSIE()
