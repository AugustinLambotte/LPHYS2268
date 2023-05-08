import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.stats import norm
import io

class AP_frcst():
    def __init__(self,file = "./Data/osisaf_nh_sie_monthly.nc",):
        """ Class: Anomaly Persistance Forecast
         
            Extract Sea Ice extent (sie) from 1979 to 2022.
            Stock 4 data. sie for sept and may and in form of array (_a) or dict (_d)
        """
        ds = xr.open_dataset(file)
        
        self.sept_sie_d = {}
        for year in range(1979,2023):
            self.sept_sie_d[year] = float(ds['sie'].sel(time = datetime(year,9,16)))
            
        sept_sie_a = []
        for year in range(1979,2023):
            sept_sie_a.append(float(ds['sie'].sel(time = datetime(year,9,16))))
        self.sept_sie_a = np.array(sept_sie_a)

        self.may_sie_d = {}
        for year in range(1979,2023):
            self.may_sie_d[year] = float(ds['sie'].sel(time = datetime(year,3,16)))
            
        may_sie_a = []
        for year in range(1979,2023):
            may_sie_a.append(float(ds['sie'].sel(time = datetime(year,3,16))))
        self.may_sie_a = np.array(may_sie_a)
        ds.close()

    def plot_data(self):
        """ 
        """
        def trend_line_coeff():
            """ Return the coefficient of the trend line.
            """
            mean_sie = np.mean([sie for sie in self.sept_sie_d.values()])
            mean_year = 2000.5
            num = 0
            denum = 0
            for year in self.sept_sie_d:
                num += (year - mean_year) * (self.sept_sie_d[year] - mean_sie)
                denum += (year - mean_year)**2
            a = num/denum
            b = mean_sie - a * mean_year
            forecast = b + a * 2023
            return a,b, forecast
        
        def trend_line(year,a,b):
            return b + a*year
        a,b,f = trend_line_coeff()
        
        fig ,axs = plt.subplots(nrows = 1, ncols = 1)

        axs.plot([year for year in range(1979,2023)], self.sept_sie_a)
        axs.plot([year for year in range(1979,2023)], [trend_line(year,a,b) for year in range(1979,2023)])
        axs.grid()
        axs.set_ylabel('SIE [1e6 km^2]')
        axs.set_title("Sea ice extent in September")
        plt.show()

    def APF_mean_std(self,year):
        """ For a given year, compute the mean and std forcasted lying on the previous years sie at may and sept.
        ma.masked_invalid([array]) is used because some data are missing (NaN). masking them allowed the var and mean operations to handle it.
        """

        nb_year = year - 1979
        obs_sept_mean = np.mean(ma.masked_invalid([self.sept_sie_d[year] for year in range(1979,year)]))
        obs_may_mean = np.mean(ma.masked_invalid([self.may_sie_d[year] for year in range(1979,year)]))
        Forecast_mean = obs_sept_mean + (self.may_sie_d[year] - obs_may_mean)     #Eq (2) in instruction
        Forecast_var = (1/nb_year) * (np.var(ma.masked_invalid([self.may_sie_d[year] for year in range(1979,year)])) + np.var(ma.masked_invalid([self.sept_sie_d[year] for year in range(1979,year)])))    #Eq(3) in instruction
        
        Forecast_std = np.sqrt(Forecast_var)
        return [Forecast_mean, Forecast_std]

    def Plot_APF(self):
        plt.plot([year for year in range(1979,2022)], [self.sept_sie_d[year] for year in range(1979,2022)], label = 'Observed')
        plt.errorbar([year for year in range(1981,2022)], [self.APF_mean_std(year)[0] for year in range(1981,2022)], [2*self.APF_mean_std(year)[1] for year in range(1981,2022)],linestyle='None', marker='^', label = 'Forecasted')
        plt.grid()
        plt.legend()
        plt.ylabel('SIE [1e6 km^2]')
        plt.title("Sea ice extent in September. Forecasted values with 2 times std.")
        plt.show()

    def proba_BTL(self,year):
        """ Return the probability (between 0 and 1) that the September sie (for year = 'year') will be Below the Trend Line (BTL) 
        """
        def trend_line_coeff():
            """ Return the coefficient of the trend line.
            """
            mean_sie = np.mean([sie for sie in self.sept_sie_d.values()])
            mean_year = 2000.5
            num = 0
            denum = 0
            for year in self.sept_sie_d:
                num += (year - mean_year) * (self.sept_sie_d[year] - mean_sie)
                denum += (year - mean_year)**2
            a = num/denum
            b = mean_sie - a * mean_year
            forecast = b + a * 2023
            return a,b, forecast
        
        #Compute the mean and std forecast for the year using APF_mean_std function
        Forecast = self.APF_mean_std(year)
        Forecast_mean = Forecast[0]
        Forecast_std = Forecast[1]

        #Compute the trend line
        a, b, useless = trend_line_coeff()
        trend_line_forecast = b + a*year    # trend_line_forecast is the sie forecasted by simple extension of the trend line
        #Compute the probability of the BTL event
        proba_BTL = norm.cdf((trend_line_forecast - Forecast_mean)/Forecast_std)
        return proba_BTL

    def Forecast_LPY(self):
        def proba_LPY(year):
            """ Return the Probability (btwn 0 and 1) that sie will be Less than Previous Year (LPY)
            """
            #Compute the mean and std forecast for the year using APF_mean_std function
            Forecast = self.APF_mean_std(year)
            Forecast_mean = Forecast[0]
            Forecast_std = Forecast[1]

            Last_year_sept_sie = self.sept_sie_d[year-1]
            #Compute the probability of the LPY event
            proba_LPY = norm.cdf((Last_year_sept_sie - Forecast_mean)/Forecast_std)
            return proba_LPY

        fig, axs = plt.subplots(nrows = 2, ncols = 1)
        axs[0].bar([year for year in range(1981,2022)],[proba_LPY(year) -1/2 for year in range(1981,2022)])
        axs[0].title.set_text('Forecasted probability that the sie will be less than previous year.')
        axs[0].set_ylabel('P(LPY) - 1/2')
        axs[0].grid()

        axs[1].plot([year for year in range(1979,2022)], [self.sept_sie_d[year] for year in range(1979,2022)], label = 'Observed')
        axs[1].errorbar([year for year in range(1981,2022)], [self.APF_mean_std(year)[0] for year in range(1981,2022)], [2*self.APF_mean_std(year)[1] for year in range(1981,2022)],linestyle='None', marker='^', label = 'Forecasted')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel('SIE [1e6 km^2]')
        axs[1].title.set_text("Sea ice extent in September. Forecasted values with 2 times std.")
        plt.show()

    def Brier_Score_LPY(self):

        def proba_LPY(year):
            """ Return the Probability (btwn 0 and 1) that sie will be Less than Previous Year (LPY)
            """
            #Compute the mean and std forecast for the year using APF_mean_std function
            Forecast = self.APF_mean_std(year)
            Forecast_mean = Forecast[0]
            Forecast_std = Forecast[1]

            Last_year_sept_sie = self.sept_sie_d[year-1]
            #Compute the probability of the LPY event
            proba_LPY = norm.cdf((Last_year_sept_sie - Forecast_mean)/Forecast_std)
            return proba_LPY
        
        # For each year btwn 1981 and 2023 (included) compute the probability of LPY event and store it in "Forecast" array.
        Forecast = []


        for year in range(1981,2023):
            Forecast.append(proba_LPY(year))
    
        observation = []
        for year in range(1981,2023):
            last_year_sie = self.sept_sie_d[year-1]
            current_year_sie = self.sept_sie_d[year]
            
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



APF = AP_frcst()
print(APF.Brier_Score_LPY())