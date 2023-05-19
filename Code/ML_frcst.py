import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import tensorflow as tf
from scipy.stats import norm
from tensorflow import keras
from keras.layers import Dense, Lambda
from sklearn.preprocessing import StandardScaler
import io
import seaborn as sns

import NN_model
class ML_frcst():

    def __init__(self,clim_time = 20,is_siv = True, epochs = 50, file_sie = "./Data/osisaf_nh_sie_monthly.nc", file_siv = "Data/PIOMAS.2sst.monthly.Current.v2.1.txt",sie_range = 0.1*1e6):
        
        def data_arange(SIE,SIV):
            """
                Turns x in the good format for a SIE sept extend forecast.
            """
            climatology = [] # Here, we stock the climatological sept sie mean of the "clim_time" last years of the considered year.
            summer_sie = SIE[clim_time:,:5]
            summer_siv = SIV[clim_time:,:5]
            for year in range(clim_time,len(SIE)):   # First, we interpolate the last "clim_year" sept sie to find the next one.
                deg = 1
                coeff = np.polyfit(SIE[year - clim_time:year,8],[y for y in range(clim_time)],deg = deg)
                current_climatology = 0
                for i in range(len(coeff)):
                    current_climatology += coeff[i] * (year+1)**(deg-i) 
                climatology.append([current_climatology])
            climatology = np.array(climatology)

            if self.is_siv:
                x = np.concatenate((climatology,
                                summer_sie,
                                summer_siv),axis = 1) 
            else:
                x = np.concatenate((sept_to_dec_last_year_sie,jan_to_may_current_year_sie),axis = 1)
            
            #x = np.concatenate((sept_to_dec_last_year_sie,jan_to_may_current_year_sie),axis = 1)
            
            # Normalizaton of input datas
            #sc = StandardScaler()
            #x = sc.fit_transform(x)

            return x
        
        def extract_SIV():
            SIV = np.genfromtxt(file_siv, delimiter = " ")
            SIV = SIV[:-1,1:] * 1e3 #[km^3]
            return SIV
        
        def extract_SIE():
            ds = xr.open_dataset(file_sie)
            SIE = np.zeros((44,12))
            for year in range(1979,2023):
                for month in range(1,13):
                    SIE[year-1979,month-1] = float(ds['sie'].sel(time = datetime(year,month,16)))*1e6
            ds.close()
            return SIE
        
        self.is_siv = is_siv
        self.SIE = extract_SIE() #[km^2]
        
        
        # Creating the neural model
        NN = NN_model.NN(is_siv = self.is_siv, clim_time = clim_time)
        NN.form()
        NN.constr(epochs = epochs)
        NN.test()
        self.model = NN.model_SIEFrcst


        self.clim_time = clim_time
        self.sept_sie = self.SIE[:,8] #[km^2]
        self.SIV = extract_SIV() #[km^3]     
        self.sie_range = sie_range #[km^2]
        self.x = data_arange(self.SIE,self.SIV)

    def SIE_frcst(self, show = True):
        def post_process():
            """
                Return a new forecast with biais and variability correction.
            """

            # Mean Biais
            true_mean = np.mean([self.sept_sie[year-1980] for year in range((1980+self.clim_time),2023)])
            
            Forecasted_mean = np.mean(ma.masked_invalid(mean_frcsted))
            biais =  Forecasted_mean - true_mean
            unbiaised_frcst = mean_frcsted - biais
            
            # Variability Biais
            true_std = np.std([self.sept_sie[year-1980] for year in range((1980+self.clim_time),2022)])
            frcst_std = np.std(ma.masked_invalid(mean_frcsted))

            variability_ampl = true_std/frcst_std # As defined slide 11 Postprocessed
            new_frcst = (unbiaised_frcst - np.mean(ma.masked_invalid(unbiaised_frcst)))*variability_ampl + np.mean(ma.masked_invalid(unbiaised_frcst))
            
            return new_frcst,biais, variability_ampl 
        
        # Forecasting of mean and std
        forecast = self.model.predict(self.x)
        mean_frcsted = forecast[:,0]
        std_frcsted = forecast[:,1]

        # Post-processing
        mean_postprcssed,biais,variability_ampl = post_process()

        if show:
            fig,axs = plt.subplots(nrows = 1, ncols = 2)
            print(np.shape(mean_postprcssed))
            print(np.shape(mean_frcsted))
            print((self.clim_time),len(self.sept_sie))
            axs[0].scatter(mean_postprcssed,[self.sept_sie[i] for i in range((self.clim_time),len(self.sept_sie))], label =f'post-processed: Biais correction = {round(biais,3)} 1e6km^2\nAmpliccation of variability = {round(variability_ampl,3)}')
            axs[0].scatter(mean_frcsted,[self.sept_sie[i] for i in range((self.clim_time),len(self.sept_sie))], label ='Forecast')
            
            # Display the diagonal
            #axs[0].plot([0,1e8],[0,1e8],color = 'grey', linestyle='dashed')

            # Display the trend line of post-processed data.
            sns.regplot(x=mean_postprcssed, y=[self.sept_sie[i] for i in range(self.clim_time,len(self.sept_sie))], ci=False, line_kws={'color':'blue', 'linestyle':'dashed'}, ax=axs[0])

            # Display the trend line of unpost-processed data.
            sns.regplot(x=mean_frcsted, y=[self.sept_sie[i] for i in range(self.clim_time,len(self.sept_sie))], ci=False, line_kws={'color':'orange','linestyle':'dashed'}, ax=axs[0])

            # Scatter plots
            axs[0].grid()
            axs[0].legend()
            axs[0].set_xlabel('September SIE forecasted [1e6km^2]')
            axs[0].set_ylabel('September SIE from data [1e6km^2]')
            axs[0].title.set_text('Comparison Forecasted and real value. \n with and without post-processing.')

            # Time series
            axs[1].plot([year for year in range(1979,2023)], [self.sept_sie[year-1979] for year in range(1979,2023)], label = 'Observed')      
            axs[1].errorbar([year for year in range((1979+self.clim_time),2023)], [mean_postprcssed[year] for year in range(len(mean_postprcssed))], [2*std_frcsted[year] for year in range(len(mean_postprcssed))],linestyle='dashed', marker='^', label = f'NN Forcasted post processed, mean std = {np.mean(ma.masked_invalid(std_frcsted))}')
            axs[1].plot([year for year in range((1979+self.clim_time),2023)], [mean_frcsted[year] for year in range(len(mean_postprcssed))], label = 'NN Forcasted',color = 'grey', linestyle='dashed')
            axs[1].grid()
            axs[1].legend()
            axs[1].set_ylabel('SIE [1e6 km^2]')

            plt.show()
        self.mean_frcsted,self.std_frcsted = mean_postprcssed, std_frcsted
    
    def LPY(self):
        """
            This function forecast the probability of a LPY event using the sept SIE computed by self.Forecast_septSIE()       
        """
        def Brier_score(proba_LPY):

            #Proba of LPY for each year from 1980 to 2022 (both included)
            observation = []
            for year in range(len(self.sept_sie)-1):
                last_year_sie = self.sept_sie[year]
                current_year_sie = self.sept_sie[year +1]
                
                #LPY = 1 if event occurs and LPY = 0 if not
                if last_year_sie > current_year_sie:
                    LPY = 1
                else: 
                    LPY = 0
                observation.append(LPY)
            mask = np.zeros(len(observation))
            mask[6] = 1
            mask[8] = 1
            
            #we delete the observartions and forecast for 1986 nad 1988 because not enough data.
            observation = np.delete(observation,[6,8])
            print(observation)
            observed_occurence_frequence = np.sum(observation)/(len(observation))
            self.BS = 1/(len(proba_LPY)) * np.sum(ma.masked_invalid([(proba_LPY[i] - observation[i])**2 for i in range(len(proba_LPY))])) # First equation in "Question8"
            BS_ref = 1/(len(observation)) * np.sum([(observed_occurence_frequence - observation_)**2 for observation_ in observation])
            self.BSS = (self.BS - BS_ref)/(-BS_ref)
            self.observation = observation
        
        proba_LPY = np.zeros(len(self.mean_frcsted))

        for year in range(len(self.mean_frcsted)):
            Last_year_sept_sie = self.sept_sie[year]
            #Compute the probability of the LPY event
            proba_LPY[year] = norm.cdf((Last_year_sept_sie - self.mean_frcsted[year])/self.std_frcsted[year])
        proba_LPY = np.delete(proba_LPY,[6,8])
        
        Brier_score(proba_LPY) # Creation of self.BS and self.BSS
        fig, axs = plt.subplots(nrows = 2, ncols = 1)
        axs[0].bar([year for year in np.delete(np.arange(1980,2023),[6,8])],[proba_LPY[year] - 0.5 for year in range(len(proba_LPY))])
        axs[0].title.set_text(f'Forecasted probability that the sie will be less than previous year.\n BS = {self.BS}, BSS = {self.BSS}')
        axs[0].set_ylabel('P(Less SIE) - P(More SIE)')
        axs[0].scatter([year for year in np.delete(np.arange(1980,2023),[6,8])],[self.observation[year] -1/2 for year in range(len(self.observation))])
        axs[0].grid()

        axs[1].plot([year for year in range(1979,2023)], [self.sept_sie[year-1979] for year in range(1979,2023)], label = 'Observed')
        axs[1].errorbar([year for year in range(1980,2023)], [self.mean_frcsted[year] for year in range(len(self.mean_frcsted))], [2*self.std_frcsted[year] for year in range(len(self.mean_frcsted))],linestyle='dashed', marker='^', label = 'NN Forcasted post processed')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel('SIE [1e6 km^2]')
        axs[1].title.set_text("Sea ice extent in September. ")
        plt.show()


ML_frcst = ML_frcst(epochs = 10, is_siv=True)
ML_frcst.SIE_frcst(show = True)
ML_frcst.LPY()