import xarray as xr 
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.stats import norm
from tensorflow import keras
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import io
import seaborn as sns
class ML_frcst():

    def __init__(self, NN_model = './Machine_Learning/Models/NN',file_sie = "./Data/osisaf_nh_sie_monthly.nc", file_siv = "Data/PIOMAS.2sst.monthly.Current.v2.1.txt",sie_range = 0.1):
        
        def data_arange(SIE,SIV):
            """
                Turns x in the good format for a SIE sept extend forecast.
            """
            sept_to_dec_last_year_sie = SIE[:-1,8:]
            jan_to_may_current_year_sie = SIE[1:,:5]

            sept_to_dec_last_year_siv = SIV[:-1,8:]
            jan_to_may_current_year_siv = SIV[1:,:5]
            x = np.concatenate((sept_to_dec_last_year_sie,jan_to_may_current_year_sie,sept_to_dec_last_year_siv,jan_to_may_current_year_siv),axis = 1)
            #x = np.concatenate((sept_to_dec_last_year_sie,jan_to_may_current_year_sie,jan_to_may_current_year_siv),axis = 1)
            #x = np.concatenate((sept_to_dec_last_year_sie,jan_to_may_current_year_sie),axis = 1)
            
            # Normalizaton of input datas
            sc = StandardScaler()
            x = sc.fit_transform(x)
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
                    SIE[year-1979,month-1] = float(ds['sie'].sel(time = datetime(year,month,16)))
            ds.close()
            return SIE

        self.SIE = extract_SIE()
        self.sept_sie = self.SIE[:,8]
        self.SIV = extract_SIV()        
        self.sie_range = sie_range
        self.x = data_arange(self.SIE,self.SIV)
        self.model = keras.models.load_model(NN_model)



    ##### - LPY - #######
    def proba_LPY(self):
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

    def Forecast_LPY(self):
        prediction = self.proba_LPY()
        fig, axs = plt.subplots(nrows = 2, ncols = 1)
        axs[0].bar([year for year in range(1980,2023)],[prediction[line,1] - prediction[line,0] for line in range(len(prediction))])
        axs[0].title.set_text('Forecasted probability that the sie will be more than previous year.')
        axs[0].set_ylabel('P(More SIE) - P(Less SIE)')
        axs[0].grid()

        axs[1].plot([year for year in range(1979,2022)], [self.sept_sie[year-1979] for year in range(1979,2022)], label = 'Observed')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel('SIE [1e6 km^2]')
        axs[1].title.set_text("Sea ice extent in September. ")
        plt.show()

    def Brier_score(self):
        prediction = self.proba_LPY()
        #Proba of LPY for each year from 1980 to 2022 (both included)
        Forecast = [prediction[year, 0] for year in range(len(prediction))]
        observation = []
        for year in range(1980,2023):
            last_year_sie = self.sept_sie[year-1]
            current_year_sie = self.sept_sie[year]
            
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

    ##### - Sept SIE - #####
    def Forecast_septSIE(self):

        def predict_NN_septSIE():
            
           
            pred_NN = self.model.predict(self.x)
            # Recovering the september predicted SIE in km^2 (pred_km) from the neural state (pred_NN)
            pred_km = np.zeros(len(pred_NN))
            std_km = np.zeros(len(pred_NN))
            for sample_pred in range(len(pred_NN)):
                distribution = []
                predicted_val = 0
                for neuron in range(len(pred_NN[0])):
                    predicted_val += self.sie_range * pred_NN[sample_pred][neuron] * neuron
                    distribution.append(self.sie_range * pred_NN[sample_pred][neuron] * neuron)
                print(distribution)
                std_km[sample_pred] = np.std(np.array(distribution))
                pred_km[sample_pred] = predicted_val
            """ 
            print(np.std(pred_NN[1] * self.sie_range))
            print(np.std(pred_NN[-1]))
            plt.plot([self.sie_range * i for i in range(len(pred_NN[1]))], pred_NN[1])
            plt.plot([self.sie_range * i for i in range(len(pred_NN[5]))], pred_NN[5])
            plt.plot([self.sie_range * i for i in range(len(pred_NN[10]))], pred_NN[10])
            plt.plot([self.sie_range * i for i in range(len(pred_NN[-5]))], pred_NN[-5])
            plt.plot([self.sie_range * i for i in range(len(pred_NN[-1]))], pred_NN[-1])
            plt.show() """
            pred_km += 4
            return pred_km,std_km
        
        def post_process():
            """
                Return a new forecast with biais and variability correction.
            """

            # Mean Biais
            true_mean = np.mean([self.sept_sie[year-1980] for year in range(1980,2022)])
            Forecasted_mean = np.mean(ma.masked_invalid(forecast))
            biais =  Forecasted_mean - true_mean
            unbiaised_frcst = forecast - biais
            
            # Variability Biais
            true_std = np.std([self.sept_sie[year-1980] for year in range(1980,2022)])
            frcst_std = np.std(ma.masked_invalid(forecast))

            variability_ampl = true_std/frcst_std # As defined slide 11 Postprocessed
            new_frcst = (unbiaised_frcst - np.mean(ma.masked_invalid(unbiaised_frcst)))*variability_ampl + np.mean(ma.masked_invalid(unbiaised_frcst))
            
            return new_frcst,biais, variability_ampl      
        
        forecast, std = predict_NN_septSIE()
        fig,axs = plt.subplots(nrows = 1, ncols = 2)
        new_forecast,biais,variability_ampl = post_process()
        
        axs[0].scatter(new_forecast,[self.sept_sie[i] for i in range(1,len(self.sept_sie))], label =f'post-processed: Biais correction = {round(biais,3)} 1e6km^2\nAmpliccation of variability = {round(variability_ampl,3)}')
        axs[0].scatter(forecast,[self.sept_sie[year-1980] for year in range(1980,2023)], label ='Forecast')
        
        # Display the diagonal
        axs[0].plot([0,100],[0,100],color = 'grey', linestyle='dashed')

        # Display the trend line of post-processed data.
        sns.regplot(x=new_forecast, y=[self.sept_sie[i] for i in range(1,len(self.sept_sie))], ci=False, line_kws={'color':'blue', 'linestyle':'dashed'}, ax=axs[0])

        # Display the trend line of unpost-processed data.
        sns.regplot(x=forecast, y=[self.sept_sie[i] for i in range(1,len(self.sept_sie))], ci=False, line_kws={'color':'orange','linestyle':'dashed'}, ax=axs[0])

        # Scatter plots
        axs[0].set_xlim(3.5,10)
        axs[0].set_ylim(3.5,9)
        axs[0].grid()
        axs[0].legend()
        axs[0].set_xlabel('September SIE forecasted [1e6km^2]')
        axs[0].set_ylabel('September SIE from data [1e6km^2]')
        axs[0].title.set_text('Comparison Forecasted and real value. \n with and without post-processing.')

        # Time series
        axs[1].plot([year for year in range(1979,2023)], [self.sept_sie[year-1979] for year in range(1979,2023)], label = 'Observed')      
        axs[1].errorbar([year for year in range(1980,2023)], [new_forecast[year] for year in range(len(new_forecast))], [2*std[year] for year in range(len(new_forecast))],linestyle='dashed', marker='^', label = 'NN Forcasted post processed')
        axs[1].plot([year for year in range(1980,2023)], [forecast[year] for year in range(len(new_forecast))], label = 'NN Forcasted',color = 'grey', linestyle='dashed')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel('SIE [1e6 km^2]')

        plt.show()

ML_frcst = ML_frcst()

ML_frcst.Forecast_septSIE()