import xarray as xr 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.stats import norm
import seaborn as sns

import NN_model
class ML_frcst():

    def __init__(self,interp_deg = 1,clim_time = 10,is_siv = True, epochs = 50, file_sie = "./Data/osisaf_nh_sie_monthly.nc", file_siv = "Data/PIOMAS.2sst.monthly.Current.v2.1.txt"):
        """
            This class is used to predict next september SIE based on previous SIE and SIV measurement.
            It uses the class NN_model to build and train the network before using it.

            Parameters:
            ----------
            clim_time | int: is the range of time over which we want to compute the trend line.
            interp_deg | int: is the degree of interpolation if the trend line. Should be keep to 1 in practice.
            is_siv | bool: True if we want to use the SIV data False if not
            epoch | int: Number of epoch for training NN
        """
        def data_arange(SIE,SIV):
            """
                Turns x in the good format for a SIE sept extend forecast based on the shape of NN_model.x tensor.
                Parameters:
                ----------
                SIE | list: list of array. Each list element is a N_year x 12 array. Each line for a year (year have to be consecutive)
                            and each column for a month. filled with monthl SIE in km^2
                SIV | list: list of array. Each list element is a N_year x 12 array. Each line for a year (year have to be consecutive)
                            and each column for a month. filled with monthl SIV in km^2
            """
            climatology_sie = [] # Climatological trend sept sie mean of the "clim_time" last years of the considered year.
            climatology_siv = [] # Climatological trend sept siv mean of the "clim_time" last years of the considered year.
            summer_sie = SIE[clim_time:,:5] # SIE of the last 5 month (jan -> may)
            summer_siv = SIV[clim_time:,:5] # SIV of the last 5 month (jan -> may)

            
            for year in range(clim_time,len(SIE)): 
                deg = self.interp_deg
                coeff_sie = np.polyfit([y for y in range(clim_time)],SIE[year - clim_time:year,8],deg = deg)
                coeff_siv = np.polyfit([y for y in range(clim_time)],SIV[year - clim_time:year,8],deg = deg)
                current_climatology_sie = 0
                current_climatology_siv = 0
                for i in range(len(coeff_sie)):
                    current_climatology_sie += coeff_sie[i] * (clim_time)**(deg-i) 
                    current_climatology_siv += coeff_siv[i] * (clim_time)**(deg-i) 
                climatology_sie.append([current_climatology_sie])
                climatology_siv.append([current_climatology_siv])
                
            climatology_sie = np.array(climatology_sie)
            climatology_siv = np.array(climatology_siv)

            if self.is_siv:
                x = np.concatenate((climatology_sie,
                                climatology_siv,
                                summer_sie,
                                summer_siv),axis = 1) 
            else:
                x = np.concatenate((climatology_sie,summer_sie,),axis = 1)
            

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
        self.SIV = extract_SIV() #[km^3]     

        
        # Creating the neural model
        NN = NN_model.NN(interp_deg = interp_deg, is_siv = self.is_siv, clim_time = clim_time)
        NN.constr(epochs = epochs)


        self.model = NN.model_SIEFrcst
        self.clim_time = clim_time
        self.interp_deg = interp_deg
        self.sept_sie = self.SIE[:,8] #[km^2]
        self.x = data_arange(self.SIE,self.SIV)

    def SIE_frcst(self, show = True):
        """
            Create self.mean_frcsted and self.std_frcsted. The mean and std of the normal distribution forcasted for 
            sept SIE over the range of time available.
        """
        def post_process():
            """
                Return a new forecast with biais and variability correction.
            """
            #-----------
            # Mean Biais
            #-----------
            true_mean = np.mean([self.sept_sie[year-1979] for year in range((1979+self.clim_time),2023)])
            Forecasted_mean = np.mean(mean_frcsted)
            biais =  Forecasted_mean - true_mean
            unbiaised_frcst = mean_frcsted - biais
            
            #------------------
            # Variability Biais
            #------------------
            true_std = np.std([self.sept_sie[year-1979] for year in range((1979+self.clim_time),2023)])
            frcst_std = np.std(mean_frcsted)

            variability_ampl = true_std/frcst_std # As defined slide 11 Postprocessed
            new_frcst = (unbiaised_frcst - np.mean(unbiaised_frcst))*variability_ampl + np.mean(unbiaised_frcst)
            
            return new_frcst,biais, variability_ampl 
        
        #----------------------------
        # Forecasting of mean and std
        #----------------------------
        forecast = self.model.predict(self.x)
        mean_frcsted = forecast[:,0]
        std_frcsted = forecast[:,1]

        #----------------
        # Post-processing
        #----------------
        mean_postprcssed,biais,variability_ampl = post_process()

        if show:
            fig,axs = plt.subplots(nrows = 1, ncols = 2)

            #-------------
            # Scatter plots
            #--------------
            axs[0].scatter(mean_postprcssed,[self.sept_sie[i] for i in range((self.clim_time),len(self.sept_sie))], label =f'post-processed: Biais correction = {round(biais,3)} 1e6km^2\nAmpliccation of variability = {round(variability_ampl,3)}')
            axs[0].scatter(mean_frcsted,[self.sept_sie[i] for i in range((self.clim_time),len(self.sept_sie))], label ='Forecast')
            
            # Display the trend line of post-processed and unpost-processed data.
            sns.regplot(x=mean_postprcssed, y=[self.sept_sie[i] for i in range(self.clim_time,len(self.sept_sie))], ci=False, line_kws={'color':'blue', 'linestyle':'dashed'}, ax=axs[0])
            sns.regplot(x=mean_frcsted, y=[self.sept_sie[i] for i in range(self.clim_time,len(self.sept_sie))], ci=False, line_kws={'color':'orange','linestyle':'dashed'}, ax=axs[0])

            axs[0].grid()
            axs[0].legend()
            axs[0].set_xlabel('September SIE forecasted [1e6km^2]')
            axs[0].set_ylabel('September SIE from data [1e6km^2]')
            axs[0].title.set_text('Comparison Forecasted and real value. \n with and without post-processing.')

            #-------------
            # Time series 
            #-------------
            axs[1].plot([year for year in range(1979,2023)], [self.sept_sie[year-1979] for year in range(1979,2023)], label = 'Observed')      
            axs[1].errorbar([year for year in range((1979+self.clim_time),2023)], [mean_postprcssed[year] for year in range(len(mean_postprcssed))], [2*std_frcsted[year] for year in range(len(mean_postprcssed))],linestyle='dashed', marker='^', label = 'NN Forcasted post processed')
            axs[1].plot([year for year in range((1979+self.clim_time),2023)], [mean_frcsted[year] for year in range(len(mean_postprcssed))], label = 'NN Forcasted',color = 'grey', linestyle='dashed')
            axs[1].grid()
            axs[1].legend()
            axs[1].set_ylabel('SIE [1e6 km^2]')

            plt.show()

        self.mean_frcsted,self.std_frcsted = mean_postprcssed, std_frcsted
    
    def LPY(self):
        """
            Compute the probability of a LPY event using the self.mean_frcsted and self.std_frcsted created by self.SIE_frcst()
            and compute the scores associated to the forecast.
        """
        def Brier_score(proba_LPY):
            """
                Compute self.BS and self.BSS the Brier score and the Brier skill score, respectively.
                
                Parameters:
                ----------
                proba_LPY | array: An array with on each line the probability of the event for the corresponding year.
            """
            
            observation = []
            for year in range(self.clim_time-1, len(self.sept_sie)-1):
                last_year_sie = self.sept_sie[year]
                current_year_sie = self.sept_sie[year +1]
                
                #LPY = 1 if event occurs and LPY = 0 if not
                if last_year_sie > current_year_sie:
                    LPY = 1
                else: 
                    LPY = 0
                observation.append(LPY)
            
            self.observation = observation
            
            observed_occurence_frequence = np.sum(observation)/(len(observation))
            BS_ref = 1/(len(observation)) * np.sum([(observed_occurence_frequence - observation_)**2 for observation_ in observation])
            
            self.BS = 1/(len(proba_LPY)) * np.sum([(proba_LPY[i] - observation[i])**2 for i in range(len(proba_LPY))]) # First equation in "Question8"
            self.BSS = (self.BS - BS_ref)/(-BS_ref)
        
        #-----------------------------------------------------------------------------------------
        # Compute the probability of LPY event for each year by using the forcasted mean and std.
        #-----------------------------------------------------------------------------------------
        proba_LPY = np.zeros(len(self.mean_frcsted))

        for year in range(len(self.mean_frcsted)):
            Last_year_sept_sie = self.sept_sie[self.clim_time + year - 1]
            proba_LPY[year] = norm.cdf((Last_year_sept_sie - self.mean_frcsted[year])/self.std_frcsted[year])
        
        #------------------------------------------
        # Compute Brier score and Brier skill score.
        #------------------------------------------
        Brier_score(proba_LPY) 

        #----------
        # Plotting 
        #----------
        fig, axs = plt.subplots(nrows = 2, ncols = 1)
        axs[0].bar([year for year in range(1979 + self.clim_time,2023)],[proba_LPY[year] - 0.5 for year in range(len(proba_LPY))])
        axs[0].title.set_text(f'Forecasted probability that the sie will be less than previous year.\n BS = {self.BS}, BSS = {self.BSS}')
        axs[0].set_ylabel('P(Less SIE) - P(More SIE)')
        axs[0].scatter([year for year in range(self.clim_time + 1979,2023)],[self.observation[year] -1/2 for year in range(len(self.observation))])
        axs[0].grid()

        axs[1].plot([year for year in range(1979 + self.clim_time,2023)], [self.sept_sie[year-1979] for year in range(1979 + self.clim_time,2023)], label = 'Observed')      
        axs[1].errorbar([year for year in range((1979+self.clim_time),2023)], [self.mean_frcsted[year] for year in range(len(self.mean_frcsted))], [2*self.std_frcsted[year] for year in range(len(self.mean_frcsted))],linestyle='dashed', marker='^', label = 'NN Forcasted post processed')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel('SIE [1e6 km^2]')

        
        plt.show()


ML_frcst = ML_frcst(clim_time = 10, epochs = 40, is_siv=True)
ML_frcst.SIE_frcst(show = True)
ML_frcst.LPY()