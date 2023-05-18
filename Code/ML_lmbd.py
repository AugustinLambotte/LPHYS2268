import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda
from sklearn.metrics import accuracy_score
import xarray as xr
import tensorflow_probability as tfp

############# - Extraction of the data - ###############
class NN:
    def __init__(self,file_siv = 'Machine_Learning/Data/SIV_mensual_90-5100_plsm.txt', file_sie = 'Machine_Learning/Data/SIE_mensual_90-5100_plsm.txt'):
        """
            This class "NN" is build to create a Neural Network model to predict futur Sea Ice Extend (SIE) 
            based on SIE and SIV (Sea ice volume) data from last September to current May.

            Two different constructions are possible:
                - The first predict the possiblity of a LPY event (i.e. next september SIE will be less than the last one).
                  to do this, first formating the data with self.formating_data_LPY() then construct the NN with 
                  self.construct_LPY(save = False) to record the model.
                - The second predict futur septembre SIE. First formating the data with self.formating_data_SIEfrcst() and construct
                  and save with construct_SIEfrcst(save = True). 

            Once created, these models can be test with self.test_LPY() and self.test_SIEfrcst(), respectively.
        """
        def data_arange(SIE_plsm,SIV_plsm, SIE_CESM1, SIV_CESM1, SIE_CESM2, SIV_CESM2):
            """
                Return:
                    x: An array with the mensual SIE from September of the previous year to may of the current (both included)
                       and siv from last septmber to current may concatenated 
                       e.g. x[year] = [sie_sept, sie_oct,...,sie_may,siv_sept,siv_oct,...siv_may].
                       This will be the input data.

                    y: An array of SIE september data, this will be used to compare with the data output.       
            """
            month_range_SIE = [9,5] #Range of month which will be used for predictant (e.g. [9,5] -> We use data from last sept to current may)
            month_range_SIV = [9,5]

            ############# - PLASIM - #############

            sept_to_dec_last_year_sie_plsm = SIE_plsm[:-1,month_range_SIE[0]-1:]
            jan_to_may_current_year_sie_plsm = SIE_plsm[1:,:month_range_SIE[1]]

            sept_to_dec_last_year_siv_plsm = SIV_plsm[:-1,month_range_SIV[0]-1:]
            jan_to_may_current_year_siv_plsm = SIV_plsm[1:,:month_range_SIV[1]]

            plsm_input = np.concatenate((sept_to_dec_last_year_sie_plsm,
                                jan_to_may_current_year_sie_plsm,
                                sept_to_dec_last_year_siv_plsm, 
                                jan_to_may_current_year_siv_plsm),axis = 1)
            
            """ plsm_input = np.concatenate((sept_to_dec_last_year_sie_plsm,
                                jan_to_may_current_year_sie_plsm),axis = 1) """
            
             ############# - CESM 2 - #############
           
            sept_to_dec_last_year_sie_cesm1 = SIE_CESM1[:-1,month_range_SIE[0]-1:]
            jan_to_may_current_year_sie_cesm1 = SIE_CESM1[1:,:month_range_SIE[1]]

            sept_to_dec_last_year_siv_cesm1 = SIV_CESM1[:-1,month_range_SIV[0]-1:]
            jan_to_may_current_year_siv_cesm1 = SIV_CESM1[1:,:month_range_SIV[1]]

            cesm1_input = np.concatenate((sept_to_dec_last_year_sie_cesm1,
                                jan_to_may_current_year_sie_cesm1,
                                sept_to_dec_last_year_siv_cesm1, 
                                jan_to_may_current_year_siv_cesm1),axis = 1)
            
            """ cesm1_input = np.concatenate((sept_to_dec_last_year_sie_cesm1,
                                jan_to_may_current_year_sie_cesm1),axis = 1) """
            
            ############# - CESM 1 - #############

            sept_to_dec_last_year_sie_cesm2 = SIE_CESM2[:-1,month_range_SIE[0]-1:]
            jan_to_may_current_year_sie_cesm2 = SIE_CESM2[1:,:month_range_SIE[1]]

            sept_to_dec_last_year_siv_cesm2 = SIV_CESM2[:-1,month_range_SIV[0]-1:]
            jan_to_may_current_year_siv_cesm2 = SIV_CESM2[1:,:month_range_SIV[1]]

            cesm2_input = np.concatenate((sept_to_dec_last_year_sie_cesm2,
                                jan_to_may_current_year_sie_cesm2,
                                sept_to_dec_last_year_siv_cesm2, 
                                jan_to_may_current_year_siv_cesm2),axis = 1)
            
            """ cesm2_input = np.concatenate((sept_to_dec_last_year_sie_cesm2,
                                jan_to_may_current_year_sie_cesm2),axis = 1) """
            

            x = np.concatenate((plsm_input, cesm1_input,cesm2_input),axis = 0)
            #x = np.concatenate((cesm1_input,cesm2_input),axis = 0)

            sept_plsm = SIE_plsm[1:,8:9]
            sept_cesm1 = SIE_CESM1[1:,8:9]
            sept_cesm2 = SIE_CESM2[1:,8:9]
            y = np.concatenate((sept_plsm, sept_cesm1,sept_cesm2), axis = 0)
            #y = np.concatenate(( sept_cesm1,sept_cesm2), axis = 0)

            return x,y
        SIE_mensual_plsm = np.genfromtxt(file_sie,delimiter=' ')
        SIV_mensual_plsm = np.genfromtxt(file_siv, delimiter =' ')

        SIE_mensual_CESM1 = np.genfromtxt('Machine_Learning/Data/CMIP/SIE_CESM1.txt', delimiter = ' ')
        SIV_mensual_CESM1 = np.genfromtxt('Machine_Learning/Data/CMIP/SIV_CESM1.txt', delimiter = ' ')

        SIE_mensual_CESM2 = np.genfromtxt('Machine_Learning/Data/CMIP/SIE_CESM2.txt', delimiter = ' ')
        SIV_mensual_CESM2 = np.genfromtxt('Machine_Learning/Data/CMIP/SIV_CESM2.txt', delimiter = ' ')

        SIE_mensual_CESM1 *= 1e6 # Passing from [1e6km^2] to [km^2]
        SIE_mensual_CESM2 *= 1e6 # Passing from [1e6km^2] to [km^2]
        
        self.x,self.y = data_arange(SIE_mensual_plsm,SIV_mensual_plsm,SIE_mensual_CESM1,SIV_mensual_CESM1,SIE_mensual_CESM2,SIV_mensual_CESM2)
        print(f"Number of training year = {len(self.x)}")
    def form(self, test_size = 0.01):
        
        
        # Normalization of input datas
        sc = StandardScaler()
        x = sc.fit_transform(self.x)
        # Splitting our data set in training and testing parts
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,self.y,test_size = test_size)

    def constr(self, epochs = 60,save = False):
        """
            Construct the neural network and train him to predict sept_SIE
        """
        def normal_distrib_loss(y_true, y_pred):
            """
            Normal distribution loss function.
            Assumes tensorflow backend.
            
            Parameters
            ----------
            y_true : tf.Tensor
                Ground truth values of predicted variable.
            y_pred : tf.Tensor
                n and p values of predicted distribution.
                
            Returns
            -------
            nll : tf.Tensor
                Negative log likelihood.
            """
            import tensorflow as tf
            # Separate the parameters
            mu, sigma = tf.unstack(y_pred, num=2, axis=-1)
            
            # Add one dimension to make the right shape
            mu = tf.expand_dims(mu, -1)
            sigma = tf.expand_dims(sigma, -1)
            
            # Calculate the negative log likelihood
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            loss = tf.reduce_mean(-dist.log_prob(y_true))   
             

            return loss
        
        def Gaussian_layer(x):
            """
            Lambda function for generating gaussian parameters
            n and p from a Dense(2) output.
            Assumes tensorflow 2 backend.
            
            
            Parameters
            ----------
            x : tf.Tensor
                output tensor of Dense layer
                
            Returns
            -------
            out_tensor : tf.Tensor
                
            """
            # Get the number of dimensions of the input
            num_dims = len(x.get_shape())
            
            # Separate the parameters
            mu,sigma = tf.unstack(x, num=2, axis=-1)
            
            # Add one dimension to make the right shape
            mu = tf.expand_dims(mu, -1)
            sigma = tf.expand_dims(sigma, -1)
                
            # Apply a softplus to make positive
            mu = tf.keras.activations.softplus(mu)

            sigma = tf.keras.activations.sigmoid(sigma/200000)*200000

            # Join back together again
            out_tensor = tf.concat((mu, sigma), axis=num_dims-1)

            return out_tensor
        
        # Contruction
        
        self.model_SIEFrcst = Sequential()
        self.model_SIEFrcst.add(Dense(1000, input_dim=len(self.x[0]), activation='relu')) 
        self.model_SIEFrcst.add(Dense(500, activation='relu'))
        self.model_SIEFrcst.add(Dense(500, activation='relu'))
        self.model_SIEFrcst.add(Dense(500, activation='relu'))
        self.model_SIEFrcst.add(Dense(500, activation='relu'))
        self.model_SIEFrcst.add(Dense(2, activation = 'relu'))
        self.model_SIEFrcst.add(Lambda(Gaussian_layer))

        
        self.model_SIEFrcst.compile(loss=normal_distrib_loss, optimizer= 'Adam')
        # Training
        history = self.model_SIEFrcst.fit(self.x_train, self.y_train, epochs=epochs, batch_size=128)
        

    def test(self):
        y_pred = self.model_SIEFrcst.predict(self.x_test)
        print(self.y_test[0])
        plt.hist(np.random.normal(loc = y_pred[0,0], scale = y_pred[0,1],size = 100000),bins = 50,label = f'std = {y_pred[0,1]}')
        plt.plot([int(self.y_test[0]),int(self.y_test[0])],[0,7000])
        plt.legend()
        plt.show()
        # Plot
        
        plt.scatter(y_pred[:,0],self.y_test)
        plt.errorbar(y_pred[:,0],self.y_test,xerr=y_pred[:,1], linestyle="None")

        plt.xlabel('predicted SIE [1e7 km^2]')
        plt.ylabel('True SIE [1e7 km^2]')
        plt.grid()
        plt.title('Comparison btwn forecasted and true September SIE, \n based on Neural Network model and data coming from PlaSim run.')
        
        plt.show()
""" NN = NN()
NN.form()
NN.constr(epochs = 100,save = True)
NN.test() """
#R.formating_data_SIE_Frcst()





