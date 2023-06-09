import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Lambda
import xarray as xr
import tensorflow_probability as tfp

############# - Extraction of the data - ###############
class NN:
    def __init__(self,is_siv,clim_time,interp_deg,file_siv = 'Training_data/SIV_mensual_plsm.txt', file_sie = 'Training_data/SIE_mensual_plsm.txt'):
        """
            This class "NN" create a Neural Network model to predict futur Sea Ice Extend (SIE) 
            based on previous SIE and SIV (Sea ice volume) data.
            
            Parameters:
            ----------
            clim_time | int: is the range of time over which we want to compute the trend line.
            interp_deg | int: is the degree of interpolation if the trend line. Should be keep to 1 in practice.
            is_siv | bool: True if we want to use the SIV data False if not

        """
        def data_arange(SIE_data,SIV_data, test_size = 0.005):
            """
                Format the data in order to be used by the network.

                Parameters:
                ----------
                SIE_data | list: list of array. Each list element is a N_year x 12 array. Each line for a year (year have to be consecutive)
                                 and each column for a month. filled with monthl SIE in km^2
                SIV_data | list: list of array. Each list element is a N_year x 12 array. Each line for a year (year have to be consecutive)
                                 and each column for a month. filled with monthl SIV in km^2
                tes_size | float: btwn 0 and 1. Define the proportion of the data which will be separate from the training data
                                  in order to have a set of test if we need.

                Return:
                -------
                self.x | array: it's the inputs data. self.x_train for the train part and self.x_test for the test part
                                it is made of an interpolation of the climatological trend of septembre SIE and SIV. And
                                the SIE and SIV value from Januray to may of the current year.
                slef.y | array: is the output data, the sept_sie.   
            """
            
            x = np.array([])
            for SIE,SIV in zip(SIE_data,SIV_data):
                
                clim_time = self.clim_time
                climatology_siv = [] # Climatological trend sept sie mean of the "clim_time" last years of the considered year.
                climatology_sie = [] # Climatological trend sept siv mean of the "clim_time" last years of the considered year.
                
                summer_sie = SIE[clim_time:,:5] # SIE of the last 5 month (jan -> may)
                summer_siv = SIV[clim_time:,:5] # SIV of the last 5 month (jan -> may)

                for year in range(clim_time,len(SIE)):
                    deg = interp_deg
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
                    current = np.concatenate((climatology_sie,
                                climatology_siv,
                                summer_sie,
                                summer_siv,),axis = 1) 
                else: 
                    current = np.concatenate((climatology_sie,
                                summer_sie), axis = 1)

                if len(x) == 0:
                    x = current
                else:
                    x = np.concatenate((x,current))

            y = np.array([]) 
            for SIE in SIE_data:
                current = SIE[clim_time:,8:9]
                if len(y) == 0:
                    y = current
                else:
                    y = np.concatenate((y,current))
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size = test_size)

        
        self.clim_time = clim_time
        self.is_siv = is_siv

        #---------
        # Extraction of training datas
        #---------

        SIE_mensual_plsm = np.genfromtxt(file_sie,delimiter=' ')
        SIV_mensual_plsm = np.genfromtxt(file_siv, delimiter =' ')

        SIE_mensual_CESM2 = np.genfromtxt('Training_data/CMIP/SIE_CESM2.txt', delimiter = ' ')
        SIV_mensual_CESM2 = np.genfromtxt('Training_data/CMIP/SIV_CESM2.txt', delimiter = ' ')
        SIE_mensual_CESM2 *= 1e6 # Passing in [km^2]
        
        SIE_data = [SIE_mensual_plsm,SIE_mensual_CESM2]
        SIV_data = [SIV_mensual_plsm,SIV_mensual_CESM2]      
        
        data_arange(SIE_data,SIV_data)

        print('######################')
        print('Creation of Neural Network')
        print('#####################')
        print(f"Number of training year = {len(self.x_train)}")
        print(f'input size = {len(self.x_train[0])}')
        print('------------------------')
    
    def constr(self, epochs = 60):
        """
            Construct the neural network and train him to predict sept_SIE

            Parameters:
            -----------
            epochs | int: number of epochs for NN training.
        """
        def normal_distrib_loss(y_true, y_pred):
            """
            Normal distribution loss function.
            Assumes tensorflow backend.
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
            mu and sigma from a Dense(2) output.
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
        
        #------------
        # Contruction
        #------------
        N_neuron= 30
        N_layer = 10


        print("---------------")
        print("Number of hidden layer = ",N_layer)
        print("Number of neuron per layer = ",N_neuron)
        print('---------------')
        
        # Initialization of the model
        self.model_SIEFrcst = Sequential()


        # Layers
        self.model_SIEFrcst.add(tf.keras.layers.BatchNormalization())
        self.model_SIEFrcst.add(Dense(N_neuron, input_dim=len(self.x_train[0]), activation='relu'))

        for _ in range(N_layer):
            self.model_SIEFrcst.add(Dense(N_neuron, activation='relu'))
        
        self.model_SIEFrcst.add(Dense(2, activation = 'relu'))
        self.model_SIEFrcst.add(Lambda(Gaussian_layer))

        
        self.model_SIEFrcst.compile(loss=normal_distrib_loss, optimizer= 'Adam')
        
        #----------
        # Training
        #----------
        history = self.model_SIEFrcst.fit(self.x_train, self.y_train, epochs=epochs, batch_size=128)        

    def test(self):
        """
            Test the prediction skills of the neural network on self.x_test: data woh hasn't been used for training.
        """
        y_pred = self.model_SIEFrcst.predict(self.x_test)
        
        #------
        # Plot
        #------
        plt.scatter(y_pred[:,0],self.y_test)
        plt.errorbar(y_pred[:,0],self.y_test,xerr=y_pred[:,1], linestyle="None")
        plt.xlabel('predicted SIE [1e7 km^2]')
        plt.ylabel('True SIE [1e7 km^2]')
        plt.grid()
        plt.title('Comparison btwn forecasted and true September SIE, \n based on Neural Network model and data coming from PlaSim run.')
        
        plt.show()







