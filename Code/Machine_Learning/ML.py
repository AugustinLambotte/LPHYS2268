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
class NN():
    def __init__(self,file_siv = 'Data/SIV_mensual_90-5100_plsm.txt', file_sie = 'Data/SIE_mensual_90-5100_plsm.txt'):
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
        def data_arange(SIE_plsm,SIV_plsm, SIE_cmip, SIV_cmip):
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

            sept_to_dec_last_year_sie_plsm = SIE_plsm[:-1,month_range_SIE[0]-1:]
            jan_to_may_current_year_sie_plsm = SIE_plsm[1:,:month_range_SIE[1]]

            sept_to_dec_last_year_siv_plsm = SIV_plsm[:-1,month_range_SIV[0]-1:]
            jan_to_may_current_year_siv_plsm = SIV_plsm[1:,:month_range_SIV[1]]

            plsm_input = np.concatenate((sept_to_dec_last_year_sie_plsm,
                                jan_to_may_current_year_sie_plsm,
                                sept_to_dec_last_year_siv_plsm, 
                                jan_to_may_current_year_siv_plsm),axis = 1)
            
            sept_to_dec_last_year_sie_cmip = SIE_cmip[:-1,month_range_SIE[0]-1:]
            jan_to_may_current_year_sie_cmip = SIE_cmip[1:,:month_range_SIE[1]]

            sept_to_dec_last_year_siv_cmip = SIV_cmip[:-1,month_range_SIV[0]-1:]
            jan_to_may_current_year_siv_cmip = SIV_cmip[1:,:month_range_SIV[1]]

            cmip_input = np.concatenate((sept_to_dec_last_year_sie_cmip,
                                jan_to_may_current_year_sie_cmip,
                                sept_to_dec_last_year_siv_cmip, 
                                jan_to_may_current_year_siv_cmip),axis = 1)
            

            x = np.concatenate((plsm_input, cmip_input),axis = 0)

            #x = np.concatenate((sept_to_dec_last_year_sie,jan_to_may_current_year_sie),axis = 1)
            sept_plsm = SIE_plsm[1:,8:9]
            sept_cmip = SIE_cmip[1:,8:9]
            y = np.concatenate((sept_plsm, sept_cmip), axis = 0)

            return x,y
        SIE_mensual_plsm = np.genfromtxt(file_sie,delimiter=' ')
        SIV_mensual_plsm = np.genfromtxt(file_siv, delimiter =' ')

        SIE_mensual_CMIP = np.genfromtxt('Data/CMIP/SIE_CMIP.txt', delimiter = ' ')
        SIV_mensual_CMIP = np.genfromtxt('Data/CMIP/SIV_CMIP.txt', delimiter = ' ')

        SIE_mensual_CMIP *= 1e6 # Passing from [1e6km^2] to [km^2]
    
        self.x,self.y = data_arange(SIE_mensual_plsm,SIV_mensual_plsm,SIE_mensual_CMIP,SIV_mensual_CMIP)

    ######## - LPY - #######
     
    def formating_data_LPY(self, test_size = 0.1):
        """
            Format data to be in the right shape for LPY events prediction.
            test_size can be adjust: it is the proportion of the dataset which will not
            be used for training but for verification.
        """
        x = self.x
        y = self.y
        # Filling y with 0 and 1. 0 if SIE will be smaller than previous year, 1 if it will bigger.
        y = [int(np.modf(y[year]/x[year,0])[1]) for year in range(len(y))]
        y = np.array(y)
        # Turning the output data in the form [.,.]: [1,0] if it was equals to 0 (smaller) and [0,1] if it was 1 (bigger)
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y.reshape(-1, 1)).toarray()
        # Normalizaton of input datas
        sc = StandardScaler()
        x = sc.fit_transform(x)
        # Splitting our data set in training and testing parts
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size = test_size)
    
    def construct_LPY(self, epochs = 110,save = False):
        """
            Construct the neural network and train him to predict LPY events.
        """
        # Contruction
        self.model_LPY = Sequential()
        self.model_LPY.add(Dense(16, input_dim=len(self.x[0]), activation='relu'))
        self.model_LPY.add(Dense(12, activation='relu'))
        self.model_LPY.add(Dense(2, activation='softmax'))
        self.model_LPY.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Training
        history = self.model_LPY.fit(self.x_train, self.y_train, epochs=epochs, batch_size=64)
        if save:
            self.model_LPY.save('LPY_modd')

    def test_LPY(self):
        y_pred = self.model_LPY.predict(self.x_test)
        #Converting predictions to label ([0.8,0.2] -> 0 (it is read as [1,0]))
        pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        #Converting one hot encoded test label to label
        test = [np.argmax(self.y_test[i]) for i in range(len(self.y_test))]

        #Compute the accuracy between pred and test
        a = accuracy_score(pred,test)
        print('Accuracy is:', a*100)

    ######## - SIE - #######

    def formating_data_SIEfrcst(self,sie_range = 0.1 *1e6, test_size = 0.01):
        """
            Turns x and y in the good format for a SIE sept extend forecast.
        """
        #self.out_layer_size = output_size
        # The smallest data are always over 4*1e6 km^2 so we put the set 'to the ground'.
        y = self.y - 4*1e6 
        # Subdivision of SEPT_SIE in range spanning sie_range each.
        #self.sie_range = np.max(y)/output_size
        self.sie_range = sie_range
        y = np.array([np.modf(y[year]/(self.sie_range)) for year in range(len(y))])
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
        x = sc.fit_transform(self.x)
        # Splitting our data set in training and testing parts
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,ohe_y,test_size = test_size)

    def construct_SIEfrcst(self, epochs = 60,save = False):
        """
            Construct the neural network and train him to predict sept_SIE
        """
        # Contruction
        self.model_SIEFrcst = Sequential()
        self.model_SIEFrcst.add(Dense(18, input_dim=len(self.x[0]), activation='relu'))    
        self.model_SIEFrcst.add(Dense(8, activation='relu'))
        self.model_SIEFrcst.add(Dense(len(self.y_train[0]), activation='softmax'))
        self.model_SIEFrcst.compile(loss='log_cosh', optimizer='adam', metrics = ['accuracy'])
        # Training
        history = self.model_SIEFrcst.fit(self.x_train, self.y_train, epochs=epochs, batch_size=128)
        if save:
            model_name = "NN2"
            self.model_SIEFrcst.save(model_name)
        
    def test_SIEfrcst(self):
        y_pred = self.model_SIEFrcst.predict(self.x_test)
        

        # Turning the state of the prediction layer in a value in km^2, stored in prediction.
        prediction = np.zeros(len(y_pred))
        for sample_pred in range(len(y_pred)):
            predicted_val = 0
            for neuron in range(len(y_pred[0])):
                predicted_val += self.sie_range * y_pred[sample_pred][neuron] * (neuron+1)
            prediction[sample_pred] = predicted_val

        
        
        # Recovery the test SIE in km from self.y_test which is ohe_encoded.
        test = np.zeros(len(y_pred))
        for sample_test in range(len(y_pred)):
            test_val = 0
            for neuron in range(len(y_pred[0])):
                test_val += self.sie_range * self.y_test[sample_test][neuron] * (neuron+1)
            test[sample_test] = test_val
        
        # The data has been reduce by 4*1e6 km^2 in self.formating_data_SIEFrcst() we undo,
        test += 4*1e6
        prediction += 4*1e6

        # Plot
        plt.scatter(prediction,test)
        plt.xlabel('predicted SIE [1e7 km^2]')
        plt.ylabel('True SIE [1e7 km^2]')
        plt.grid()
        plt.title('Comparison btwn forecasted and true September SIE, \n based on Neural Network model and data coming from PlaSim run.')
        plt.show()

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
            import numpy as np
            import tensorflow as tf
            from tensorflow import keras
            # Get the number of dimensions of the input
            num_dims = len(x.get_shape())
            
            # Separate the parameters
            mu,sigma = tf.unstack(x, num=2, axis=-1)
            
            # Add one dimension to make the right shape
            mu = tf.expand_dims(mu, -1)
            sigma = tf.expand_dims(sigma, -1)
                
            # Apply a softplus to make positive
            mu = tf.keras.activations.softplus(mu)

            sigma = tf.keras.activations.sigmoid(sigma/250000)*250000

            # Join back together again
            out_tensor = tf.concat((mu, sigma), axis=num_dims-1)

            return out_tensor
        
        # Contruction
        
        self.model_SIEFrcst = Sequential()
        self.model_SIEFrcst.add(Dense(500, input_dim=len(self.x[0]), activation='relu')) 
        self.model_SIEFrcst.add(Dense(300, activation='relu'))
        self.model_SIEFrcst.add(Dense(100, activation='relu'))
        self.model_SIEFrcst.add(Dense(2, activation = 'relu'))
        self.model_SIEFrcst.add(Lambda(Gaussian_layer))

        
        self.model_SIEFrcst.compile(loss=normal_distrib_loss, optimizer= 'Adam')
        # Training
        history = self.model_SIEFrcst.fit(self.x_train, self.y_train, epochs=epochs, batch_size=128)
        if save:
            model_name = "NN2"
            self.model_SIEFrcst.save(model_name)

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
NN = NN()
NN.form()
NN.constr(epochs = 100,save = True)
NN.test()
#R.formating_data_SIE_Frcst()






