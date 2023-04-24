import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

############# - Extraction of the data - ###############
class NN():
    def __init__(self, file = 'Data/SIE_mensual_plsm.txt'):
        """
            comment class
        """
        def data_arange(SIE):
            """
                x is an array with the mensual SIE from September of the previous year to may of the current (both included). This will be the input data.
                y is an array of SIE september data, this will be used to compare with the data output.
            """
            sept_to_dec_last_year = SIE[:-1,8:]
            jan_to_may_current_year = SIE[1:,:5]

            x = np.concatenate((sept_to_dec_last_year,jan_to_may_current_year),axis = 1)
            y = SIE[1:,8:9]
            return x,y
        SIE_mensual = np.genfromtxt(file,delimiter=' ')
        self.x,self.y = data_arange(SIE_mensual)
        print(SIE_mensual[0])
        print(SIE_mensual[1])
        print(self.x[0])
        print(self.y[0])

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
    
    def construct_LPY(self, epochs = 150,save = False):
        """
            Construct the neural network and train him to predict LPY events.
        """
        # Contruction
        self.model_LPY = Sequential()
        self.model_LPY.add(Dense(16, input_dim=9, activation='relu'))
        self.model_LPY.add(Dense(12, activation='relu'))
        self.model_LPY.add(Dense(2, activation='softmax'))
        self.model_LPY.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Training
        history = self.model_LPY.fit(self.x_train, self.y_train, epochs=epochs, batch_size=64)
        if save:
            self.model_LPY.save('NN_model')

    def testing_LPY(self):
        y_pred = self.model_LPY.predict(self.x_test)
        #Converting predictions to label ([0.8,0.2] -> 0 (it is read as [1,0]))
        pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        #Converting one hot encoded test label to label
        test = [np.argmax(self.y_test[i]) for i in range(len(self.y_test))]

        #Compute the accuracy between pred and test
        a = accuracy_score(pred,test)
        print('Accuracy is:', a*100)
        
    def construct_SIEfrcst(self, epochs = 30,save = False):
        """
            Construct the neural network and train him to predict sept_SIE
        """
        # Contruction
        self.model_SIEFrcst = Sequential()
        self.model_SIEFrcst.add(Dense(16, input_dim=9, activation='relu'))
        self.model_SIEFrcst.add(Dense(12, activation='relu'))
        self.model_SIEFrcst.add(Dense(len(self.y_train[0]), activation='softmax'))
        self.model_SIEFrcst.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Training
        print(np.shape(self.x_train))
        print(np.shape(self.y_train))
        history = self.model_SIEFrcst.fit(self.x_train, self.y_train, epochs=epochs, batch_size=64)
        if save:
            model_name = "NN_mod"
            self.model_SIEFrcst.save(model_name)
        
    def formating_data_SIEFrcst(self,sie_range = 0.5*1e6, test_size = 0.1):
        """
            Turns x and y in the good format for a SIE sept extend forecast.
        """
        self.sie_range = sie_range
        # The smallest data are always over 4*1e6 km^2 so we put the set 'to the ground'.
        y = self.y - 4*1e6 
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
        x = sc.fit_transform(self.x)
        # Splitting our data set in training and testing parts
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,ohe_y,test_size = test_size)
    
    def test_SIEFrcst(self):
        y_pred = self.model_SIEFrcst.predict(self.x_test)
        print(y_pred)
        print(np.shape(y_pred))

        # Turning the state of the preidction layer in a value in km^2, stored in prediction.
        prediction = np.zeros(len(y_pred))
        for sample_pred in range(len(y_pred)):
            predicted_val = 0
            for neuron in range(len(y_pred[0])):
                predicted_val += self.sie_range * y_pred[sample_pred][neuron] * neuron
            prediction[sample_pred] = predicted_val

        print(prediction)
        
        # Recovery the test SIE in km from self.y_test which is ohe_encoded.
        test = np.zeros(len(y_pred))
        for sample_test in range(len(y_pred)):
            test_val = 0
            for neuron in range(len(y_pred[0])):
                test_val += self.sie_range * self.y_test[sample_test][neuron] * neuron
            test[sample_test] = test_val
        
        # The data has been reduce by 4*1e6 km^2 in self.formating_data_SIEFrcst() we undo,
        test += 4*1e6
        prediction += 4*1e6

        # Plot
        plt.scatter(prediction,test)
        plt.xlabel('predicted SIE [1e6 km^2]')
        plt.ylabel('True SIE [1e6 km^2]')
        plt.grid()
        plt.title('Comparison btwn forecasted and true September SIE, \n based on Neural Network model and data coming from PlaSim run.')
        plt.show()

R = NN()
#R.formating_data_SIEFrcst()
#R.construct_SIEfrcst()
#R.test_SIEFrcst()
#R.formating_data_SIE_Frcst()





