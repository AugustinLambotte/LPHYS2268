# LPHYS2268 predicting next september Sea Ice Extent (SIE)

## Why this repository?

GitHub made for project of the course LPHYS2268 - Forecast, prediction and projection in climate science, teach by Mr. Massonnet at UCLouvain.

The goal of this project is to:
- Create a model which can predict futur september SIE based only on data available in the beginning of June - i.e. data until may.
- Test this model on the previous year data and evaluate the skill of the forecast
- Forecast 2023 september SIE.

## The model?

We use a Neural Network (NN) model. These model work in layer, there is tree type of layer. Input layer (only one), the hidden layers (we can use several of them) and the output layer (only one). We "feed" the NN by the input layer, with the data we think relevant to make the prediction (called the predictors) and these data will propagate through the network. In the end the NN "prediction" comes out by the output layer.

![alt text](https://github.com/AugustinLambotte/Figure/blob/main/1_nevKs6306VMnE3aP-C0zbg.jpg?raw=true)

### Propagation, how does it works?
There is two types of elements in the network: neurons and connection. The neurons are carrying the information and the connection decide how the information is transmitted between the neurons.
At the begging only the input layer is filled with the predictors. In our case they are previous value of SIE and SIV (Sea ice Volume). Then the neurons of the input layer propagate their information in the next layer, the first hidden layer. Each neuron is sending its value at all the neuron of the next one (this is what is called a dense network) but the value are weighted by the connection. Indeed, each connection has a weight, between 0 and 1, and this weight descide how much this neurons should send his information in this neuron. The information by this way the information are propagating through the network until the output layer - where we take the "predicted value".

Therefore, the prediction depends on the weight of each connections. This NN is fully determined by its topology - number of layer and neuron by layer -  and the weight assigned on each connection. Note that deeper NN (more hidden layer) allows higher non linearity regression but expose the network to problem such that vanishing/exploding gradient.

How does the NN find good prediction?
As we said, the only "cursor" the NN has to change the output - given a fix input - are its connections' weight. But how does he determine these weight? This is the purpose of the training phase!

### Training phase.

At the begging, the weight are initialized randomly between 0 and 1. The NN has to train to select the more accurate value of each weight.
To do so we use a training data set. This data set is made of data where we have the predictors but also the real value that the network has to predict. We feed the input layer with the predictor and he made a prediction - totaly randomly in the beginning - we compare this prediction with the real value and the network perform little corrections over all its weight to be more accurate given the error he maid.

This correction is not a magic tricks. This is called the backward propagation, the error is computed and a gradiant is calculated to determine how much each weight connecting to the previous layer are resposible of the error. When it is determine it pass to the layer before and re calculate the gradient and so forth until the input layer. Each connection weight, knowing how much he influenced the error, can correct himself a bit. The error computation is perform using a "loss function" that the network will try to minimize. In some case it can be simply a mean squared error or things like this but in our case it will be a bit more tricky, we talk about in the next section.

The training phase will repeat this scheme (a foreward propagation, an evaluation of the error and then a backward propagation to correct the weight) a lot of time over all the data set. Note that the data set has to be big enough to expose the NN to a maximum of different configuration.

### Prediction

After the network is train, i.e. all the weight are fixed in the best possible value, we can use it over real data to predict what we want by a simple foreward propagation using the predictor we want. In our case, in fact we doesn't only want a prediction of the SIE. We want a distribution of probabilty. Therefore we don't have only one output neuron which gives the guess of the next SIE but two, which are giving the mean and the standard deviation of the predicted SIE distribution. In order to do this, the loss function in the training part can't only compute a simple error. The loss function we choosed is the log likelihood function of normal distribution which has the mean and the std given by the output layer. With this the NN will predict mean and std in such a way that it increases the proability to observe the correct answer knowing we have this parametrization of normal distribution.

## How did we use it?

### The predictors
We use 12 predictors - i.e. our input layer has 12 neurons. The SIV and SIE of the last 5 month (jan, feb, mar, apr, may) and two linear interpolations, one for the SIE and one for the SIV. These are the next sept SIE and SIV value given by the trend line interpolation of last 10 years. Our SIE data comes from the satelitary record https://osisaf-hl.met.no/v2p1-sea-ice-index and the SIV data from http://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/.

### The training data set
Our training data set is made of 6419 years of simulated data. 1184 from CMIP and 5235 from PLASIM control run.




## How to use it?
All the scripts are running under python 3.8, the following package are required:
- tensorflow
- tensorflow-probability
- xarray
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- netcdf4

The script ML_frcst.py can be run directly. NN_model.py is used by ML_frcst.py to create the Neural network (creation and training over the training data set). All the structure is settle by two class, one in ML_frcst.py for the prediction and oone in NN_model.py for the NN creation, all classes and function are documented. 
Data are the real world data used for the prediction and Training_data are the data used to train the model.