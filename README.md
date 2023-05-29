# LPHYS2268 predicting next september Sea Ice Extent (SIE)

### Why this repository?

GitHub made for project of the course LPHYS2268 - Forecast, prediction and projection in climate science, teach by Mr. Massonnet at UCLouvain.

The goal of this project is to:
- Create a model which can predict futur september SIE based only on data available in the beginning of June - i.e. data until may.
- Test this model on the previous year data and evaluate the skill of the forecast
- Forecast 2023 september SIE.

### How does it work?

We use a Neural Network (NN) model. These model work in layer, there is tree type of layer. Input layer (only one), the hidden layers (we can use several of them) and the output layer (only one). We "feed" the NN by the input layer, with the data we think relevant to make the prediction (called the predictors) and these data will propagate through the network. In the end the NN "prediction" comes out by the output layer.

![alt text](https://github.com/AugustinLambotte/Figure/main/1_nevKs6306VMnE3aP-C0zbg.jpg?raw=true)