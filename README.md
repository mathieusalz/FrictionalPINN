This code is adapted from the code for the paper Physics-Informed Neural Networks for fault slip monitoring: simulation, frictional parameter estimation, and prediction on slow slip events in a spring-slider system by authors Rikuto Fukushima, Masayuki Kano, and Kazuro Hirahara.

The original paper is available here: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JB027384
The original code is available here: https://zenodo.org/records/8405977

My contribution to the code has been:
- added different methods for normalizing the input to the network
- added different methods for nondimensionalizing the ODEs used in the loss function
- added functions to create training animations 
- adapted the plotting functions to handle the different normalizations
- cleaned up print statements during training