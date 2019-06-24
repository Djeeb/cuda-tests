# Implementing a neural net from scratch using libtorch data structure (on MNIST)
The aim of this section is to precisely describe how to implement a 1-layer simple neural network using SGD on both mathematical and coding side. It will be trained on MNIST database for illustration.

**Table of contents**
- Neural network steps
    - structure used
    - computing backward propagation
- Coding side

## Choice of neural network model

## Implementing Neural network steps

### 1- Parameters initialization 
Regarding to our model, we should initialize 4 objects : two couples of weight/bias matrices. The first one would be used to pass our samples from the input layer (size : 784) to the hidden layer (size : 64). The second one would be used to pass our sample from the hidden layer to the output layer (size : 10). The simplest and most logical idea for intializing them is to use an element-wise *standard normal distribution* for weights and to set bias to a **zero-vector** : 
    
![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B200%7D%20W_%7B1%7D%20%5Csim%20%5Cmathcal%7BN%7D%280%2C1%29%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B64%5Ctimes784%7D)


![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B200%7D%20W_%7B2%7D%20%5Csim%20%5Cmathcal%7BN%7D%280%2C1%29%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B10%5Ctimes64%7D)


![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20b_%7B1%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%200%5C%5C%20...%5C%5C%200%20%5Cend%7Bpmatrix%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B64%7D)


![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20b_%7B2%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%200%5C%5C%20...%5C%5C%200%20%5Cend%7Bpmatrix%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B10%7D)

Hence, a simple way to code initialization method for our neuralnet class is to use '<torch::tensor>' objects : 


2- Forward propagation

3- Cost function used

4- Backward propagation

5- Parameters update

6- Model evaluation

## Structure used
blabla

## Computing backward propagation
