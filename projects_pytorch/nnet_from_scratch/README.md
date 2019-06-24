# Implementing a neural net from scratch using libtorch data structure (on MNIST)
The aim of this section is to describe step by step how to implement a 1-layer simple neural network using SGD on both mathematical and coding side. It will be trained on MNIST database for illustration. You can check `nnet_from_scratch.cpp` to look at the whole code. 

- Choice of neural network model
- Implementing our neural network


## Choice of neural network model


## Implementing our neural network

### 1- Parameters initialization 
Regarding to our model, we should initialize 4 objects : two couples of weight/bias matrices. The first one would be used to pass our samples from the input layer (size : 784) to the hidden layer (size : 64). The second one would be used to pass our sample from the hidden layer to the output layer (size : 10). The simplest and most logical idea for intializing them is to use an element-wise *standard normal distribution* for weights and to set bias to a **zero-vector** : 
    
![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W_%7B1%7D%20%5Csim%20%5Cmathcal%7BN%7D%280%2C1%29%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B64%5Ctimes784%7D)
 
 
![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W_%7B2%7D%20%5Csim%20%5Cmathcal%7BN%7D%280%2C1%29%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B10%5Ctimes64%7D)
 
 
![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20b_%7B1%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%200%5C%5C%20...%5C%5C%200%20%5Cend%7Bpmatrix%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B64%7D)
 
 
![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20b_%7B2%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%200%5C%5C%20...%5C%5C%200%20%5Cend%7Bpmatrix%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B10%7D)
 
 
Hence, a simple way to code initialization method for our nnet class is to use `torch::randn` and `torch::zeros` and to set their dtype to `torch::kFloat64` for homogeneity : 
```c++
nnet::nnet(int n_i,int n_h,int n_o, double alpha): n_input(n_i), n_hidden(n_h), n_output(n_o), learning_rate(alpha) {
	//First couple
	W1 = torch::randn({n_hidden,n_input}, torch::dtype(torch::kFloat64));
	b1 = torch::zeros({n_hidden,1}, torch::dtype(torch::kFloat64));

	//Second couple
	W2 = torch::randn({n_output,n_hidden}, torch::dtype(torch::kFloat64));
	b2 = torch::zeros({n_output,1}, torch::dtype(torch::kFloat64));
}
```

### 2- Forward propagation
Next step is to implement forward propagation, i.e. evaluation of samples by the neural network. If our neural network was a black-box represented by a function, our foward propagation would ideally look like :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20f%28X%29%20%3D%20%5Cwidehat%7BY%7D)

It follows that we can decompose our forward propagation in to main steps. 
- First step, from 784 features to 64 features, using a **sigmoid activation function** :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20Z_%7B1%7D%20%3D%20W_%7B1%7DX%20&plus;%20b_%7B1%7D)

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20G_%7B1%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-Z_%7B1%7D%7D%7D)

- Second step, from 64 features to 10 features, using a **sigmoid activation function** :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20Z_%7B2%7D%20%3D%20W_%7B2%7DG_%7B1%7D&plus;b_%7B2%7D)

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20G_%7B2%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-Z_%7B2%7D%7D%7D%20%3D%20%5Cwidehat%7BY%7D)

Here is a simple implementation of the forward method, using `torch::tensor::mm` and `torch::sigmoid` methods :
```c++
void nnet::forward(const torch::Tensor & X){
	//input_layer -> hidden_layer
	z1 = W1.mm(X) + b1;
	g1 = torch::sigmoid(z1);
	
	//hidden_layer -> output
	z2 = W2.mm(g1) + b2;
	g2 = torch::sigmoid(z2);
}
```

### 3- Cost function used
Choice of the cost function J is a key element in neural network modeling as it directly impact the first gradient calculation (in our case, dJ/dg2) as we will see in next section.
As the cost function should represent how "bad" or how "well" the learning task is converging to an estimator, there is a plenty of choice. The most common one is the [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) : 



```c++
void nnet::compute_cost(torch::Tensor & Y,int batch_size){
	J += (- (Y * torch::log(g2) + (1-Y) * torch::log(1-g2))) / double(batch_size);
}
```



### 4- Backward propagation

### 5- Parameters update

### 6- Model evaluation

