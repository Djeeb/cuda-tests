# Implementing a neural net from scratch using libtorch data structure (on MNIST)
The aim of this section is to describe step by step how to implement a 1-layer simple neural network using SGD on both mathematical and coding side. It will be trained on MNIST database for illustration. You can check `nnet_from_scratch.cpp` to look at the whole code and check how `nnet` class is precisely implemented.

- Choice of neural network model
- Implementing our neural network


## Neural network model choice

Will we use the most simple neural network architecture, a 1-hidden layer fully connected neural network which could look like this : 

![1 hidden layer fully-connected neural network](data/nn_model.svg)

here are the main parameters of our model :
- input layer size : 784 
- hidden layer size : 64
- output layer size : 10
- first activation function : sigmoid
- second activation function : sigmoid
- cost function used : mean squared error
- update algorithm used : stochastic gradient descent

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

It follows that we can decompose our forward propagation in two main steps. 
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
Choice of the cost function J is a key element in neural network modeling as it directly impact the first gradient calculation (in our case, dJ/dg2) as we will see in next section. As the cost function should represent how "bad" or how "well" the learning task is converging to an estimator, there is a plenty of choice. 

#### - MSE
The most common one is the [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error). Indexing columns of matrices by i and number of samples by n, we have this equation for MSE :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20J%5Cleft%20%28%20%5Cwidehat%7BY%7D%20%5Cright%20%29%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum%20%28%5Cwidehat%7BY%7D_%7Bi%7D-Y_%7Bi%7D%29%5E%7BT%7D%28%5Cwidehat%7BY%7D_%7Bi%7D-Y_%7Bi%7D%29)

Implementing cost computing is not necessarily for the neural network in itself but it is a good way to see how well your model is training during the learning phase. We use the methods `sum()` that sums all matrix coefficients to output a single coefficient tensor, and `item<double>()` to convert the coefficient to a `double`. Also note that we use the batch size to scale the cost and harmonize the results :

```c++
void nnet::compute_cost(torch::Tensor & Y){
	J += (g2-Y)*(g2-Y).sum().item<double>() / double(batch_size);
}
```

#### - Cross entropy loss

Another option is given by the [Cross entropy loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a), a more refined loss function that has the advantage to strongly penalize the model if the estimation differs from the actual answer : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20J%5Cleft%20%28%20%5Cwidehat%7BY%7D%20%5Cright%20%29%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum%20-%5C%3A%5Cleft%20%28%20%5C%3A%20%5Clog%28%5Cwidehat%7BY%7D_%7Bi%7D%29%5E%7BT%7DY_%7Bi%7D%5C%3A%20&plus;%20%5C%3A%20%5Clog%281-%5Cwidehat%7BY%7D_%7Bi%7D%29%5E%7BT%7D%5Cleft%20%281-Y_%7Bi%7D%20%5Cright%20%29%20%5Cright%20%29)

Again, the implementation is quite simple with the `torch::log` function :

```c++
void nnet::compute_cost(torch::Tensor & Y){
	J += (- (Y * torch::log(g2) + (1-Y) * torch::log(1-g2))).sum().item<double>() / double(batch_size);
}
```

To end this section, we use an auxiliary function to both display and reset the cost :

```c++
double nnet::reset_cost() { 
	double x = J;
	J = 0.;
	return x;}
}
```

### 4- Backward propagation
This is the trickiest part of neural network implementation as it requires a bit of calculus and linear algebra skills to compute the gradients. As our goal is to slightly change weights and biases with their slope regarding the cost function J, we have to use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) and calculate intermediary values to compute dJ w.r.t W1, W2, b1, and b2. Here is the step-by-step mathematical path **for MSE** :







### 5- Parameters update

### 6- Model evaluation

