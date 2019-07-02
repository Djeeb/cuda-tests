# Implementing SAGA and testing it vs SGD
The aim of this page is to discuss about the implementation and the interest of a variant of SGD, called SAGA (from  [Lacoste-Julien et al., 2014](http://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives.pdf) ). It is very similar to Stochastic Average Gradient (SAG) but has better theoretical convergence rates according to its creators.

- **I- [ Intuition behind SAGA ](#intuition)**

- **II- [ Implementing SAGA on libtorch ](#implementing)**
	- 1- [Algorithm ](#algorithm)
	- 2- [Gradient storage issue ](#storage)
	- 3- [Initializing gradients ](#init)
	- 4- [SAGA update ](#update)

- **III- [ Results vs SGD on MNIST ](#results)**

<a name="intuition"></a>
## I- Intuition behind SAGA

Stochastic gradient descent is the most popular way to update parameters in a deep learning algorithm task. Let say we have a weight W to update with gradient of J w.r.t W, and a learning rate alpha. At the (k+1)th iteration (i.e. the (k+1) th individual used by our neural network), SGD update is given by :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k&plus;1%29%7D%20%3A%3D%20W%5E%7B%28k%29%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%5E%7B%28k%29%7D%7D)

Even if this gradient descent algorithm is great, one can wonder how much impact an individual X that diverges from the reste of the dataset can have on our neural network. Of course, thanks to a small learning rate, our estimator will converge anyway, but it might not take the shortest path to the minimum :

![image](https://www.researchgate.net/profile/Balint_Gersey/publication/326676131/figure/fig20/AS:653646912028672@1532852976155/The-red-path-represents-the-path-followed-by-stochastic-gradient-descent-using-Momentum.png)

A smoother approach to update our parameters could involve an **average** of all the gradients computed on each individual. But unlike a simple gradient descent, this algorithm would also update the gradient of a random individual at each iteration, and use it in the update equation. This is what SAGA algorithm attempts to do. 

To describe this algorithm on a simple weight W, we will denote the i th gradient (out of n) linked to the i th individual, at iteration k by :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20dW_%7Bi%7D%5E%7B%28k%29%7D%20%3D%20%5Cleft%20%28%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%7D%5Cright%20%29%20_%7Bi%7D%5E%7B%28k%29%7D)

The average of the n gradients is given by : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20d%5Cmathcal%7BW%7D%5E%7Bk%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7DdW_%7Bi%7D%5E%7Bk%7D)

At the (k+1)th iteration, denoting by i an individual chosen randomly, SAGA algorithm consists in the following update : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k&plus;1%29%7D%3A%3DW%5E%7B%28k%29%7D%20-%20%5Calpha%20%5Cleft%20%28%20dW_%7Bi%7D%5E%7B%28k&plus;1%29%7D%20-%5C%3B%20dW_%7Bi%7D%5E%7B%28k%29%7D%20&plus;%5C%3B%20d%5Cmathcal%7BW%7D%5E%7B%28k%29%7D%20%5Cright%20%29)

As we can see, it clearly involves three quantities :
	- the new gradient of individual i
	- the former gradient of individual i
	- the average gradient of the n individuals

The objective of SAGA is to reduce variance by attaching less importance on each individual variation of gradients. It is a variant of the Stochastic Average Gradient (SAG, you can check [Le Roux et al., 2014](https://arxiv.org/pdf/1309.2388.pdf) for more information.). SAG has a reduced variance compared to SAGA, but is biased. Here is the relatively similar implementation of SAG :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k&plus;1%29%7D%3A%3DW%5E%7B%28k%29%7D%20-%20%5Calpha%20%5Cleft%20%28%20%5Cfrac%7BdW_%7Bi%7D%5E%7B%28k&plus;1%29%7D%20-%5C%3B%20dW_%7Bi%7D%5E%7B%28k%29%7D%7D%7Bn%7D%20&plus;%5C%3B%20d%5Cmathcal%7BW%7D%5E%7B%28k%29%7D%20%5Cright%20%29)

Basically, each individual changes has less impact on the algorithm (this is why the variance is smaller) but it makes the assumption that the dataset is realitvely homogeneous, which is not obvious nor right in some cases.

<a name="implementing"></a>
## II- Implementing SAGA on libtorch 

While the idea behind SAGA is quite simple (see Algorithm), the implementation of the algorithm is not simple in the case of neural network. First, the complexity of the update algorithm is greater than the SGD one as we have to update an entire table to compute the average of gradients (see SAGA update).
Then, this table of gradients is quite big when it comes to neural network. (see Gradient storage issue).

<a name="algorithm"></a>
### 1- Algorithm


________________________________________

- Initialization :

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Cmathcal%7BS%7D%20%3A%3D%20%5Cleft%20%5B%20dW_%7B1%7D%2C...%2CdW_%7Bn%7D%20%5Cright%20%5D)

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%20d%5Cmathcal%7BW%7D%20%3A%3D%20%5Cleft%20%28%20%5Cmathcal%7BS%7D%5B1%5D%20&plus;...&plus;%5Cmathcal%7BS%7D%5Bn%5D%20%5Cright%20%29/n)


- Iteration :

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%201.%5C%3B%5C%3B%5C%3B%5C%3B%20choose%5C%3B%20i%5C%3B%20at%5C%3B%20random%5C%3B%20and%5C%3B%20perform%5C%3B%20forward/back-prop%5C%3B%20to%5C%3B%20compute%5C%3B%20W.grad%28%29)

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%202.%20%5C%3B%20%5C%3B%20%5C%3B%20%5C%3B%20W%20%3A%3D%20W%20-%20%5Calpha%5Cleft%20%28%20W.grad%28%29-S%5Bi%5C%2C%20%5D&plus;%20d%5Cmathcal%7BW%7D%20%5Cright%20%29)

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%203.%5C%3B%5C%3B%5C%3B%5C%3Bd%5Cmathcal%7BW%7D%20%3A%3D%20%5Cleft%20%28W.grad%28%29%20-%5Cmathcal%7BS%7D%5Bi%5C%2C%20%5D%20%5Cright%20%29/n)

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%204.%5C%3B%5C%3B%5C%3B%5C%3B%5Cmathcal%7BS%7D%5Bi%5C%2C%20%5D%20%3A%3D%20W.grad%28%29)
________________________________________


<a name="results"></a>
## III- Results vs SGD on MNIST
