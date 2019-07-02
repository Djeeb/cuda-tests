# Implementing SAGA and testing it vs SGD
The aim of this page is to discuss about the implementation and the interest of a variant of SGD, called SAGA (from  [Lacoste-Julien et al., 2014](http://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives.pdf) ). It is very similar to Stochastic Average Gradient (SAG) but has better theoretical convergence rates according to its creators.

- **I- [ Intuition behind SAGA ](#intuition)**

- **II- [ Implementing SAGA in libtorch ](#implementing)**
	- 1- [Gradient storage issue ](#storage)
	- 2- [Initializing gradients ](#init)
	- 3- [SAGA update ](#update)

- **III- [ Results vs SGD on MNIST ](#results)**

<a name="intuition"></a>
## I- Intuition behind SAGA

Stochastic gradient descent is the most popular way to update parameters in a deep learning algorithm task. Let say we have a weight W to update with gradient of J w.r.t W, and a learning rate alpha. At the (k+1)th iteration (i.e. the (k+1) th individual used by our neural network), SGD update is given by :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k&plus;1%29%7D%20%3A%3D%20W%5E%7B%28k%29%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%5E%7B%28k%29%7D%7D)

Even if this gradient descent algorithm is great, one can wonder how much impact an individual X that diverges from the reste of the dataset can have on our neural network. Of course, thanks to a small learning rate, our estimator will converge anyway, but it might not take the shortest path to the minimum :

![image](https://www.researchgate.net/profile/Balint_Gersey/publication/326676131/figure/fig20/AS:653646912028672@1532852976155/The-red-path-represents-the-path-followed-by-stochastic-gradient-descent-using-Momentum.png)

A smoother approach to update our parameters could involve an **average** of all the gradients computed on each individual. But unlike a simple gradient descent, this algorithm would also update the gradient of a random individual at each path, and use it in the update equation. This is what SAGA algorithm attempts to do. 

To describe this algorithm on a simple weight W, we will denote the i th gradient (out of n) linked to the i th individual, at iteration k by :

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20dW_%7Bi%7D%5E%7B%28k%29%7D%20%3D%20%5Cleft%20%28%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%7D%5Cright%20%29%20_%7Bi%7D%5E%7B%28k%29%7D)

The average of the n gradients is given by : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20d%5Cmathcal%7BW%7D%5E%7Bk%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7DdW_%7Bi%7D%5E%7Bk%7D)

At the (k+1)th iteration, denoting by i an individual chosen randomly, SAGA algorithm consists in the following update : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k&plus;1%29%7D%3A%3DW%5E%7B%28k%29%7D%20-%20%5Calpha%20%5Cleft%20%28%20dW_%7Bi%7D%5E%7B%28k&plus;1%29%7D%20-%5C%3B%20dW_%7Bi%7D%5E%7B%28k%29%7D%20&plus;%5C%3B%20d%5Cmathcal%7BW%7D%5E%7B%28k%29%7D%20%5Cright%20%29)

<a name="implementing"></a>
## II- Implementing SAGA in libtorch 

<a name="results"></a>
## III- Results vs SGD on MNIST
