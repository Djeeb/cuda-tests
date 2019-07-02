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

Stochastic gradient descent is the most popular way to update parameters in a deep learning algorithm task. Let say we have a weight W to update with gradient of J w.r.t W, and a learning rate alpha. At the k+1 th iteration, SGD update is given by :

#### - SGD : ![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k&plus;1%29%7D%20%3A%3D%20W%5E%7B%28k%29%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%5E%7B%28k%29%7D%7D)



<a name="implementing"></a>
## II- Implementing SAGA in libtorch 

<a name="results"></a>
## III- Results vs SGD on MNIST
