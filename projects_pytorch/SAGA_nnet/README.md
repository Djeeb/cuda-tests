# Implementing SAGA and testing it vs SGD
The aim of this page is to discuss about the implementation of a variant of SGD, called SAGA (from  [Lacoste-Julien et al., 2014](http://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives.pdf) ). It is very similar to Stochastic Average Gradient (SAG) but has better theoretical convergence rates.

- **I- [ Intuition behind SAGA ](#intuition)**

- **II- [ Implementing SAGA in libtorch ](#implementing)**
	- 1- [Gradient storage issue ](#storage)
	- 2- [Initializing gradients ](#init)
	- 3- [SAGA update ](#update)

- **III- [ Results vs SGD on MNIST ](#results)**

<a name="intuition"></a>
## I- Intuition behind SAGA

ok

<a name="implementing"></a>
## II- Implementing SAGA in libtorch 

<a name="results"></a>
## III- Results vs SGD on MNIST
