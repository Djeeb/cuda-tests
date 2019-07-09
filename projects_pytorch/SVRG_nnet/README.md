# Implementing SVRG and testing it vs SGD
Following our exploration of variants of SGD algorithm (check our previous analysis of [SAGA](https://github.com/Djeeb/stage_DL/tree/master/projects_pytorch/SAGA_nnet) algorithm),
we decided to dig into the promising **Stochastic Variance Reduced Gradient** algorithm from [R. Johnson et al., 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf).
It is directly inspired from SDCA and SAG, but unlike SAG, it doesn't involve full gradients storage. Researchers say it is actually easily applicable for neural network learning.
We want to compare convergence rate to SGD algorithm on a neural network structure. 
You can check the whole implementation in `SVRG_nnet.hpp`.

- **I- [ Intuition behind SVRG ](#intuition)**

<a name="intuition"></a>
## I- Intuition behind SVRG

As usual, we denote the gradient of J w.r.t a parameter W at iteration (k) on the ith sample by : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20dW_%7Bi%7D%5E%7B%28k%29%7D%3A%3D%20%5Cleft%20%28%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%7D%20%5Cright%20%29_%7Bi%7D%5E%7B%28k%29%7D)

At iteration (k), SVRG consists in picking i randomly among the n samples and update the parameter W as follows : 

![equation](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%5E%7B%28k%29%7D%20%3D%20W%5E%7B%28k-1%29%7D%5C%3B%20-%20%5C%3B%20%5Calpha%20%5Cleft%20%28%20dW_%7Bi%7D%5E%7B%28k-1%29%7D-d%5Cwidetilde%7BW%7D_%7Bi%7D%20%5C%3B%20&plus;%20%5C%3B%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7Dd%5Cwidetilde%7BW%7D_%7Bj%7D%20%5Cright%20%29) 

where W tilde denotes a *snapshot* of the parameter taken at a certain iteration index. Typically, this W is stored every 2n or 5n. 
