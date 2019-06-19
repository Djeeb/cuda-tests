#include <torch/torch.h>
#include <iostream>
using namespace std;



int main(){

//NNet from scratch

//Paramètres
	int n = 100; //nombre d'échantillons
	int d = 10; // nombre de variables
	float data[n*d];
	for(int i=0;i<n*d;i++) data[i] = i;
	
	//Input
	torch::Tensor X = torch::randn({d,n});
	
	//Couche 1
	torch::Tensor W1 = torch::randn({10,d});
	torch::Tensor B1 = torch::ones({10,1});

	//Couche 2
	torch::Tensor W2 = torch::randn({1,10});
	torch::Tensor B2 = torch::ones({1,1});
	
//Stochastic gradient descent
auto g1 = W1.mm(X.slice(1,0,1)) + B1;
auto g2 = W2.mm(g2) + B2;
auto loss = g2*g2;

auto d_W2 = 
		//Forward propagation
		g1 = W1.mm(X.slice(1,0,1)) + B1;
		g2 = W2.mm(g2) + B2;
		loss = g2*g2;
		
		//Backward propagation
		
}
