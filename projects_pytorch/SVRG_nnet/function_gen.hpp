#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <string>
#include <chrono>
#include <vector>


//_______________________________________________Function to approximate


//Euclidean norm
torch::Tensor euclidean_norm(const torch::Tensor & X){
	torch::Tensor Y = torch::zeros({X.size(0)});
	for(int i=0; i < X.size(0); i++) Y[i] = X[i].norm();
	return Y;
}
