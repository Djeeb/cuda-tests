#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <string>
#include <chrono>
#include <vector>


//_______________________________________________Function to approximate

torch::Tensor euclidean_norm(const torch::Tensor & X){
	Y = torch::zeros({X.size(1)});
	for(int i=0; i < X.size(1); i++) Y[i] = X[i].norm();
	return Y;
}
