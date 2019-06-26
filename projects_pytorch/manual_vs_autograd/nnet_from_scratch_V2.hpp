#include <torch/torch.h>
#include <iostream>
#include <chrono>
using namespace std;

class nnet : public torch::nn::Module {
	public:
		torch::nn::Linear z1{nullptr}, z2{nullptr};
		
		nnet(int,int,int);
		torch::Tensor forward(torch::Tensor &);

};

//__________________________________________________________Constructeur
nnet::nnet(int n_input,int n_hidden,int n_output){
	
	z1 = register_module("z1", torch::nn::Linear(n_hidden,n_input));
	z2 = register_module("z2", torch::nn::Linear(n_output,n_hidden));
	
}

//_______________________________________________________________Forward
torch::Tensor nnet::forward(torch::Tensor & X){
	
	X = z1->forward(X);
	X = torch::sigmoid(X);
	X = z2->forward(X);
	X = torch::sigmoid(X);
	
	return X;
}

