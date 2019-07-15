#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <chrono>
#include <vector>
using namespace std;
auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);


class nnet : public torch::nn::Module {
	public:
		int training_size;
		double cost;
		double learning_rate;
		string optimizer;
		
		torch::DeviceType device_type;
		torch::nn::Linear z1{nullptr}, z2{nullptr};
		
		nnet(int,int,int,int,double,string device="CPU");
		
		torch::Tensor forward( torch::Tensor &);
		torch::Tensor mse_loss(const torch::Tensor &, const torch::Tensor &);
		double compute_cost();
		void update_SGD();

};

//________________________________________________________Initialization
nnet::nnet(int n_train, int n_input,int n_hidden,int n_output,double alpha, string device): training_size(n_train),
		   cost(0.), learning_rate(alpha) {

	//Device choice
	device_type = (device=="GPU")?torch::kCUDA:torch::kCPU;

	//Activation functions initialization
	z1 = register_module("z1", torch::nn::Linear(n_input,n_hidden));
	z2 = register_module("z2", torch::nn::Linear(n_hidden,n_output));
	
	//Send Module to device and convert to double
	this->to(device_type,torch::kFloat64);
	
}

//_______________________________________________________________Forward
torch::Tensor nnet::forward(torch::Tensor & X){
	X = z1->forward(X);
	X = torch::tanh(X)*1.2;	
	X = z2->forward(X);
	X = torch::tanh(X)*1.2;		
	return X;
}

//____________________________________________________________Reset cost
double nnet::compute_cost(){
	double x = cost / ( double(training_size) );
	cost = 0;
	return x;
}


//------------------- CUSTOM LOSS FUNCTIONS ----------------------------

//______________________________________________________________MSE loss
torch::Tensor nnet::mse_loss(const torch::Tensor & X, const torch::Tensor & Y){
	torch::Tensor J = ((X-Y)*(X-Y)).sum() ;
	cost += J.item<double>();
	return J;
}

//------------------- CUSTOM UPDATE ALGORITHMS METHODS -----------------

//___________________________________________________________________SGD
void nnet::update_SGD(){
	for(int i=0; i < 4; i++){
		this->parameters()[i].set_data(this->parameters()[i] - learning_rate * this->parameters()[i].grad());			
	}
}
