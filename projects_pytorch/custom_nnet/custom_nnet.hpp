#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <string>
#include <chrono>
using namespace std;

class nnet : public torch::nn::Module {
	public:
		double cost;
		double learning_rate;
		
		torch::DeviceType device_type;
		torch::nn::Linear z1{nullptr}, z2{nullptr};

		
		nnet(int,int,int,double,string device="CPU");
		torch::Tensor forward(torch::Tensor &);
		torch::Tensor predict(torch::Tensor &);
		torch::Tensor cross_entropy_loss(const torch::Tensor &, const torch::Tensor &);
		double compute_cost(int, int);
		void update_SGD();
		

};

//________________________________________________________Initialization
nnet::nnet(int n_input,int n_hidden,int n_output,double alpha, string device): cost(0.), learning_rate(alpha) {

	//Device choice
	device_type = (device=="GPU")?torch::kCUDA:torch::kCPU;

	//Activation functions initialization
	z1 = register_module("z1", torch::nn::Linear(n_input,n_hidden));
	z2 = register_module("z2", torch::nn::Linear(n_hidden,n_output));
	
	//Custom weight-bias initialization
	this->parameters()[0].set_data(torch::randn({n_hidden,n_input}));
	this->parameters()[2].set_data(torch::randn({n_output,n_hidden}));

	//Send Module to device and convert to double
	this->to(device_type,torch::kFloat64);
	
}

//_______________________________________________________________Forward
torch::Tensor nnet::forward(torch::Tensor & X){
	X = z1->forward(X);
	X = torch::sigmoid(X);
	X = z2->forward(X);
	X = torch::sigmoid(X);
	
	return X;

}

//____________________________________________________________Prediction
torch::Tensor nnet::predict(torch::Tensor & X_test){
	
	return this->forward(X_test).argmax(1).to(torch::TensorOptions().dtype(torch::kInt64));
	
}


//____________________________________________________________Error_rate
double error_rate(const torch::Tensor & Y_test, const torch::Tensor & Y_hat){
	
	return 1 - (at::one_hot(Y_test,10) - at::one_hot(Y_hat,10)).abs().sum().item<double>()/(2.*double(Y_test.size(0)));
	
}

//____________________________________________________________Reset cost
double nnet::compute_cost(int batch_size,int training_size){
	double x = cost * double(batch_size) / double(training_size);
	cost = 0;
	return x;
}


//------------------- CUSTOM LOSS FUNCTIONS ----------------------------


//____________________________________________________Cross-entropy loss
torch::Tensor nnet::cross_entropy_loss(const torch::Tensor & X, const torch::Tensor & Y){
	torch::Tensor J = (- ( Y * torch::log( X ) + ( 1 - Y ) * torch::log( 1 - X ))).sum() / double(X.size(0));
	cost += J.item<double>();
	return J;
	
}


//------------------- CUSTOM UPDATE ALGORITHMS METHODS -----------------


//____________________________________________________________Manual SGD
void nnet::update_SGD(){
	
	for(int i=0; i < 4; i++){
		//cout << this->parameters()[i].grad() << endl;
		this->parameters()[i].set_data(this->parameters()[i] - learning_rate * this->parameters()[i].grad());
				
	}
	
}


