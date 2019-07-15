#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <chrono>
#include <vector>
#include <random>
using namespace std;
auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);


class nnet : public torch::nn::Module {
	public:
		int training_size;
		double cost;
		double learning_rate;
		
		torch::DeviceType device_type;
		torch::nn::Linear z1{nullptr}, z2{nullptr};
		
		//SVRG gradients storage (snapshots)
		torch::nn::Linear z1_snapshot{nullptr}, z2_snapshot{nullptr};
		vector<torch::Tensor> 	mu;
		
		nnet(int,int,int,int,double,string device="CPU");
		torch::Tensor forward( torch::Tensor &);
		torch::Tensor cross_entropy_loss(const torch::Tensor &, const torch::Tensor &);
		torch::Tensor mse_loss(const torch::Tensor &, const torch::Tensor &, bool is_cost=true);
		double compute_cost();
		
		//SVRG functions
		torch::Tensor forward_snapshot( torch::Tensor &);
		void set_snapshot();
		void update_SVRG();
		void update_SGD();
		void update_mu();
		
		
		//update algorithms
		void update(int,int &);
};

//________________________________________________________Initialization
nnet::nnet(int n_train, int n_input,int n_hidden,int n_output,double alpha, string device): training_size(n_train),
		   cost(0.), learning_rate(alpha) {

	//Device choice
	device_type = (device=="GPU")?torch::kCUDA:torch::kCPU;

	//Real activation functions initialization
	z1 = register_module("z1", torch::nn::Linear(n_input,n_hidden));
	z2 = register_module("z2", torch::nn::Linear(n_hidden,n_output));

	//Snapshot activation functions initialization
	z1_snapshot = register_module("z1_snapshot", torch::nn::Linear(n_input,n_hidden));
	z2_snapshot = register_module("z2_snapshot", torch::nn::Linear(n_hidden,n_output));

	//Initializating mu and snapshots
	mu.resize(4);
	this->set_snapshot();
	
	//Send Module to device and convert to double
	this->to(device_type,torch::kFloat64);
	
}

//_______________________________________________________________Forward
torch::Tensor nnet::forward( torch::Tensor & X ){	
	X = z1->forward(X);
	X = torch::tanh(X)*1.2;
	X = z2->forward(X);
	X = torch::tanh(X)*1.2;
		
	return X;
}

torch::Tensor nnet::forward_snapshot( torch::Tensor & X ){	
	X = z1_snapshot->forward(X);
	X = torch::tanh(X)*1.2;
	X = z2_snapshot->forward(X);
	X = torch::tanh(X)*1.2;
		
	return X;
}

//____________________________________________________________Reset cost
double nnet::compute_cost(){
	double x = cost / ( double(training_size) );
	cost = 0.;
	return x;
}

//______________________________________________________________MSE loss
torch::Tensor nnet::mse_loss(const torch::Tensor & X, const torch::Tensor & Y,bool is_cost){
	torch::Tensor J = ((X-Y)*(X-Y)).sum();
	if(is_cost) cost += J.item<double>();
	return J;
}

//------------------- CUSTOM UPDATE ALGORITHM METHODS -----------------

void nnet::set_snapshot(){
	
	//set snapshot to the most recent value of W
	for(int i=0;i<4;i++){
		this->parameters()[i+4].set_data( this->parameters()[i].clone() );
	}	
	//reinitialize mu
	mu[0] = torch::zeros({this->parameters()[0].size(0),this->parameters()[0].size(1)}).to(options_double);
	mu[1] = torch::zeros({this->parameters()[1].size(0)}).to(options_double);
	mu[2] = torch::zeros({this->parameters()[2].size(0),this->parameters()[2].size(1)}).to(options_double);
	mu[3] = torch::zeros({this->parameters()[3].size(0)}).to(options_double);
	
}


void nnet::update_SVRG(){
	for(int i=0;i<4;i++){
		this->parameters()[i].set_data(this->parameters()[i].clone() - learning_rate * ( this->parameters()[i].grad().clone() - this->parameters()[i+4].grad().clone() + mu[i] ) );
	}	
}

void nnet::update_mu(){
	for(int i=0;i<4;i++){
		mu[i] += this->parameters()[i+4].grad().clone() / double(training_size);
	}
}



//___________________________________________________________________SGD
void nnet::update_SGD(){
	for(int i=0; i < 4; i++){
		this->parameters()[i].set_data(this->parameters()[i] - learning_rate * this->parameters()[i].grad());			
	}
}
