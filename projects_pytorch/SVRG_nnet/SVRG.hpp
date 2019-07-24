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
auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);


class nnet : public torch::nn::Module {
	public:
		int training_size;
		int batch_size;	
		double cost;
		double learning_rate;

		torch::DeviceType device_type;
		torch::nn::Linear z1{nullptr}, z2{nullptr};
		
		//SVRG gradients storage (snapshots)
		torch::nn::Linear z1_snapshot{nullptr}, z2_snapshot{nullptr};
		vector<torch::Tensor> 	mu;
		
		nnet(int,int,int,int,int,double,string device="CPU");
		torch::Tensor forward( torch::Tensor &);
		torch::Tensor cross_entropy_loss(const torch::Tensor &, const torch::Tensor &, bool is_cost=true);
		torch::Tensor mse_loss(const torch::Tensor &, const torch::Tensor &, bool is_cost=true);
		double compute_cost();
		
		//SVRG functions
		torch::Tensor forward_snapshot( torch::Tensor &);
		void set_snapshot();
		void update_SVRG();
		void update_SGD();
		void update_mu();

		
		
		//auxiliary functions
		torch::Tensor predict(torch::Tensor &);
		void load_parameters();
};

//________________________________________________________Initialization
nnet::nnet(int n_train,int n_batch, int n_input,int n_hidden,int n_output,double alpha, string device): training_size(n_train), batch_size(n_batch),
		   cost(0.), learning_rate(alpha) {

	//Device choice
	device_type = (device=="GPU")?torch::kCUDA:torch::kCPU;

	//Real activation functions initialization
	z1 = register_module("z1", torch::nn::Linear(n_input,n_hidden));
	z2 = register_module("z2", torch::nn::Linear(n_hidden,n_output));

	//Snapshot activation functions initialization
	z1_snapshot = register_module("z1_snapshot", torch::nn::Linear(n_input,n_hidden));
	z2_snapshot = register_module("z2_snapshot", torch::nn::Linear(n_hidden,n_output));
	
	for(int i=0; i < 4; i++){
		this->parameters()[i].set_data(this->parameters()[i]*double(2));			
	}

	//Initializating mu and snapshots
	mu.resize(4);
	this->set_snapshot();
	
	//Send Module to device and convert to double
	this->to(device_type,torch::kFloat64);
	
}

//_______________________________________________________________Forward
torch::Tensor nnet::forward( torch::Tensor & X ){	
	X = z1->forward(X);
	X = torch::tanh(X) * 1.5;
	X = z2->forward(X);
	X = torch::tanh(X) * 1.5;
		
	return X;
}

torch::Tensor nnet::forward_snapshot( torch::Tensor & X ){	
	X = z1_snapshot->forward(X);
	X = torch::tanh(X) * 1.5;
	X = z2_snapshot->forward(X);
	X = torch::tanh(X) * 1.5;
		
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
	torch::Tensor J = ((X-Y)*(X-Y)).sum() / double(batch_size);
	if(is_cost) cost += J.item<double>();
	return J;
}

//____________________________________________________Cross-entropy loss
torch::Tensor nnet::cross_entropy_loss(const torch::Tensor & X, const torch::Tensor & Y,bool is_cost){
	torch::Tensor J = (- ( Y * torch::log( X ) + ( 1 - Y ) * torch::log( 1 - X ))).sum() / double(batch_size);
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
		mu[i] += (this->parameters()[i+4].grad().clone() / double(training_size))*batch_size;
	}
}



//___________________________________________________________________SGD
void nnet::update_SGD(){
	for(int i=0; i < 4; i++){
		this->parameters()[i].set_data(this->parameters()[i] - learning_rate * this->parameters()[i].grad());			
	}
}






//___________________________________________________Auxiliary functions

double max_mnist(){
	double x = 0.;
	auto train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),1);

	for(auto& sample : *(train_set)){
		auto X_train = sample.data.reshape({1,784}).to(options_double);
		x = max(x,(X_train * X_train).sum().item<double>());
	}
	return x;
}

double min_mnist(){
	double x = 100000.;
	auto train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),1);

	for(auto& sample : *(train_set)){
		auto X_train = sample.data.reshape({1,784}).to(options_double);
		x = min(x,(X_train * X_train).sum().item<double>());
	}
	return x;
}

torch::Tensor nnet::predict(torch::Tensor & X_test){
	
	return this->forward(X_test).argmax(1).to(torch::TensorOptions().dtype(torch::kInt64));
	
}

double error_rate(const torch::Tensor & Y_test, const torch::Tensor & Y_hat){
	
	return 1 - (at::one_hot(Y_test,10) - at::one_hot(Y_hat,10)).abs().sum().item<double>()/(2.*double(Y_test.size(0)));
	
}

void nnet::load_parameters(){
	
	torch::Tensor W1 = torch::zeros({this->parameters()[0].size(0),this->parameters()[0].size(1)}).to(options_double);
	torch::Tensor b1 = torch::zeros({this->parameters()[1].size(0)}).to(options_double);
	torch::Tensor W2 = torch::zeros({this->parameters()[2].size(0),this->parameters()[2].size(1)}).to(options_double);
	torch::Tensor b2 = torch::zeros({this->parameters()[3].size(0)}).to(options_double);
	
	ifstream data_W1("../../data/sin_SVRG/W1.pt");
	ifstream data_b1("../../data/sin_SVRG/b1.pt");
	ifstream data_W2("../../data/sin_SVRG/W2.pt");
	ifstream data_b2("../../data/sin_SVRG/b2.pt");	
		
		for(int i=0;i<W1.size(0);i++){
			double a,b,c;
			data_W1 >> a;
			W1[i] = a;
			data_b1 >> b;
			b1[i] = b;
			data_W2 >> c;
			W2.slice(1,i,i+1) = c;	
		}
		
		double e;
		data_b2 >> e;
		b2[0] = e;	

		this->parameters()[0].set_data(W1.clone());
		this->parameters()[1].set_data(b1.clone());
		this->parameters()[2].set_data(W2.clone());
		this->parameters()[3].set_data(b2.clone());		
		
		this->set_snapshot();
}


