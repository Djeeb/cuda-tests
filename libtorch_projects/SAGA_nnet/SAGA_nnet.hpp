#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <string>
#include <chrono>
#include <vector>
using namespace std;
auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);


class nnet : public torch::nn::Module {
	public:
		int training_size;
		int batch_size;
		double cost;
		double learning_rate;
		string optimizer;
		
		torch::DeviceType device_type;
		torch::nn::Linear z1{nullptr}, z2{nullptr};
		
		vector<torch::Tensor> SAGA_W1, SAGA_b1, SAGA_W2, SAGA_b2;
		
		nnet(int,int,int,int,int,double,string device="CPU",string opt="SGD");
		torch::Tensor forward(torch::Tensor &);
		torch::Tensor predict(torch::Tensor &);
		torch::Tensor cross_entropy_loss(const torch::Tensor &, const torch::Tensor &);
		torch::Tensor mse_loss(const torch::Tensor &, const torch::Tensor &);
		double compute_cost();
		
		//update algorithms
		void update(int,int);
		void update_SGD();
		void update_SAGA(int,int);
		void update_SAG(int,int);		

};

//________________________________________________________Initialization
nnet::nnet(int n_train, int n_batch, int n_input,int n_hidden,int n_output,double alpha, string device,string opt): training_size(n_train), batch_size(n_batch),
		   cost(0.), learning_rate(alpha), optimizer(opt) {

	//Device choice
	device_type = (device=="GPU")?torch::kCUDA:torch::kCPU;

	//Activation functions initialization
	z1 = register_module("z1", torch::nn::Linear(n_input,n_hidden));
	z2 = register_module("z2", torch::nn::Linear(n_hidden,n_output));
	
	//Custom weight initialization
	//this->parameters()[0].set_data(torch::randn({n_hidden,n_input}));
	//this->parameters()[2].set_data(torch::randn({n_output,n_hidden}));
	
	if(optimizer == "SAGA" or optimizer == "SAG"){
		cout << "Initializing gradients lists..." << endl;
		SAGA_W1.resize(n_train+1);
		SAGA_b1.resize(n_train+1);
		SAGA_W2.resize(n_train+1);
		SAGA_b2.resize(n_train+1);
		for(int i=0; i < training_size+1; i++){
			SAGA_W1[i] = torch::zeros({n_hidden,n_input}).to(options_double);
			SAGA_b1[i] = torch::zeros({n_hidden}).to(options_double);
			SAGA_W2[i] = torch::zeros({n_output,n_hidden}).to(options_double);
			SAGA_b2[i] = torch::zeros({n_output}).to(options_double);
		}
		cout << "initialization ended." << endl;
	}
	
	//Send Module to device and convert to double
	this->to(device_type,torch::kFloat64);
	
}

//_______________________________________________________________Forward
torch::Tensor nnet::forward(torch::Tensor & X){
	X = z1->forward(X);
	X = torch::relu(X);
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
double nnet::compute_cost(){
	double x = cost * double(batch_size) / double(training_size);
	cost = 0;
	return x;
}

//__________________________________________________________max/min norm

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


//------------------- CUSTOM LOSS FUNCTIONS ----------------------------


//____________________________________________________Cross-entropy loss
torch::Tensor nnet::cross_entropy_loss(const torch::Tensor & X, const torch::Tensor & Y){
	torch::Tensor J = (- ( Y * torch::log( X ) + ( 1 - Y ) * torch::log( 1 - X ))).sum() / double(X.size(0));
	cost += J.item<double>();
	return J;
}

//______________________________________________________________MSE loss
torch::Tensor nnet::mse_loss(const torch::Tensor & X, const torch::Tensor & Y){
	torch::Tensor J = ((X-Y)*(X-Y)).sum() / double(X.size(0));
	cost += J.item<double>();
	return J;
}

//------------------- CUSTOM UPDATE ALGORITHMS METHODS -----------------

//________________________________________________________________UPDATE
void nnet::update(int epoch, int iter){
	
	if(optimizer=="SGD") this->update_SGD();
	else if(optimizer=="SAGA") this->update_SAGA(epoch,iter);
	else if(optimizer=="SAG") this->update_SAG(epoch,iter);
}





//___________________________________________________________________SGD
void nnet::update_SGD(){
	for(int i=0; i < 4; i++){
		
		this->parameters()[i].set_data(this->parameters()[i] - learning_rate * this->parameters()[i].grad());
				
	}
}

//__________________________________________________________________SAGA
void nnet::update_SAGA(int epoch,int i){
	
	//Init with SGD
	if(epoch==1){

		SAGA_W1[i].set_data(this->parameters()[0].grad().clone());
		SAGA_b1[i].set_data(this->parameters()[1].grad().clone());
		SAGA_W2[i].set_data(this->parameters()[2].grad().clone());
		SAGA_b2[i].set_data(this->parameters()[3].grad().clone());
		
		SAGA_W1[training_size] += SAGA_W1[i] / double(training_size);
		SAGA_b1[training_size] += SAGA_b1[i] / double(training_size);
		SAGA_W2[training_size] += SAGA_W2[i] / double(training_size);
		SAGA_b2[training_size] += SAGA_b2[i] / double(training_size);
				
	}
	
	//SAGA
	else{
		
		this->parameters()[0].set_data(this->parameters()[0] - learning_rate * ( this->parameters()[0].grad() - SAGA_W1[i] + SAGA_W1[training_size] ) );
		this->parameters()[1].set_data(this->parameters()[1] - learning_rate * ( this->parameters()[1].grad() - SAGA_b1[i] + SAGA_b1[training_size] ) );
		this->parameters()[2].set_data(this->parameters()[2] - learning_rate * ( this->parameters()[2].grad() - SAGA_W2[i] + SAGA_W2[training_size] ) );
		this->parameters()[3].set_data(this->parameters()[3] - learning_rate * ( this->parameters()[3].grad() - SAGA_b2[i] + SAGA_b2[training_size] ) );
		
		SAGA_W1[training_size] += ( this->parameters()[0].grad() - SAGA_W1[i] ) / double(training_size);
		SAGA_b1[training_size] += ( this->parameters()[1].grad() - SAGA_b1[i] ) / double(training_size);
		SAGA_W2[training_size] += ( this->parameters()[2].grad() - SAGA_W2[i] ) / double(training_size);
		SAGA_b2[training_size] += ( this->parameters()[3].grad() - SAGA_b2[i] ) / double(training_size);
	
		SAGA_W1[i].set_data(this->parameters()[0].grad().clone());
		SAGA_b1[i].set_data(this->parameters()[1].grad().clone());
		SAGA_W2[i].set_data(this->parameters()[2].grad().clone());
		SAGA_b2[i].set_data(this->parameters()[3].grad().clone());
					
	}
}

//___________________________________________________________________SAG
void nnet::update_SAG(int epoch,int i){
	
	//Init with SGD
	if(epoch==1){

		SAGA_W1[i].set_data(this->parameters()[0].grad().clone());
		SAGA_b1[i].set_data(this->parameters()[1].grad().clone());
		SAGA_W2[i].set_data(this->parameters()[2].grad().clone());
		SAGA_b2[i].set_data(this->parameters()[3].grad().clone());
		
		SAGA_W1[training_size] += SAGA_W1[i] / double(training_size);
		SAGA_b1[training_size] += SAGA_b1[i] / double(training_size);
		SAGA_W2[training_size] += SAGA_W2[i] / double(training_size);
		SAGA_b2[training_size] += SAGA_b2[i] / double(training_size);
				
	}
	
	//SAG
	else{

		SAGA_W1[training_size] += ( this->parameters()[0].grad() - SAGA_W1[i] ) / double(training_size);
		SAGA_b1[training_size] += ( this->parameters()[1].grad() - SAGA_b1[i] ) / double(training_size);
		SAGA_W2[training_size] += ( this->parameters()[2].grad() - SAGA_W2[i] ) / double(training_size);
		SAGA_b2[training_size] += ( this->parameters()[3].grad() - SAGA_b2[i] ) / double(training_size);
		
		this->parameters()[0].set_data(this->parameters()[0] - learning_rate * ( SAGA_W1[training_size] ) );
		this->parameters()[1].set_data(this->parameters()[1] - learning_rate * ( SAGA_b1[training_size] ) );
		this->parameters()[2].set_data(this->parameters()[2] - learning_rate * ( SAGA_W2[training_size] ) );
		this->parameters()[3].set_data(this->parameters()[3] - learning_rate * ( SAGA_b2[training_size] ) );

		SAGA_W1[i].set_data(this->parameters()[0].grad().clone());
		SAGA_b1[i].set_data(this->parameters()[1].grad().clone());
		SAGA_W2[i].set_data(this->parameters()[2].grad().clone());
		SAGA_b2[i].set_data(this->parameters()[3].grad().clone());
					
	}
}
