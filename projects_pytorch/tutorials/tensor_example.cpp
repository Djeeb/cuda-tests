#include <torch/torch.h>
#include <iostream>
using namespace std;

//Définition du module simpleneuralnet

class simpleneuralnet : public torch::nn::Module {
	private:
	
		torch::nn::Linear input = nullptr;
		torch::nn::Linear hidden = nullptr;
		torch::nn::Linear output = nullptr;
		
	public:
	
		simpleneuralnet();
		torch::Tensor forward(torch::Tensor &);
		
};


//initialisation
simpleneuralnet::simpleneuralnet(){
			input = register_module("input",torch::nn::Linear(10,10));
			hidden = register_module("hidden",torch::nn::Linear(10,5));
			output = register_module("output",torch::nn::Linear(5,1));	
}

//forward propagation
torch::Tensor simpleneuralnet::forward(torch::Tensor & X){
	X = torch::relu(input->forward(X));
	X = torch::tanh(hidden->forward(X));
	X = torch::sigmoid(output->forward(X));
	
	return X;
}

int main(){
	cout << "\nTest de forward propagation sur une matrice aléatoire :" << endl;
	auto X = torch::rand({10,});
	cout << X << endl;
	
	simpleneuralnet NN;
	auto Y = NN.forward(X);
	
	
	cout << "\nRésultat de la forward propagation :" << endl;
	cout << Y << endl;
}
