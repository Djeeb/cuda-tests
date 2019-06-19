#include <torch/torch.h>
#include <iostream>
#include <typeinfo>
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
//Opérations usuelles
	cout << "\nTest d'opérations simples sur des tenseurs :" << endl;
	
	auto A = torch::ones({1,10});
	auto B = torch::ones({1,10});
	B = B*3;
	cout << "A = " << A << endl;
	cout << "\nB = " << B << endl;
	cout << "\nB * 5 = " << B*5 << endl;
	cout << "\nA * B = " << A*B << endl;
	B = at::reshape(B,{10,});
	A = at::reshape(A,{10,});
	cout << "\ndot(A,B.T) = " << at::dot(A,B) << endl;
	
//Forward propagation
	cout << "\nTest de forward propagation sur une matrice aléatoire :" << endl;
	//initialisation de host et device
	torch::Device host(torch::kCPU);
	torch::Device device(torch::kCUDA);

	torch::Tensor X = torch::rand({10,},host);
	cout << X << endl;
	cout << typeid(X).name() << endl;
	
	simpleneuralnet NN;
	torch::Tensor Y;
	Y = NN.forward(X).to(device);
	
	
	cout << "\nRésultat de la forward propagation :" << endl;
	cout << Y << endl;
	cout << typeid(Y).name() << endl;
}
