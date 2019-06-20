#include <torch/torch.h>
#include <iostream>
using namespace std;

class nnet {
	private:
		double learning_rate; //learning rate
		
		int n_input,n_hidden,n_output;
	
		torch::Tensor W1, b1, W2, b2; //Poids et biais
		torch::Tensor z1, g1, z2, g2; //Fonctions intermédiaires linéaires et activations
		torch::Tensor dW1, db1, dW2, db2, dz1, dg1, dz2, dg2; //gradients
		torch::Tensor J; //cost function
		
	public:
		nnet(int,int,int,double);
		void forward(torch::Tensor &);
		void compute_cost(torch::Tensor &);
		void backward(const torch::Tensor &,const torch::Tensor &);
		void update();
		
	//Visualisation paramètres
		torch::Tensor Weight1() const { return W1;}
		torch::Tensor Weight2() const { return W2;}
		torch::Tensor bias1() const { return b1;}
		torch::Tensor bias2() const { return b2;}
		torch::Tensor cost() const { return J.sum();}
};

//initialisation
nnet::nnet(int n_i,int n_h,int n_o, double alpha): n_input(n_i), n_hidden(n_h), n_output(n_o), learning_rate(alpha) {
	
	//1ere couche
	W1 = torch::randn({n_hidden,n_input}, torch::dtype(torch::kFloat64));
	b1 = torch::zeros({n_hidden,1}, torch::dtype(torch::kFloat64));
	
	//2eme couche
	W2 = torch::randn({n_output,n_hidden}, torch::dtype(torch::kFloat64));
	b2 = torch::zeros({n_output,1}, torch::dtype(torch::kFloat64));
	
}

//forward propagation
void nnet::forward(torch::Tensor & X){
	
	//1ere couche
	z1 = W1.mm(X) + b1;
	g1 = torch::sigmoid(z1);
	
	//2eme couche
	z2 = W2.mm(g1) + b2;
	g2 = torch::sigmoid(z2);
	
}

//Calcul du coût
void nnet::compute_cost(torch::Tensor & Y){
	
	J = (torch::log(g2) * Y ) + (torch::log(1-g2) * (1-Y)) ;
	cout << J.sum() << endl;
	//cout << g2 << endl;
	
}

//Backward propagation
void nnet::backward(const torch::Tensor & X,const torch::Tensor & Y){

	// dJ/dg2
	dg2 = -(Y / g2) - ((1-Y) / (1-g2));
	//cout << "g2 : "<< g2.size(0) <<" x "  << g2.size(1) << endl;
	//cout << "dg2 : "<< dg2.size(0) <<" x "  << dg2.size(1) << endl;
	
	// dJ/dz2
	dz2 = dg2 * z2 * (1 - z2);
	//cout << "\nz2 : "<< z2.size(0) <<" x "  << z2.size(1) << endl;
	//cout << "dz2 : "<< dz2.size(0) <<" x "  << dz2.size(1) << endl;
	
	// dJ/dW2
	dW2 = dz2.mm(g1.transpose(0,1));
	//cout << "\nW2 : "<< W2.size(0) <<" x "  << W2.size(1) << endl;
	//cout << "dW2 : "<< dW2.size(0) <<" x "  << dW2.size(1) << endl;
	
	// dJ/db2
	
	 
	db2 = dz2.transpose(0,1)[0].reshape({n_output,1});
	for(long i = 1; i < n_output; i++){
		db2 += dz2.transpose(0,1)[i].reshape({n_output,1});
	}
	//cout << "\nb2 : "<< b2.size(0) <<" x "  << b2.size(1) << endl;
	//cout << "db2 : "<< db2.size(0) <<" x "  << db2.size(1) << endl;
	
	// dJ/dg1
	dg1 = W2.reshape({64,10});
	dg1 = dg1.mm(dz2);
	//cout << "\ng1 : "<< g1.size(0) <<" x "  << g1.size(1) << endl;
	//cout << "dg1 : "<< dg1.size(0) <<" x "  << dg1.size(1) << endl;
	
	// dJ/dz1
	dz1 = dg1 * z1 * (1 - z1);
	//cout << "\nz1 : "<< z1.size(0) <<" x "  << z1.size(1) << endl;
	//cout << "dz1 : "<< dz1.size(0) <<" x "  << dz1.size(1) << endl;
	
	// dJ/dW1 
	dW1 = dz1.mm(X.transpose(0,1));
	//cout << "\nW1 : "<< W1.size(0) <<" x "  << W1.size(1) << endl;
	//cout << "dW1 : "<< dW1.size(0) <<" x "  << dW1.size(1) << endl;
	
	// dJ/db1 
	db1 = dz1.transpose(0,1)[0].reshape({n_hidden,1});
	for(long i = 1; i < n_hidden; i++){
		db1 += dz1.transpose(0,1)[i].reshape({n_hidden,1});
	}
	//cout << "\nb1 : "<< b1.size(0) <<" x "  << b1.size(1) << endl;
	//cout << "db1 : "<< db1.size(0) <<" x "  << db1.size(1) << endl;
}

void nnet::update(){
	
	//1ere couche
	W2 -= dW2 * learning_rate;
	b2 -= db2 * learning_rate;
	//2eme couche
	W1 -= dW1 * learning_rate;
	b1 -= db1 * learning_rate;
	
}



int main(){
	nnet neuralnet(784,64,10,10);
	torch::Tensor Y;
	torch::Tensor X;
	
	
	auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("../data").map(
          torch::data::transforms::Stack<>()),64);
          
	int i=0;
	for(auto& sample : *data_loader){
		X = sample.data.reshape({784,64}).to(torch::TensorOptions().dtype(torch::kFloat64));
		Y = torch::zeros({10,64},torch::TensorOptions().dtype(torch::kFloat64));
		for(long k=0;k<64;k++){
			auto j = sample.target.accessor<long,1>()[k];
			Y[j][k] = 1.;
		}
		//Y = sample.target.reshape({1,1}).to(torch::TensorOptions().dtype(torch::kFloat64));
		
		neuralnet.forward(X);
		neuralnet.compute_cost(Y);
		neuralnet.backward(X,Y);
		neuralnet.update();
		
		i++;
		cout << neuralnet.cost() << endl;
		if(i==5) break;
	}
          

	//cout << neuralnet.Weight1() << endl;
	//cout << neuralnet.Weight2() << endl;
	//cout << neuralnet.bias1() << endl;
	//cout << neuralnet.bias2() << endl;
	
}
