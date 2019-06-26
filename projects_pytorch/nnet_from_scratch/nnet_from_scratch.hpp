#include <torch/torch.h>
#include <iostream>
#include <chrono>
using namespace std;

class nnet {
	private:

		int n_input, n_hidden, n_output; //dimensions
		double learning_rate; //learning rate	
		int batch_size; //nombre d'échantillons par actualisation		
		bool momentum;
		torch::Tensor J; //cost function
		torch::DeviceType device_type; //utilisé pour switcher facilement de CPU à GPU
		
		torch::Tensor W1, b1, W2, b2; //Poids et biais
		torch::Tensor z1, g1, z2, g2; //Fonctions intermédiaires linéaires et activations
		torch::Tensor dW1, db1, dW2, db2, dz1, dg1, dz2, dg2; //gradients
		torch::Tensor VdW1,VdW2,Vdb1,Vdb2; //momentum

		
	public:
		nnet(int,int,int,int,double,torch::DeviceType);
		void forward(const torch::Tensor &);
		void compute_cost(torch::Tensor &);
		void backward(const torch::Tensor &,const torch::Tensor &);
		void autoback();
		void update();
		torch::Tensor predict(const torch::Tensor &);
		
	//Options
		void set_momentum(bool x) { momentum = x;}
		
	//Visualisation paramètres
		torch::Tensor Weight1() const { return W1;}
		torch::Tensor Weight2() const { return W2;}
		torch::Tensor bias1() const { return b1;}
		torch::Tensor bias2() const { return b2;}
		torch::Tensor dWeight1() const { return dW1;}
		torch::Tensor dWeight2() const { return dW2;}
		torch::Tensor dbias1() const { return db1;}
		torch::Tensor dbias2() const { return db2;}
		double reset_cost(int );
		torch::Tensor activation2() const { return g2;}
};

//________________________________________________________INITIALISATION 

nnet::nnet(int n_i,int n_h,int n_o, int n_batch, double alpha, torch::DeviceType device): n_input(n_i), n_hidden(n_h), n_output(n_o), 
															   learning_rate(alpha), batch_size(n_batch), device_type(device) {
	
	//1ere couche
	W1 = torch::randn({n_hidden,n_input}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(true));
	b1 = torch::zeros({n_hidden,1}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(true));
	
	//2eme couche
	W2 = torch::randn({n_output,n_hidden}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(true));
	b2 = torch::zeros({n_output,1}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(true));
	
	//momentum
	VdW1 = torch::zeros({n_hidden,n_input}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(false));
	Vdb1 = torch::zeros({n_hidden,1}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(false));
	
	VdW2 = torch::zeros({n_output,n_hidden}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(false));
	Vdb2 = torch::zeros({n_output,1}, torch::dtype(torch::kFloat64).device(device_type).requires_grad(false));
	
	
	//cost initialization
	J = torch::zeros({1}, torch::dtype(torch::kFloat64).device(device_type));
	
}


//___________________________________________________FORWARD PROPAGATION
void nnet::forward(const torch::Tensor & X){
	
	//1ere couche
	z1 = W1.mm(X) + b1;
	g1 = torch::sigmoid(z1);
	
	//2eme couche
	z2 = W2.mm(g1) + b2;
	g2 = torch::sigmoid(z2);
	
}


//________________________________________________________CALCUL DU COÛT
void nnet::compute_cost(torch::Tensor & Y){
	J += (- (Y * torch::log(g2) + (1-Y) * torch::log(1-g2))).sum() / double(batch_size);;
	
	
	//J += ((g2-Y)*(g2-Y)).sum().item<double>() / double(batch_size);;
}

double nnet::reset_cost(int training_size) { 
	double x = J.item<double>()*double(batch_size)/double(training_size);
	J = torch::zeros({1}, torch::dtype(torch::kFloat64).device(device_type));
	return x;}

//____________________________________________________BACK - PROPAGATION
void nnet::backward(const torch::Tensor & X,const torch::Tensor & Y){

	// dJ/dg2
	dg2 = -((Y / g2) - ((1-Y) / (1-g2)))/double(batch_size);
	//dg2 = 2*(g2-Y)/double(batch_size);
	//cout << "g2 : "<< g2.size(0) <<" x "  << g2.size(1) << endl;
	//cout << "dg2 : "<< dg2.size(0) <<" x "  << dg2.size(1) << endl;
	
	// dJ/dz2
	dz2 = dg2 * g2 * (1 - g2);
	//cout << "\nz2 : "<< z2.size(0) <<" x "  << z2.size(1) << endl;
	//cout << "dz2 : "<< dz2.size(0) <<" x "  << dz2.size(1) << endl;
	
	// dJ/dW2
	dW2 = dz2.mm(g1.transpose(0,1));
	//cout << "\nW2 : "<< W2.size(0) <<" x "  << W2.size(1) << endl;
	//cout << "dW2 : "<< dW2.size(0) <<" x "  << dW2.size(1) << endl;
	
	// dJ/db2
	dz2 = dz2.transpose(0,1);
	db2 = dz2[0].reshape({n_output,1});
	for(long i = 1; i < batch_size; i++){
		db2 += dz2[i].reshape({n_output,1});
	}
	dz2 = dz2.transpose(0,1);
	//cout << "\nb2 : "<< b2.size(0) <<" x "  << b2.size(1) << endl;
	//cout << "db2 : "<< db2.size(0) <<" x "  << db2.size(1) << endl;
	
	// dJ/dg1
	dg1 = (W2.transpose(0,1)).mm(dz2);
	//cout << "\ng1 : "<< g1.size(0) <<" x "  << g1.size(1) << endl;
	//cout << "dg1 : "<< dg1.size(0) <<" x "  << dg1.size(1) << endl;
	
	// dJ/dz1
	dz1 = dg1 * g1 * (1 - g1);
	//cout << "\nz1 : "<< z1.size(0) <<" x "  << z1.size(1) << endl;
	//cout << "dz1 : "<< dz1.size(0) <<" x "  << dz1.size(1) << endl;
	
	// dJ/dW1 
	dW1 = dz1.mm(X.transpose(0,1));
	//cout << "\nW1 : "<< W1.size(0) <<" x "  << W1.size(1) << endl;
	//cout << "dW1 : "<< dW1.size(0) <<" x "  << dW1.size(1) << endl;
	
	// dJ/db1
	dz1 = dz1.transpose(0,1);
	db1 = dz1[0].reshape({n_hidden,1});
	for(long i = 1; i < batch_size; i++){
		db1 += dz1[i].reshape({n_hidden,1});
	}
	dz1 = dz1.transpose(0,1);
	//cout << "\nb1 : "<< b1.size(0) <<" x "  << b1.size(1) << endl;
	//cout << "db1 : "<< db1.size(0) <<" x "  << db1.size(1) << endl;
}


//____________________________________________________BACK WITH AUTOGRAD
void nnet::autoback(){

	J.backward(c10::nullopt,true,true);
	
	dW2 = W2.grad();
	db2 = b2.grad();
	dW1 = W1.grad();
	db1 = b1.grad();
	cout << b1.grad() << endl;

}

//__________________________________________ACTUALISATION DES PARAMÈTRES
void nnet::update(){
	if(momentum){
		//calcul du momentum
		VdW2 = 0.9 * VdW2 + 0.1 * dW2;
		Vdb2 = 0.9 * Vdb2 + 0.1 * db2;
		
		VdW1 = 0.9 * VdW1 + 0.1 * dW1;
		Vdb1 = 0.9 * Vdb1 + 0.1 * db1;
		
		//1ere couche
		W2 = W2 - VdW2 * learning_rate;
		b2 = b2 - Vdb2 * learning_rate;
		
		//2eme couche
		W1 = W1 - VdW1 * learning_rate;
		b1 = b1 - Vdb1 * learning_rate;
	}
	else{
		//1ere couche
		W2 = W2 - dW2 * learning_rate;
		b2 = b2 - db2 * learning_rate;
		
		//2eme couche
		W1 = W1 - dW1 * learning_rate;
		b1 = b1 - db1 * learning_rate;
	}		
		
		

}


//____________________________________________________________PREDICTION
torch::Tensor nnet::predict(const torch::Tensor & X_test){
	this->forward(X_test);
	return g2.argmax(0).to(torch::TensorOptions().dtype(torch::kInt64));
}


//____________________________________________________________ERROR RATE
double error_rate(const torch::Tensor & Y_test, const torch::Tensor & Y_hat){
	return 1 - (at::one_hot(Y_test,10) - at::one_hot(Y_hat,10)).abs().sum().item<double>()/(2.*double(Y_test.size(0)));
}
