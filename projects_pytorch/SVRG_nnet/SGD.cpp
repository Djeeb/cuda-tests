#include "SGD.hpp"
#include "function_gen.hpp"
using namespace std;

int main(){

	
	//______________________________________________Initializing dataset
		int d = 1;
		int n = 4000; 
		int n_test = 200;
		
		auto X_train = torch::rand({n,d}).to(options_double) * 6.28;
		auto X_test = torch::rand({n_test,d}).to(options_double) * 6.28;
		
		//auto Y_train = euclidean_norm(X_train).to(options_double);
		//auto Y_test = euclidean_norm(X_test).to(options_double);
		auto Y_train = sin(X_train).to(options_double);
		auto Y_test = sin(X_test).to(options_double);	
	
	
	//_______________________________________Initializing neural network
		int epochs = 40;
		double learning_rate = 0.02;
		nnet neuralnet(n,d,20,1,learning_rate,"GPU");
		torch::optim::SGD optimizer(neuralnet.parameters(), 0.01);	
		
		
		
	//_________________________________________________Running algorithm
	cout << "Iter" << "\t\t" << "loss" << endl;
	auto t1 = chrono::system_clock::now();

		for(int i=1; i <= epochs;i++){
			for(int k=0; k < n; k++){
				optimizer.zero_grad();
				
				auto X = X_train[k].clone();
				auto Y = Y_train[k].clone();
				
				X = neuralnet.forward( X );
				auto loss =  neuralnet.mse_loss( X , Y );
				loss.backward();
				neuralnet.update_SGD();
			
				if((k+1)%n == 0 ) cout << "" << k+1+(i-1)*n << "\t" << neuralnet.compute_cost() << endl;
			}
		}

	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	cout << "Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
		
	//______________________________________________________Evaluating model
	
	auto test = X_test.clone();
	auto Y_hat = neuralnet.forward( X_test );
	auto loss =  neuralnet.mse_loss( Y_hat , Y_test );
	
	ofstream file("../../data/sin_app_SGD_V2.dat");
	for(int i=0;i<test.size(0);i++){
		file << test[i].item<double>() << "\t" << Y_hat[i].item<double>() << endl;
	}
	
	cout << "\nTEST SET" << endl;
	cout << "MSE = " << neuralnet.compute_cost() * n / n_test << endl;
}
