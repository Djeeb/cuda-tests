#include "SVRG_nnet.hpp"
#include "function_gen.hpp"
using namespace std;

int main(){
	//______________________________________________Initializing dataset
		int d = 1;
		int n = 20000;
		int n_test = 200;
		auto X_train = torch::rand({n,d}).to(options_double) * 6.28;
		auto X_test = torch::rand({n_test,d}).to(options_double) * 6.28;
		
		
		//auto Y_train = euclidean_norm(X_train).to(options_double);
		//auto Y_test = euclidean_norm(X_test).to(options_double);
		auto Y_train = sin(X_train).to(options_double);
		auto Y_test = sin(X_test).to(options_double);	
	
	
	//_______________________________________Initializing neural network
		int epochs = 41;
		int batch_size = 1;
		double learning_rate = 0.0001;
		nnet neuralnet(n,batch_size,d,20,1,learning_rate,"GPU","SAGA");
		torch::optim::SGD optimizer(neuralnet.parameters(), 0.01);	
		
		
		
	//_________________________________________________Running algorithm
	cout << "Epoch" << "\t" << "loss" << endl;
	auto t1 = chrono::system_clock::now();

		for(int i=1; i <= epochs;i++){
			
			for(int k=0; k < n; k++){
				
				optimizer.zero_grad();
				
				//Forward propagation
				auto X = X_train[k].clone();
				auto Y = Y_train[k].clone();
				X = neuralnet.forward( X );
				
				//Compute loss function
				auto loss =  neuralnet.mse_loss( X , Y );		
				
				//Back-propagation
				loss.backward();
				
				//update
				neuralnet.update(i,k);
			}
			cout << "" << i << "\t" << neuralnet.compute_cost() << endl;
		}

	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	cout << "Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
		
	//______________________________________________________Evaluating model
	
	auto test = X_test.clone();
	auto Y_hat = neuralnet.forward( X_test );
	auto loss =  neuralnet.mse_loss( Y_hat , Y_test );
	
	ofstream file("../../data/sin_app_SGD.dat");
	for(int i=0;i<test.size(0);i++){
		file << test[i].item<double>() << "\t" << Y_hat[i].item<double>() << endl;
	}
	
	cout << "\nTEST SET" << endl;
	cout << "MSE = " << neuralnet.compute_cost() * n << endl;
}
