#include "SVRG.hpp"
#include "function_gen.hpp"
using namespace std;

int main(){

	
	//______________________________________________Initializing dataset
		int batch = 100;
		int d = 1;
		int n = 10000;
		int n_test = 200;
		int n_hidden = 20;
		int m_max = 5; //number of passes
		int warm_epoch = 60;
		double x;
		
		auto X_train = torch::zeros({n,d}).to(options_double);
		auto X_test = torch::zeros({n_test,d}).to(options_double);

		ifstream data_train("../../data/sin_SVRG/X_train.pt");
		for(int i=0;i<n;i++){	
			data_train >> x;
			X_train[i] = x;
		}

		ifstream data_test("../../data/sin_SVRG/X_test.pt");
		for(int i=0;i<n_test;i++){	
			data_test >> x;
			X_test[i] = x;
		}
		
		//auto Y_train = euclidean_norm(X_train).to(options_double);
		//auto Y_test = euclidean_norm(X_test).to(options_double);
		auto Y_train = sin(X_train).to(options_double);
		auto Y_test = sin(X_test).to(options_double);

	//_______________________________________Initializing neural network
		int epochs = 30;
		double learning_rate = 0.15;
		double decay = 0.1;
		nnet neuralnet(n,batch,d,n_hidden,1,learning_rate,"CPU");
		torch::optim::SGD optimizer(neuralnet.parameters(), 0.01);	
		neuralnet.load_parameters();
		
	//_________________________________________________Running algorithm
	ofstream loss_val("../../data/sin_SVRG/loss_SVRG_best");
	cout << "Iter" << "\t\t" << "loss" << endl;
	auto t1 = chrono::system_clock::now();

	for(int i=1; i <= warm_epoch; i++ ){
		for(int k=0; k < (n/batch); k++){
			optimizer.zero_grad();
			
			auto X = X_train.slice(0,k*batch,(k+1)*batch).clone();
			auto Y = Y_train.slice(0,k*batch,(k+1)*batch).clone();
			
			X = neuralnet.forward( X );

			auto loss =  neuralnet.mse_loss( X , Y );
			loss.backward();
			neuralnet.update_SGD();
		}
		double cost_val = neuralnet.compute_cost();
		cout << i << "\t" << cost_val << endl;
		loss_val << i << "\t" << cost_val << endl;
		neuralnet.learning_rate = learning_rate/(1+decay*i);
	}
	
	neuralnet.set_snapshot();
	neuralnet.learning_rate = 0.021;
	
	for(int i=1; i <= epochs;i++){
		//Update of mu
		for(int k=0; k < (n/batch); k++){
			optimizer.zero_grad();
			
			auto X = X_train.slice(0,k*batch,(k+1)*batch).clone();
			auto Y = Y_train.slice(0,k*batch,(k+1)*batch).clone();
			X = neuralnet.forward_snapshot( X );
			auto loss =  neuralnet.mse_loss( X , Y, false );
			loss.backward();
			
			neuralnet.update_mu();
		}	
		
				
		
		//SVRG algorithm on n*m iterations
		for(int m=0; m < m_max; m++){
			for(int k=0; k < (n/batch); k++){
				optimizer.zero_grad();
				
				//Snapshot gradient
				auto X = X_train.slice(0,k*batch,(k+1)*batch).clone();
				auto Y = Y_train.slice(0,k*batch,(k+1)*batch).clone();
				X = neuralnet.forward_snapshot( X );
				auto loss_snapshot =  neuralnet.mse_loss( X , Y, false );	
				loss_snapshot.backward();
			
				//Real gradient
				X = X_train.slice(0,k*batch,(k+1)*batch).clone();
				Y = Y_train.slice(0,k*batch,(k+1)*batch).clone();
				X = neuralnet.forward( X );
				auto loss =  neuralnet.mse_loss( X , Y );
				loss.backward();
				
				//update
				neuralnet.update_SVRG();
			}
		double cost_val = neuralnet.compute_cost();
		cout << (i-1)*m_max + m + warm_epoch + 1 << "\t" << cost_val << endl;
		loss_val << (i-1)*m_max + m + warm_epoch + 1 << "\t" << cost_val << endl;
		}
		neuralnet.set_snapshot();
	}

	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	cout << "Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
	loss_val << "#Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
		
	//______________________________________________________Evaluating model
	
	auto test = X_test.clone();
	auto Y_hat = neuralnet.forward( X_test );
	auto loss =  neuralnet.mse_loss( Y_hat , Y_test );
	
	ofstream file("../../data/sin_SVRG/sin_approx_SVRG");
	for(int i=0;i<test.size(0);i++){
		file << test[i].item<double>() << "\t" << Y_hat[i].item<double>() << endl;
	}
	double cost_val = neuralnet.compute_cost();
	cout << "\nTEST SET" << endl;
	cout << "MSE = " << cost_val * n / n_test << endl;
	loss_val << "\n#TEST SET" << endl;
	loss_val << "#MSE = " << cost_val * n / n_test << endl;
}
