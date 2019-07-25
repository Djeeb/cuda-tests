#include "SVRG.hpp"
#include "function_gen.hpp"
using namespace std;
ofstream file("../../data/mnist_SVRG/SVRG_best");

int main(){
	int batch = 10;

	//_______________________________________Initializing neural network
	int d = 784;
	int n = 60000; 
	int n_test = 10000;
	int epochs = 100;
	int warm_epochs = 3;
	int k = 0;
	int m_max = 5;
	double learning_rate = 0.25;
	double warm_learning_rate = 0.03;
	double warm_decay = 0.1;
	double cost;
	nnet neuralnet(n,batch,d,100,10,warm_learning_rate,"GPU");
	torch::optim::SGD optimizer(neuralnet.parameters(), 0.01);	
				
	//_________________________________________________Running algorithm
	cout << "Iter" << "\t\t" << "loss" << endl;
	auto t1 = chrono::system_clock::now();
	
		//Warm start during 10 epochs
	for(int i=1; i <= warm_epochs;i++){
		
	auto train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),batch);
		
		for(auto& sample : *(train_set)){
			optimizer.zero_grad();
			
			
			auto X = sample.data.reshape({batch,d}).to(options_double);		
			auto Y = at::one_hot(sample.target,10).to(options_double);
			
			X = neuralnet.forward( X );
			auto loss =  neuralnet.mse_loss( X , Y );
			loss.backward();
			neuralnet.update_SGD();

		}
		cost = neuralnet.compute_cost()*batch;
		cout << "" << i << "\t" << cost << endl;
		file << "" << i << "\t" << cost << endl;
		neuralnet.learning_rate = warm_learning_rate/(1+warm_decay*i);
	}
	
	neuralnet.set_snapshot();
	neuralnet.learning_rate = learning_rate;


	for(int i=1; i <= epochs;i++){
		auto train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),batch);
		
		//Update of mu
		for(auto& sample : *(train_set)){
			optimizer.zero_grad();
			
			auto X = sample.data.reshape({batch,d}).to(options_double);		
			auto Y = at::one_hot(sample.target,10).to(options_double);
			X = neuralnet.forward_snapshot( X );
			auto loss =  neuralnet.mse_loss( X , Y, false );
			loss.backward();
			
			neuralnet.update_mu();
			k++;
		}	
		
		//SVRG algorithm on n*m iterations
		for(int m=0; m<m_max;m++){
			train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),batch);

			for(auto& sample : *(train_set)){
				optimizer.zero_grad();

				//Snapshot gradient
				auto X = sample.data.reshape({batch,d}).to(options_double);		
				auto Y = at::one_hot(sample.target,10).to(options_double);
				X = neuralnet.forward_snapshot( X );
				auto loss_snapshot =  neuralnet.mse_loss( X , Y, false );	
				loss_snapshot.backward();
			
				//Real gradient
				X = sample.data.reshape({batch,d}).to(options_double);		
				Y = at::one_hot(sample.target,10).to(options_double);
				X = neuralnet.forward( X );
				auto loss =  neuralnet.mse_loss( X , Y );
				loss.backward();
				
				//update
				neuralnet.update_SVRG();
			}
		cost = neuralnet.compute_cost()*batch;
		cout << "" << (i-1)*m_max + m + warm_epochs + 1 << "\t" << cost << endl;
		file << "" << (i-1)*m_max + m + warm_epochs + 1 << "\t" << cost << endl;
		}
		neuralnet.set_snapshot();
	}

	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	cout << "Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
		
	//______________________________________________________Evaluating model
	
	auto test_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data",torch::data::datasets::MNIST::Mode::kTest).map(
                     torch::data::transforms::Stack<>()),n_test);
                     
	cout << "\n* Précision sur le test set : ";
	file << "\n# Précision sur le test set : ";
	for(auto& sample : *test_set){
			auto X_test = sample.data.reshape({n_test,d}).to(options_double);
			auto Y_test = sample.target.to(options_int);
			double error = error_rate(Y_test,neuralnet.predict(X_test));
			cout << error << endl;
			file << error << endl;
		}
}

