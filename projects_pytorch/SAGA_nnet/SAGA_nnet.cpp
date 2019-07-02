#include "SAGA_nnet.hpp"
using namespace std;

int main(){
	
//______________________________Initializing neural network and optimize
	int epochs = 20;
	int batch_size = 1;
	int training_size = 10000;
	nnet neuralnet(training_size,batch_size,784,16,10,0.001,"CPU","SGD");
	torch::optim::SGD optimizer(neuralnet.parameters(), 0.01);	
	
		
	torch::Tensor X_train, Y_train, loss, X_test, Y_test;

//_________________________________________________________Loading MNIST
	auto train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),batch_size);
                
                     
//_____________________________________________________Running algorithm
int k = 0; //sample index
auto t1 = chrono::system_clock::now();


	for(int i=0; i < epochs;i++){
		k = 0;
		
		for(auto& sample : *train_set){
			
			//Setting optimizer to zero grad
			optimizer.zero_grad();
			
			//Loading new examples
			X_train = sample.data.reshape({batch_size,784}).to(options_double);		
			Y_train = at::one_hot(sample.target,10).to(options_double);
				
			//Forward propagation
			X_train = neuralnet.forward( X_train );
			
			//Compute loss function
			loss = neuralnet.cross_entropy_loss( X_train, Y_train );
			//loss = torch::mse_loss( X_train , sample.target );			
			
			//Back-propagation
			loss.backward();
			
			//update
			neuralnet.update(i,k);
			//optimizer.step();
			k++;
			if(k==9999) break;
		}	
	cout << "Epoch: " << i+1 << "\t | Loss: " << neuralnet.compute_cost() << endl;
	}

auto t2 = chrono::system_clock::now();
chrono::duration<double> diff = t2 - t1;
cout << "Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
	
//______________________________________________________Evaluating model
	auto test_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data",torch::data::datasets::MNIST::Mode::kTest).map(
                     torch::data::transforms::Stack<>()),10000);
                     
	cout << "\n* Précision sur le test set : ";
	for(auto& sample : *test_set){
			X_test = sample.data.reshape({10000,784}).to(options_double);
			Y_test = sample.target.to(options_int);
			cout << error_rate(Y_test,neuralnet.predict(X_test)) << endl;
		}
		
}
