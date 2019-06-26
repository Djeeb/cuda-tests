#include "nnet_from_scratch.hpp"
using namespace std;

int main(){
	cout << "--- RÉSEAU DE NEURONES À UNE COUCHE ENTRAÎNÉ SUR MNIST ---" << endl;
	cout << "\n* Initialisation ..." << endl;
	
//paramètres d'initialisation
	int epochs = 100;
	int training_size = 60000;
	int batch_size = 6000;
	torch::DeviceType device_type = torch::kCPU;
	nnet neuralnet(784,64,10,batch_size,0.01,device_type);
	torch::Tensor Y_train,Y_hat,Y_test;
	torch::Tensor X_train,X_test;
	
//chargement de MNIST
	auto train_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data").map(
                     torch::data::transforms::Stack<>()),batch_size);
                       
	auto test_set = torch::data::make_data_loader(
                     torch::data::datasets::MNIST("../../data",torch::data::datasets::MNIST::Mode::kTest).map(
                     torch::data::transforms::Stack<>()),10000);


//Phase d'apprentissage'
	cout << "\n* Début de la phase d'apprentissage ("<< epochs << "  epochs) :" << endl;
	auto t1 = chrono::system_clock::now();
	for(int i=0;i<epochs;i++){
		for(auto& sample : *train_set){

			X_train = sample.data.reshape({batch_size,784}).to(torch::TensorOptions().dtype(torch::kFloat64).device(device_type)).transpose(0,1);
			Y_train = at::one_hot(sample.target,10).transpose(0,1).to(torch::TensorOptions().dtype(torch::kFloat64).device(device_type));			
			
			neuralnet.forward( X_train );
			neuralnet.compute_cost( Y_train );
			neuralnet.backward( X_train, Y_train);
			neuralnet.update();
		}
		cout << "- epoch " << i+1 << ": \t loss = " << neuralnet.reset_cost(training_size) << endl;
	}
	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	cout << "Phase d'apprentissage terminée en " << diff.count() << " sec" << endl;
	
	
//Phase de test
	cout << "\n* Précision sur le test set : ";
	for(auto& sample : *test_set){
			X_test = sample.data.reshape({10000,784}).to(torch::TensorOptions().dtype(torch::kFloat64).requires_grad(false)).transpose(0,1);
			Y_test = sample.target.to(torch::TensorOptions().dtype(torch::kInt64).requires_grad(false));
			Y_hat = neuralnet.predict(X_test);
			cout << error_rate(Y_test,Y_hat) << endl;
			
		}
}
