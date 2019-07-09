#include "SVRG_nnet.hpp"
#include "function_gen.hpp"
using namespace std;





int main(){
	//______________________________________________Initializing samples
	
	
	
	
	//_______________________________________Initializing neural network
		int epochs = 10;
		int batch_size = 1;
		int training_size = 10000;
		double learning_rate = 0.01;
		nnet neuralnet(training_size,batch_size,784,128,10,learning_rate,"GPU","SGD");
		torch::optim::SGD optimizer(neuralnet.parameters(), 0.01);	
	
	
	
	
}
