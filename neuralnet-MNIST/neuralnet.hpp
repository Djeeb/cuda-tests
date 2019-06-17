#include "mshadow/tensor.h"
#include <vector>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

class SimpleNeuralNet {
	private:
		// random seed generator
		Random<gpu, double> Rand;
		// nodes in neural net
		Tensor<gpu, 2, double> layer_input, layer_hidden, layer_hiddenbak, layer_out;
		// hidden bias, gradient
		Tensor<gpu, 1, double> hbias, obias, g_hbias, g_obias;
		// weight gradient
		Tensor<gpu, 2, double> Wi2h, Wh2o, g_Wi2h, g_Wh2o;

	
	public:
		SimpleNeuralNet(int, int, int, int);
		~SimpleNeuralNet() {};
		
		void Forward(const Tensor<cpu, 2, double >& inbatch, Tensor<cpu, 2, double > &oubatch);
		void Backprop(const Tensor<cpu, 2, double >& gradout);
		void Update();
};



//Constructeur
SimpleNeuralNet::SimpleNeuralNet(int batch_size, int n_input, int n_hidden, int n_output): Rand(0) {
	
	// Paramétrage d'un stream sur tous les objets qui vont être modifiés lors de la tâche
	Stream<gpu> * stream = NewStream<gpu>(0);
	layer_input.set_stream(stream);
    layer_hidden.set_stream(stream);
    layer_hiddenbak.set_stream(stream);
    layer_out.set_stream(stream);
    hbias.set_stream(stream);
    g_hbias.set_stream(stream);
    g_obias.set_stream(stream);
    obias.set_stream(stream);
    Wi2h.set_stream(stream);
    Wh2o.set_stream(stream);
    g_Wi2h.set_stream(stream);
    g_Wh2o.set_stream(stream);
	
	// Redimension des couches
    layer_input.shape_ = Shape2(batch_size, n_input);
    layer_hidden.shape_ = Shape2(batch_size, n_hidden);
    layer_hiddenbak.shape_ = Shape2(batch_size, n_hidden);
    layer_out.shape_ = Shape2(batch_size, n_output);
    
	// Redimension des biais
    hbias.shape_ = Shape1(n_hidden); 
    g_hbias.shape_ = Shape1(n_hidden);
    obias.shape_ = Shape1(n_output); 
    g_obias.shape_ = Shape1(n_output);
	
	// Redimension des poids
	Wi2h.shape_ = Shape2(n_input, n_hidden);  
	g_Wi2h.shape_ = Shape2(n_input, n_hidden);  
	Wh2o.shape_ = Shape2(n_hidden, n_output); 
	g_Wh2o.shape_ = Shape2(n_hidden, n_output); 
	
	// Initialisation aléatoire des poids
	Rand.SampleGaussian(&Wi2h, 0.0f, 0.01f);
    Rand.SampleGaussian(&Wh2o, 0.0f, 0.01f);
	
}

