#include "mshadow/tensor.h"
#include <vector>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

class SimpleNeuralNet {
	private:
		// nodes in neural net
		TensorContainer<gpu, 2, double> ninput, nhidden, nhiddenbak, nout;
		// hidden bias, gradient
		TensorContainer<gpu, 1, double> hbias, obias, g_hbias, g_obias;
		// weight gradient
		TensorContainer<gpu, 2, double> Wi2h, Wh2o, g_Wi2h, g_Wh2o;

	
	public:
		SimpleNeuralNet(int batch_size, int n_input, int n_hidden, int n_output);
		~SimpleNeuralNet() {};
		
		void Forward(const Tensor<cpu, 2, real_t>& inbatch, Tensor<cpu, 2, real_t> &oubatch);
		void Backprop(const Tensor<cpu, 2, real_t>& gradout);
		void Update();
};

