#include <torch/torch.h>
#include <iostream>

using namespace std;

int main(){
cout << "\n--- TEST DE BACKWARD() et de l'autograd ---\n";

//exemple basique sur des réels
	torch::Tensor x = torch::rand({1,1},torch::TensorOptions().requires_grad(true));
	torch::Tensor b = torch::ones({1,1},torch::TensorOptions().requires_grad(false));
	
	//Calcul de la fonction puis du backward grâce à l'autograd
	auto fx = x*x + b;
	fx.backward();
	
	cout << "\n- EXEMPLE 1 - réels: f(x) = x*x + b";
	cout << "\n\tx.grad() = " << x.grad().item<double>() << endl;
	cout << "\t2*x = " << 2*x.item<double>() << endl;

cout << "\n------------------------------------------------------------------\n";
//exemple matriciel plus poussé
	torch::Tensor X = torch::rand({3,1},torch::TensorOptions().requires_grad(true));
	torch::Tensor W = torch::ones({1,3},torch::TensorOptions().requires_grad(false));
	torch::Tensor B = torch::ones({1,1},torch::TensorOptions().requires_grad(false));
	
	auto A = W.mm(X) + B;
	A.backward();
	cout << "\n- EXEMPLE 2 - vecteurs : f(x) = <W,X> + B";
	cout << "\n\tx.grad() = \n" << X.grad() << endl;
	cout << "\t2*Wx = \n" << W << endl;	

cout << "\n------------------------------------------------------------------\n";
	X = torch::rand({5,1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
	W = torch::ones({1,5}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
	torch::Tensor bias = torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
	
	auto weights = W.accessor<float,2>();
	for(int i=0;i<5;i++) weights[0][i] = float(i+1);
	
	auto f = W.mm(X) + bias;
	auto g = at::sigmoid(f);
	g.backward();

	cout << "\n- EXEMPLE 3 - fonction d'activation : g(x) = sigmoid( f(x) ) et f(x) = <w,x> + b";
	cout << "\n\tx.grad() = \n" << X.grad() << endl;
	cout << "\tdg/dx = (W*exp(- (W.mm(X) + bias)))/(g(x)^2) = \n" << (W*exp(- (W.mm(X) + bias)))/(g*g) << endl;
	


}
