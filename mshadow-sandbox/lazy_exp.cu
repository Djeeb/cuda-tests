#include "mshadow/tensor.h"
#include <iostream>
#include <vector>
#include <random>
class Vector;

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

class Vector{
	public:
		int size;
		double * data;
		
		Vector(int n);
		double operator[](int i) const {return data[i];}
		double & operator[](int i) {return data[i];}
		
		friend Vector operator+(const Vector &, const Vector &);
			
};

Vector::Vector(int n){
	data = new double[n];
	for(int i=0; i<n; i++) data[i] = 0.; 
}

Vector Vector::operator+(const Vector & A, const Vector & B){
	Vector C(A.size);
	for(int i=0; i < C.size; i++) C[i] = A[i] + B[i];
	return C;
}


int main(void){
	
	//initialisation
	mt19937 G;
	uniform_real_distribution<double> U(-1.,1.);
	int n = 10000;
	Vector A(n), B(n), C(n),S(n);
	for(int i = 0; i < n; i++){
		A[i] = U(G);
		B[i] = U(G);
		C[i] = U(G);
	}
	
	//
	S = A + B;
	
}
