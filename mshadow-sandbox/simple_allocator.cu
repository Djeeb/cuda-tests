#include "mshadow/tensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	InitTensorEngine<gpu>();
	
	ShutdownTensorEngine<gpu>();
}
