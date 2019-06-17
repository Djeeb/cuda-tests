CC = nvcc -ccbin g++
CXX = g++
CFLAGS = -std=c++11 -O2 -I ../mshadow -lm -lcblas -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -Wno-unused-local-typedefs -msse3 -mf16c -DMSHADOW_USE_CUDA=1 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -DMSHADOW_RABIT_PS=0 -DMSHADOW_DIST_PS=0 -DMSHADOW_USE_PASCAL=0

all : simple_allocator lazy_exp syntax defop

simple_allocator: simple_allocator.o
	$(CC) -o simple_allocator simple_allocator.o $(CFLAGS)
	
simple_allocator.o: simple_allocator.cu
	$(CC) -c simple_allocator.cu $(CFLAGS)

lazy_exp: lazy_exp.o
	$(CC) -o lazy_exp lazy_exp.o $(CFLAGS)
	
lazy_exp.o: lazy_exp.cu
	$(CC) -c lazy_exp.cu $(CFLAGS)
	
syntax: syntax.o
	$(CC) -o syntax syntax.o $(CFLAGS)
	
syntax.o: syntax.cu
	$(CC) -c syntax.cu $(CFLAGS)

defop: defop.o
	$(CC) -o defop defop.o $(CFLAGS)
	
defop.o: defop.cu
	$(CC) -c defop.cu $(CFLAGS)

clean:
	rm *.o
	rm simple_allocator lazy_exp syntax defop
