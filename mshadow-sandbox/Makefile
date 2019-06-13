CC = nvcc -ccbin g++
CXX = g++
CFLAGS = -std=c++11 -I ../mshadow -lm -lcblas

all : simple_allocator lazy_exp

simple_allocator: simple_allocator.o
	$(CC) -o simple_allocator simple_allocator.o $(CFLAGS)
	
simple_allocator.o: simple_allocator.cu
	$(CC) -c simple_allocator.cu $(CFLAGS)

lazy_exp: lazy_exp.o
	$(CC) -o lazy_exp lazy_exp.o $(CFLAGS)
	
lazy_exp.o: lazy_exp.cu
	$(CC) -c lazy_exp.cu $(CFLAGS)

clean:
	rm *.o
	rm simple_allocator lazy_exp
