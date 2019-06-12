CC = nvcc
CFLAGS = -ccbin g++ -std=c++11 -O2 -I ../mshadow -lm -lcblas

simple_allocator: simple_allocator.o
	$(CC) -o simple_allocator simple_allocator.o $(CFLAGS)
	
simple_allocator.o: simple_allocator.cu
	$(CC) -c simple_allocator.cu $(CFLAGS)

clean:
	rm simple_allocator.o
	rm simple_allocator
