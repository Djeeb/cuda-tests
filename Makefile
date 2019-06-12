CC = nvcc
CFLAGS = -ccbin g++ -std=c++11 -O2

pi-simulation: pi-simulation.o
	$(CC) -o pi-simulation pi-simulation.o $(CFLAGS)
	
pi-simulation.o: pi-simulation.cu
	$(CC) -c pi-simulation.cu $(CFLAGS)

clean:
	rm pi-simulation.o
	rm pi-simulation
