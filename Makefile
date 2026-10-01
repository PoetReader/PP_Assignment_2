# Number of processes:
N=6
# Change this as you like:
MPIEXEC_OPTS=--oversubscribe

main: main.c lib/libbmp/libbmp.c lib/bmp.c
	mpicc -o $@ $^ -lm

run: main
	mpiexec -n $(N) $(MPIEXEC_OPTS) ./main

clean:
	@rm main

.PHONY: clean
