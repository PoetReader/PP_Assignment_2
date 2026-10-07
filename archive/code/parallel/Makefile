# Number of processes:
N=6

#hostfile:
HOSTFILE=hostfile

# Number of threads:
T=1

# adress of image

IMG=bild.bmp
# Change this as you like:
MPIEXEC_OPTS=--oversubscribe

main: main.c lib/libbmp/libbmp.c lib/bmp.c
	mpicc -fopenmp -o $@ $^ -lm

opt_main: opt.c lib/libbmp/libbmp.c lib/bmp.c
	mpicc -fopenmp -o $@ $^ -lm

run: main
	mpiexec -n $(N) $(MPIEXEC_OPTS) ./main bild.bmp

cluster: main
	mpirun -np $(N) --hostfile $(HOSTFILE) --map-by ppr:1:node --bind-to none \
		--report-bindings -x OMP_NUM_THREADS=$(T) ./main $(IMG)

opt: opt_main
	mpirun -np $(N) --hostfile $(HOSTFILE) --map-by ppr:1:node --bind-to none \
		--report-bindings -x OMP_NUM_THREADS=$(T) ./opt?main $(IMG)

clean:
	@rm -f main sobelparallel.bmp

.PHONY: clean
