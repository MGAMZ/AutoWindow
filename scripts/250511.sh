set +e

# torchrun --nproc-per-node 4 $mmrun 1.1.3.1
mmrun 1.1.3.1 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.2
mmrun 1.1.3.2 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.3
mmrun 1.1.3.3 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.4
mmrun 1.1.3.4 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.5
mmrun 1.1.3.5 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.6
mmrun 1.1.3.6 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.7
mmrun 1.1.3.7 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.8
mmrun 1.1.3.8 --test
torchrun --nproc-per-node 4 $mmrun 1.1.3.9
mmrun 1.1.3.9 --test
