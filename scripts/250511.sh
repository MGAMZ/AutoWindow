set +e
torchrun --nproc-per-node 4 $mmrun 1.1.3.0
torchrun --nproc-per-node 4 $mmrun 1.1.4.1
mmrun 1.1.3.0 1.1.4.1 --test