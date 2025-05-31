# TrueBones evaluation:

## Environment Setup:
Install dependencies using conda and pip:
```
conda install -c conda-forge eigen
pip install git+https://github.com/PeizhuoLi/ganimator-eval-kernel.git
pip install pytorch3d
```

## Running the Evaluation
```
python -m eval.eval_truebones --eval_gt_dir <gt_motion_dir> --eval_gen_dir <generated_motion_dir> --benchmark_path <benchmark_path> 

```
### Arguments Description
* `--eval_gt_dir` Path to the ground truth motions directory (NumPy .npy files).
Default: dataset/truebones/zoo/truebones_processed/motions/
* `--eval_gen_dir` Path to the directory containing generated motions (NumPy .npy files).
Note: To reproduce paper results, each character in the benchmark should have 20 motions, each 120 frames long.
* `--benchmark_path` Path to a text file listing character names to evaluate.
For the full AnyToP benchmark, use: eval/benchmarks/benchmark_all.txt. For specific subsets, use: eval/benchmarks/benchmark_<subset>.txt, 
where <subset> is one of: bipeds, quadropeds, millipeds_snakes, or flying.