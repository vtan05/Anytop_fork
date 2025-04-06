# AnyTop: Character Animation Diffusion with Any Topology

The official PyTorch implementation of the paper [**"AnyTop: Character Animation Diffusion with Any Topology"**]().

Please visit our [**webpage**](https://anytop2025.github.io/Anytop-page/) for more details.

![teaser](https://github.com/Anytop2025/Anytop-page/blob/main/static/videos/anytop_teaser/teaser.gif)


## Release Timeline
✅  May 6, 2025 – Training & inference code & preprocessing code

📌 April 12, 2025 – Pretrained models, processed dataset 

📌 April 19, 2025 – Motion editing & DIFT feature correspondence code

📌 May 10, 2025 – Evaluation & rendering with mesh code

## Getting started

This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment
Setup conda env:
```shell
conda env create -f environment.yaml
conda activate anytop
pip install git+https://github.com/inbar-2344/Motion.git
```

### 2. Download Truebones dataset

Parse Truebone data yourself, using out prepocessing script:
(a) Download the full dataset [here](https://truebones.gumroad.com/l/skZMC) 
(b) Place Truebone_Z-OO dirctory in our repository, under ./datasets/
(c) Execute the following 
```shell
python -m data_loaders/truebones/truebones_utils/create_dataset.py
```

## Motion Synthesis

### Generate motion for skeleton from Truebones dataset
We categorize Truebones skeletons into five subsets: Bipeds, Quadrupeds, Millipeds, Snakes, and Flying Creatures.
In addition to a unified model trained on the entire dataset, we also trained specialized models for each subset (with Millipeds and Snakes grouped together).

The explicit skeleton-to-subset mapping is defined in
./data_loaders/truebones/truebones_utils/param_utils.py
(see the lists: BIPEDS, QUADROPEDS, FLYING, and MILLIPEDS+SNAKES).

To generate motion for a specific skeleton (object_type) from the Truebones dataset, run:

```shell
python -m sample.generate  --model_path <save/model_name/model_checkpoint.pt> --object_type <object_type> --num_repetitions 3
```

### Generate unseen skeleton outside of Truebones dataset
We support motion synthesis for skeletons outside of Truebones dataset, provided as bvh file/s. 
To do that, you must first run our pre-processing pipeline on the new skeleton to create cond.py file for the skeleton, as described in 
BVH to Skeleton section below. conce you've accomplish this part, you can synthesize motion of the new skeleton by running the command:

```shell
python -m sample.generate  --model_path <model_path> --object_type <skeleton_name> --num_repetitions 3 --cond_path <path_to_cond_npy_file>
```

**You may also define:**
* `--device` id.
* `--seed` to sample different prompts.
* `--motion_length` (text-to-motion only) in seconds (maximum is 9.8[sec]).

**Running those will get you:**

* `<object_type>_rep_<rep_id>_#<sample_id>.npy` file with xyz positions of the generated animation
* `<object_type>_rep_<rep_id>_#<sample_id>.mp4` a stick figure animation for each generated motion
* `<object_type>_rep_<rep_id>_#<sample_id>.npy` bvh file of the generated motion

It will look something like this:

![example]( assets/smaller_stick_fig.gif )

## Train AnyTop 

To reproduce the unified paper model, run:
```shell
python -m train.train_anytop --model_prefix all --objects_subset all --lambda_geo 1.0 --overwrite --balanced
```

To reproduce the bipeds paper model, run:
```shell
python -m train.train_anytop --model_prefix bipeds --objects_subset bipeds --lambda_geo 1.0 --overwrite --balanced
```

To reproduce the quadropeds paper model, run:
```shell
python -m train.train_anytop --model_prefix quadropeds --objects_subset quadropeds --lambda_geo 1.0 --overwrite --balanced
```
To reproduce the millipeds paper model, run:
```shell
python -m train.train_anytop --model_prefix millipeds_snakes --objects_subset millipeds_snakes --lambda_geo 1.0 --overwrite --balanced
```

To reproduce the flying animals paper model, run:
```shell
python -m train.train_anytop --model_prefix flying --objects_subset flying --lambda_geo 1.0 --overwrite --gen_during_training --balanced
```
* **General instructions** Checkout './utils/parser_utils.py' to view all configurable parameters and default settings. '--balanced' flag is used to activate the balancing sampler, ensuring fair sampling of all skeletons. Use '--overwrite' flag if you wish to resume training of previous checkpoint. 
* **Recommended:** Add `--gen_during_training` to generate motions for each saved checkpoint. 
  This will slow down training but will give you better monitoring.
* **Recommended:** Add `--use_ema` for Exponential Moving Average to improve performance.
* Use `--diffusion_steps 50` to train the faster model with less diffusion steps.
* Use `--device` to define GPU id.
* Add `--train_platform_type {WandBPlatform, TensorboardPlatform}` to track results with either [WandB](https://wandb.ai/site/) or [Tensorboard](https://www.tensorflow.org/tensorboard).


## Process BVH out of Truebones dataset
We provide a preprocessing code for skeletons outside the Truebones dataset. 
While designed to be as generic as possible, some skeleton-specific adjustments may be needed since it 
was originally tailored for Truebones. For example, it relies on joint names for foot classification 
and specific velocity/height thresholds for foot contact detection. However, we have tested it on BVH 
files from Mixamo and other sources to ensure its generalizability.

Input Arguments:
object_name - A character's indicative name (e.g., "Dog").
bvh_dir - Directory containing BVH files of the skeleton. More files improve statistical accuracy for motion denormalization.
face_joints - Four joints defining skeleton orientation ([right hip, left hip, right shoulder, left shoulder] or equivalent). 
            Used to align the skeleton to Z+ and XZ plane.
save_dir - Output directory.
tpos_bvh - A BVH file of the character's natural rest pose for meaningful rotation learning. 
        If missing, the code selects a pose from the provided BVH files. 

Finally, you can run the command: 

```shell
python -m data_loaders/truebones/truebones_utils/process_new_skeleton --object_name <skeleton_name> --bvh_dir <path_to_bvhs_dir> --save_dir <save_dir> --face_joints_names [right_hip_joint, left_hip_joint, right_shoulder_joint, left_shoulder_joint] --tpos_bvh <tpos_path>
        
```       
Output:
The code will create the following under save_dir:
save_dir/
        |_motions
        |_animations
        |_bvhs
        cond.npy
1. In motions directory, you will find npy files, which are the processed motion features of each bvh file. 
This is useful in case you would like to use this data for training. 
2. In animation directory, you will find mp4 files corresponding to each of the processed bvhs. 
This is a good sanity check that everything worked as expected. 
Note that face_joints are marked in blue and feet joints are marked in green.
3.In bvhs dir you can find the processed bvhs
4. cond.npy contains the skeletons representation, including joints names ambeddings and graph conditions,

## Acknowledgments
We want to thank the following contributors that our code is based on:
[mdm](https://github.com/GuyTevet/motion-diffusion-model), [GRPE](https://github.com/lenscloth/GRPE/tree/master), [audiocraft](https://github.com/facebookresearch/audiocraft)

## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries that have their own respective licenses that must also be followed.
