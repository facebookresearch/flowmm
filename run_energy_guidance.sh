#!/bin/bash 
source ~/.bashrc
mamba activate flowmm

cd ~/projects/osda_inpaint/flowmm

# training 

python scripts_model/run_docking.py

# inference 

chkpt=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-11-16/14-43-01/docking_only_coords-dock_cspnet-xim0gv6y/rfmcsp-unconditional-docking/g0obsdls/checkpoints
python scripts_model/evaluate_docking.py gen_trajectory $chkpt/epoch=0-step=2.ckpt --num_samples 2 --single_gpu --guidance_strength 5.0
# output at $chkpt/gen_trajectory_00 (see evaluate_docking for how to name the folder)


# Paused at finding a way to analyze the outputs 