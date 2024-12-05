#!/bin/bash 
source ~/.bashrc
mamba activate flowmm

cd ~/projects/osda_inpaint/flowmm

### training #################################################################################

# python scripts_model/run_docking.py

### inference #################################################################################

# expt 6
op_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-57-54/docking_only_coords-dock_cspnet-n21yhyok/
chkpt=$op_dir/rfmcsp-unconditional-docking/pgyf0wtt/checkpoints/epoch=1-step=2.ckpt

# op_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-01/10-45-56/docking_only_coords-dock_cspnet-54r5z7m0/
# chkpt=$op_dir/rfmcsp-unconditional-docking/p69qlmet/checkpoints/epoch=0-step=2.ckpt
# output at $chkpt/gen_trajectory_00 or wherever(see evaluate_docking for how to name the folder)
# - "reconstruct", "recon_trajectory", "generate", "gen_trajectory", "pred" 

# python scripts_model/evaluate_docking.py gen_trajectory $chkpt --num_samples 2 --single_gpu --guidance_strength 5.0
# python scripts_model/evaluate_docking.py recon_trajectory $chkpt --single_gpu # OOM error TODO move to supercloud

### visualization #################################################################################

# python src/rfm_docking/visualization.py --traj_file $chkpt/recon_trajectory_00/predictions_00.pt --output_gif $chkpt/recon_trajectory_00/predictions_00.gif

###################################################################################################

# expt 6
# traj_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-57-54/docking_only_coords-dock_cspnet-n21yhyok/ # for loop see expt 4
# or 
traj_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-57-54/docking_only_coords-dock_cspnet-n21yhyok/rfmcsp-unconditional-docking/pgyf0wtt/checkpoints/recon_trajectory_00

for structure_type in osda_pred_and_none_target none_pred_and_osda_target; do 
    python src/rfm_docking/visualization.py --traj_file $traj_dir/predictions_00.pt --output_gif $traj_dir/predictions_00_$structure_type.gif --structure_type $structure_type
done

# expt 5
# traj_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-48-23/docking_only_coords-dock_cspnet-ukkuvn7o/

# expt 4 
# traj_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-39-15/docking_only_coords-dock_cspnet-3xbxcwn6
# for traj in 414874138  536351841  536351847  558245058  536293713  536351842  536351850 536351838  536351844  536351851; do 
#     python src/rfm_docking/visualization.py --traj_file $traj_dir/${traj}_traj.pt --output_gif $traj_dir/traj_to_gifs/${traj}_traj.gif
# done

# expt 3 
# traj_dir=//home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-24-07/docking_only_coords-dock_cspnet-0nert963
# for traj in 414874138  536351841  536351847  558245058  536293713  536351842  536351850 536351838  536351844  536351851; do 
#     python src/rfm_docking/visualization.py --traj_file $traj_dir/${traj}_traj.pt --output_gif $traj_dir/traj_to_gifs/${traj}_traj.gif
# done

# expt 2
# op_dir=//home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/22-16-20/docking_only_coords-dock_cspnet-jew8o0mg
# python src/rfm_docking/visualization.py --traj_file $op_dir/414874133_traj.pt --output_gif $op_dir/414874133_traj.gif

# expt 1 
# op_dir=//home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/21-12-02/docking_only_coords-dock_cspnet-6hr50554
# python src/rfm_docking/visualization.py --traj_file $op_dir/414874133_traj.pt --output_gif $op_dir/414874133_traj.gif


# traj_dir=/home/mrx/projects/osda_inpaint/flowmm/runs/trash/2024-12-03/20-56-04/docking_only_coords-dock_cspnet-53twxpeb/
# for traj in 414874138  536351841  536351847  558245058  536293713  536351842  536351850 536351838  536351844  536351851; do 
#     python src/rfm_docking/visualization.py --traj_file $traj_dir/${traj}_traj.pt --output_gif $traj_dir/traj_to_gifs/${traj}_traj.gif
# done


### NOTES ##########################################################################################

# visualization is not coded to return the zeolites as of now.