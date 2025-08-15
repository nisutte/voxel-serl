export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --learner \
    --exp_name="Handover V9" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 25000 \
    --utd_ratio 8 \
    --batch_size 64 \
    --checkpoint_period 1000 \
    --log_rlds_path /home/nico/real-world-rl/serl/examples/box_handover_drq/rlds \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_handover_drq/box_picking_10_demos_aug15.pkl \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 64 \
    --debug
#    --checkpoint_preload_path "/home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints Handover V7 adapt voxel 0731-16:10/" \
#    --checkpoint_preload_step 23000 \
