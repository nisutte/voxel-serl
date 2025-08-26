export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
uv run /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --learner \
    --exp_name="Handover V10 random reset" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 50000 \
    --utd_ratio 8 \
    --batch_size 64 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_handover_drq/box_picking_10_demos_aug15.pkl \
    --checkpoint_preload_path "/home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints Handover V10 real ensemblize 0821-14:33" \
    --checkpoint_preload_step 10000 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --debug
#    --log_rlds_path /home/nico/real-world-rl/serl/examples/box_handover_drq/rlds \
