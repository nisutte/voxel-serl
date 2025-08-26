export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
uv run /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --learner \
    --exp_name="Handover 90deg V11 ensemble 2" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 50000 \
    --utd_ratio 2 \
    --batch_size 128 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_handover_drq/box_picking_6_demos_90deg_aug26.pkl \
    --activate_90_degrees \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 64 \
#    --debug
#    --log_rlds_path /home/nico/real-world-rl/serl/examples/box_handover_drq/rlds \
#    --checkpoint_preload_path "/home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints Handover V10 real ensemblize 0821-14:33" \
#    --checkpoint_preload_step 10000 \