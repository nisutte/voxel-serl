export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
uv run /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --actor \
    --exp_name="Handover 90deg V11 ensemble 2" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 20000 \
    --utd_ratio 2 \
    --batch_size 128 \
    --eval_period 0 \
    --activate_90_degrees \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 64 \
#    --debug
