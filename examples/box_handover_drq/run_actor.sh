export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
uv run /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --actor \
    --exp_name="Handover V10 random reset" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 20000 \
    --utd_ratio 8 \
    --batch_size 64 \
    --eval_period 0 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --debug
