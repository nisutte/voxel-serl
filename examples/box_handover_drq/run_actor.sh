export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --actor \
    --exp_name="Handover V1" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 20000 \
    --random_steps 0 \
    --training_starts 400 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 0 \
    \
    --encoder_type pretrained-voxnet \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --debug
