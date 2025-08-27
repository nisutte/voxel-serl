export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
uv run /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --actor \
    --exp_name="Handover V10" \
    --camera_mode pointcloud \
    --batch_size 64 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints Handover V10 random reset 0821-16:58"\
    --eval_checkpoint_step 11000 \
    --eval_n_trajs 20 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --enable_temporal_ensemble_sampling True \
    --debug
