export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --actor \
    --exp_name="Handover V8" \
    --camera_mode pointcloud \
    --batch_size 64 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints Handover V7 adapt voxel 0731-16:10"\
    --eval_checkpoint_step 23000 \
    --eval_n_trajs 20 \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 64 \
    --enable_temporal_ensemble_sampling True \
    --debug
