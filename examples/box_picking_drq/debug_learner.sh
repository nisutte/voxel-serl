export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
uv run drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name=drq_drq_policy \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 1000 \
    --random_steps 0 \
    --training_starts 0 \
    --utd_ratio 2 \
    --batch_size 32 \
    --eval_period 1000 \
    --checkpoint_period 5000 \
    --demo_path /home/nico/real-world-rl/serl/examples/box_picking_drq/box_picking_20_demos_aug15_1quat_action7.pkl \
    \
    --encoder_type voxnet \
    --state_mask no_ForceTorqueAction \
    --encoder_bottleneck_dim 128 \
    --proprio_latent_dim 16 \
    --enable_obs_rotation_wrapper \
    --debug
