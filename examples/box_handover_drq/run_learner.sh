export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python /home/nico/real-world-rl/serl/examples/box_handover_drq/drq_policy.py "$@" \
    --learner \
    --exp_name="Handover V3" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 0 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 64 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/nico/real-world-rl/serl/examples/box_handover_drq/checkpoints \
    --demo_path /home/nico/real-world-rl/serl/examples/box_handover_drq/box_picking_6_demos_jun27.pkl \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
#    --debug