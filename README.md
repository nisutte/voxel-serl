# Voxel SERL - real-world reinforcement learning in the context of vacuum gripping

## Handover Experiments

I investigated the gradients for both actor and critic and noticed that they are way too big (critic > 2e3). Since it scales with reward, i rescaled it, clipped the gradient norm and also changed the critic ensemble gradient computation from sum() to mean(), but I could not manage to get it working in time, I only have 1 day in my contract as of now. If someone continues this path:

- test utd ratios and ensemble sizes / subsampling sizes for the critic. Very important since we do not have that big of a replay buffer.
- the policy tended to be overconfident in my runs, i have not found the root cause.
- maybe look into different std parameterization for the policy, since they are capped for now (ugly). I tried it with tanh, but did not have enough time to test it fully.
- additionally, replacing VoxNet with a PointNet-based pc backbone would likely yield significant improvements.
