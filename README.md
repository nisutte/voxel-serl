# Voxel SERL - real-world reinforcement learning in the context of vacuum gripping

**Webpage: [nisutte.github.io](https://nisutte.github.io/)** \
**Paper: [arxiv.org/abs/2503.02405](https://arxiv.org/abs/2503.02405)**

<p style="display: flex; align-items: center;">
  <img src="./docs/images/box_front.jpg" height="250pt" style="margin-right: 20pt"/>
  <img src="./docs/images/Voxel_grid_example_slim.png" height="180pt"/>
</p>
  
Voxel SERL builds upon the original [SERL](https://github.com/rail-berkeley/serl] implementation)  implementation by incorporating additional modalities into the reinforcement learning pipeline. It utilizes 3D spatial perception to improve the robustness of real-world vacuum gripping.

### Getting started

Prerequisites:
- NVIDIA driver and CUDA/cuDNN matching your target JAX build (only if using GPU)
- uv installed (recommended): see `https://docs.astral.sh/uv/` or install on Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # install uv
uv sync                                             # environment setup
source .venv/bin/activate                           # environment activation (not necessarily needed)
```

### Switching CUDA versions (JAX/Flax)

By default this project uses CUDA 12 builds via `jax[cuda12]` (cudnn 8/12 toolchain) as specified in `pyproject.toml`:

```toml
# pyproject.toml → [project].dependencies (default)
"jax[cuda12]==0.4.25",
```

To switch to CUDA 11 builds, change the extra to `cuda11` and re-sync.
For CPU-only installs, you can use `"jax==0.4.25"` (no CUDA extra) with the same `jaxlib` version and re-run `uv sync`.

## Box picking

### Contributions

| Code Directory                                                                                             | Description                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [robot_controllers](https://github.com/nisutte/voxel-serl/tree/develop/serl_robot_infra/robot_controllers) | Impedance controller for the UR5 robot arm |
| [box_picking_env](https://github.com/nisutte/voxel-serl/tree/develop/serl_robot_infra/box_picking_env)     | Environment setup for the box picking task |
| [vision](https://github.com/nisutte/voxel-serl/tree/develop/serl_launcher/serl_launcher/vision)            | Point-Cloud based encoders                 |
| [utils](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/camera/utils.py)        | Point-Cloud fusion and voxelization        |

### Quick start guide for box picking with a UR5 robot arm

#### Without cameras

1. Follow the installation in the official [SERL repo](https://github.com/rail-berkeley/serl).
2. Check [envs](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/envs) and either use the provided [box_picking_env](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/envs/camera_env/box_picking_camera_env.py) or set up a new environment using the one mentioned as a template. (New environments have to be registered [here](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/__init__.py))
2. Use the [config](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/envs/camera_env/config.py) file to configure all the robot-arm specific parameters, as well as gripper and camera infos.
3. Go to the [box picking](https://github.com/nisutte/voxel-serl/blob/develop/examples/box_picking_drq) folder and modify the bash files ```run_learner.py``` and ```run_actor.py```. If no images are used, set ```camera_mode``` to ```none``` . WandB logging can be deactivated if ```debug``` is set to True.
4. Record 20 demostrations using [record_demo.py](https://github.com/nisutte/voxel-serl/blob/develop/examples/box_picking_drq/record_demo.py) in the same folder. Double check that the ```camera_mode``` and all environment-wrappers are identical to [drq_policy.py](https://github.com/nisutte/voxel-serl/blob/develop/examples/box_picking_drq/drq_policy.py).
5. Execute ```run_learner.py``` and ```run_actor.py``` simultaneously to start the RL training.
6. To evaluate on a policy, modify and execute ```run_evaluation.py``` with the specified checkpoint path and step. 

### Modaliy examples
<p>
  <img src="./docs/images/trajectory%20timeline.png" width="30%"/>
</p>


## Box handover (multi robot)

<p>
  <img src="./docs/images/box_handover.png" width="60%"/>
</p>

### Information

In this setup, two UR5 robotic arms are positioned facing each other to perform a box handover using suction grippers. Each arm is equipped with a wrist-mounted D405 camera that supplies voxelized, localized point cloud data to the RL pipeline. The episode begins after a scripted box pickup, which is handled during the environment reset. Both robots are jointly controlled by a single RL policy operating in a 14-dimensional action space. Safety mechanisms automatically detect and handle collisions if the arms move too close together or if excessive forces are detected during the handover.

- The Onshape design file for the D405 camera holder is available [here](https://cad.onshape.com/documents/adfbc29573a3362dcbc21dfa/w/157f299ac48f41d6b5d551f9/e/94e430388f5379a0342fc4b5?renderMode=0&uiState=68b060cb94d7fa50e8615103).

### Changes

| File | Description |
|-----------------------------------------------|--------------------------------------------------------------|
| [controller_client.py](serl_robot_infra/robot_controllers/controller_client.py) | New controller written in C++ (old one can still be used) |
| [dual_ur5_env.py](serl_robot_infra/ur_env/envs/dual_ur5_env.py) | Dual robot env core; relative frames/velocities, start poses, BT pickup; 14D action, 80+ D obs incl EE-to-EE relative pose. |
| [box_handover_env.py](serl_robot_infra/ur_env/envs/handover_env/box_handover_env.py) | Handover logic, safety, drop-prevention; integrates collision and terminations. |
| [config.py](serl_robot_infra/ur_env/envs/handover_env/config.py) | Handover parameters and variants; 90° handover wip. |
| [threaded_collision_detection.py](serl_robot_infra/ur_env/utils/threaded_collision_detection.py) | Threaded collision checks incl suction cup; used for safety/termination. |
| [voxel_grid_encoders.py](serl_launcher/serl_launcher/vision/voxel_grid_encoders.py) | VoxNet backbone (shared; freeze/unfreeze supported). |
| [observation_statistics_wrapper.py](serl_launcher/serl_launcher/wrappers/observation_statistics_wrapper.py) | Observation stats and normalization (RLDS-derived). |
| [relative_env.py](serl_robot_infra/ur_env/envs/relative_env.py) | Relative frames/velocities; EMAs for force and velocity. |
| [drq.py](serl_launcher/serl_launcher/agents/continuous/drq.py) | DRQ: true critic ensemble, ReLU, narrower tanh std, noise aug, grad-norm logging. |
| [encoding.py](serl_launcher/serl_launcher/common/encoding.py) | Encoder wiring incl VoxNet and actor/critic inputs. |
| [data_store.py](serl_launcher/serl_launcher/data/data_store.py) | Async RLDS logging; end-of-episode flush; consistency checks. |


Calibration: For robot-to-robot (base-to-base) calibration, I used my supervisor's repository (which is not public unfortunately). The process involves using charuco markers attached to the end effector and performing hand-eye calibration for both robots. Ultimately, the goal is to obtain the transformation matrix `T_robotBaseLeft2robotBaseRight`, saved as a `.npy` file.

### Environment details (DualUR5Env)

- **Action space**: 14D (7-DoF pose+grip per arm; concatenated left||right).
- **Observation (state)**: basic keys and shapes below. Images (if enabled) are passed through as `left/*`, `right/*`.

| Key | Shape | Notes |
|----------------------|-------|-----------------------------------------------|
| left/tcp_pose | 6 | TCP pose (xyz, MRP) |
| left/tcp_vel | 6 | Linear xyz, angular MRP |
| left/gripper_state | 2 | Left gripper object detection and pressure |
| left/tcp_force | 3 | Force at left TCP |
| left/tcp_torque | 3 | Torque at left TCP |
| left/action | 7 | Last applied action for left arm |
| right/tcp_pose | 6 |  |
| right/tcp_vel | 6 |  |
| right/gripper_state | 2 |  |
| right/tcp_force | 3 |  |
| right/tcp_torque | 3 |  |
| right/action | 7 |  |
| l2r/tcp_pose | 6 | Left-EE to Right-EE relative pose |
| l2r/tcp_vel | 6 | Relative velocity (in left EE frame) |
| r2l/tcp_pose | 6 | Right-EE to Left-EE relative pose |
| r2l/tcp_vel | 6 | Relative velocity (in right EE frame) |

Notes:
- Pose representation is quaternion by default; with `DualToMrpWrapper` it becomes 6D (xyz + MRP). Relative angular rates are mapped accordingly.
- Observation normalization (means/stds) is computed from RLDS and applied via wrappers.

### TODO list
<details>
<summary>(click to expand)</summary>

- [x] Make dual robot pointcloud work
- [x] set up BT to grip a box at the start
- [x] set up start poses
- [x] come up with a good reward
- [x] collision detection between robots
- [x] add difference parameters between the EE
- [x] share the same voxnet backbone (less gpu mem, also can be frozen)
- [x] add seconds passed since last action to the state
- [x] improve camera frame polling (sometimes very slow) 
  - camera lag time is ~60ms, quite high but i cannot do much about it
- [x] actions in the obs are transformed to base frame, we do not want that (so action input is different from action obs)
- [x] do some demos and first training runs
- [x] ADD MAX FORCE
- [x] add huge negative penalty for dropping box
- [x] fix observation statistics wrapper
- [x] I have to fix the "End of file" bug in the controller, otherwise i make no progress
- [x] Do propper normalization (over rlds dataset)
- [x] remodel to immediate reward (on the chosen actions)
- [x] make it impossible to drop the box, not just huge reward (policy is dumb)
- [x] make RLDS save the pointclouds as well, such that i can post-train the VoxNet with the data captured
- [x] make a data consistency checker for the replay buffer!
- [x] clean up relative env mess (once again, sigh...)
- [x] make training more stable and add more useful obs
  - [x] added state augmentation (noise)
  - [x] changed activation to relu
  - [x] made tanh distribution narrower (less noisy actions, less jitter)
  - [x] really ensemblize critic!
  - [x] update the voxnet with the new relu and LN 
- [x] Make a simple uv setup for future usage (from requirements, and also add external JAX links, tough...)
- [x] examine ensemble sizes and subsampling
- [x] make position augmentation for pose data (random noise)
- [ ] train only on pc data, like in the picking task
- [ ] Add pose estimation to automate the pickup

</details>


## Future work

- Automate the pickup by integrating pose estimation of the boxes
- Implement PointNet in JAX for the backbone, compare it to VoxNet
- Add simulation pretraining to the repo, to speed up training in the real world.