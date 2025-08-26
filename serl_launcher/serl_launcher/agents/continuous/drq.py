import copy
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import frozen_dict

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper, create_state_mask
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.vision.voxel_grid_encoders import MLPEncoder, VoxNet
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.vision.data_augmentations import (
    batched_random_crop,
    batched_random_shift_voxel,
    batched_random_rot90_action,
    batched_random_rot90_state,
    batched_random_rot90_voxel,
    add_gaussian_noise_state,
    build_std_vec_from_slices,
)
from ur_env.envs.handover_env.config import get_box_handover_state_noise_assignments


class DrQAgent(SACAgent):
    @classmethod
    def create(
            cls,
            rng: PRNGKey,
            observations: Data,
            actions: jnp.ndarray,
            # Models
            actor_def: nn.Module,
            critic_def: nn.Module,
            temperature_def: nn.Module,
            # Optimizer
            actor_optimizer_kwargs={
                "learning_rate": 3e-4,  # 3e-4
            },
            critic_optimizer_kwargs={
                "learning_rate": 3e-4,  # 3e-4
            },
            temperature_optimizer_kwargs={
                "learning_rate": 3e-4,
            },
            # Algorithm config
            discount: float = 0.95,
            soft_target_update_rate: float = 0.005,
            target_entropy: Optional[float] = None,
            entropy_per_dim: bool = False,
            backup_entropy: bool = False,
            critic_ensemble_size: int = 2,
            critic_subsample_size: Optional[int] = None,
            image_keys: Iterable[str] = ("image",),
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            # target_entropy = -actions.shape[-1] / 2
            from numpy import prod
            target_entropy = -prod(actions.shape)

        print(f"config: discount: {discount}, target_entropy: {target_entropy}")

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                # augmentation config
                state_noise_apply_prob=1.0,
                state_noise_std_assignments=None,   # dict for build_std_vec_from_slices
                state_noise_std_vec=None,           # precomputed std vec
            ),
        )

    @classmethod
    def create_drq(
            cls,
            rng: PRNGKey,
            observations: Data,
            actions: jnp.ndarray,
            # Model architecture
            encoder_type: str = "small",
            use_proprio: bool = False,
            state_mask: str = "all",
            proprio_latent_dim: int = 64,
            critic_network_kwargs: dict = {
                "hidden_dims": [256, 256],
            },
            policy_network_kwargs: dict = {
                "hidden_dims": [256, 256],
            },
            policy_kwargs: dict = {
                "tanh_squash_distribution": True,
                "std_parameterization": "uniform",
            },
            encoder_kwargs: dict = {
                "pooling_method": "spatial_learned_embeddings",
                "num_spatial_blocks": 8,
                "bottleneck_dim": 256,
            },
            critic_ensemble_size: int = 2,
            critic_subsample_size: Optional[int] = None,
            temperature_init: float = 1.0,
            image_keys: Iterable[str] = ("image",),
            **kwargs,
    ):
        """
        Create a new pixel-based agent.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder
            small_encoder = SmallEncoder(
                    features=(64, 64, 32, 32),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 1, 1),
                    padding="VALID",
                    pool_method="spatial_learned_embeddings",
                    bottleneck_dim=128,
                    spatial_block_size=8,
                    name=f"small_encoder",
                )
            # use the same encoder
            encoders = {
                image_key: small_encoder
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    name=f"encoder_{image_key}",
                    **encoder_kwargs
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )

            use_single_channel = [value for key, value in observations.items() if key != "state"][0].shape[-3:] == (
                128, 128, 1)
            print(f"use single channel only: {use_single_channel}")

            encoders = {
                image_key: PreTrainedResNetEncoder(
                    rng=rng,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                    use_single_channel=use_single_channel,
                    **encoder_kwargs
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained-18":
            # pretrained ResNet18 from pytorch
            from serl_launcher.vision.resnet_v1_18 import resnetv1_18_configs
            from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder

            pretrained_encoder = resnetv1_18_configs["resnetv1-18-frozen"](
                name="pretrained_encoder",
            )

            use_single_channel = [value for key, value in observations.items() if key != "state"][0].shape[-3:] == (
                128, 128, 1)
            print(f"use single channel only: {use_single_channel}")

            encoders = {
                image_key: PreTrainedResNetEncoder(
                    rng=rng,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                    use_single_channel=use_single_channel,
                    **encoder_kwargs
                )
                for image_key in image_keys
            }
        elif encoder_type == "distance-sensor":
            from serl_launcher.vision.range_sensor import RangeSensorEncoder
            # use depth image as range-like sensor
            assert [value for key, value in observations.items() if key != "state"][0].shape[-3:] == (128, 128, 1)
            import numpy as np

            # 3x3 points centered in the middle
            keypoints = [tuple(k) for k in np.stack(np.meshgrid([32, 64, 96], [32, 64, 96])).reshape((-1, 2))]
            keypoint_size = (5, 5)

            encoders = {
                image_key: RangeSensorEncoder(
                    name=f"encoder_{image_key}",
                    keypoints=keypoints,
                    keypoint_size=keypoint_size,
                )
                for image_key in image_keys
            }
        elif encoder_type == "voxel-mlp":  # not used, too many weights...
            encoders = {
                image_key: MLPEncoder(
                    mlp=MLP(
                        hidden_dims=[64],
                        activations=nn.relu,
                        activate_final=True,
                        use_layer_norm=True,
                    ),
                    bottleneck_dim=encoder_kwargs["bottleneck_dim"],
                )
                for image_key in image_keys
            }
        elif encoder_type in ["voxnet", "voxnet-pretrained", "voxnet-color", "voxnet-pretrained-color"]:
            voxnet = VoxNet(
                    bottleneck_dim=encoder_kwargs["bottleneck_dim"],
                    use_conv_bias=True,
                    final_activation=lambda x: nn.leaky_relu(x, negative_slope=0.1),
                    pretrained="pretrained" in encoder_type,
                    use_color="color" in encoder_type,
                    fix_pretrained_gradient=encoder_kwargs.get("fix_pretrained_gradient", "pretrained" in encoder_type),
                )
            encoders = {
                image_key: voxnet       # use the same one
                for image_key in image_keys
            }
        elif encoder_type.lower() == "none":
            encoders = None
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        if actions.shape[-1] == 7:
            state_mask_arr = create_state_mask(state_mask)
            print(f"state_mask: {state_mask}  {state_mask_arr.astype(jnp.int32)}")
        else:
            state_mask_arr = jnp.ones((observations["state"].shape[-1],), dtype=jnp.bool)

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
            proprio_latent_dim=proprio_latent_dim,  # ignored
            state_mask=state_mask_arr
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Define networks
        from serl_launcher.networks.actor_critic_nets import SharedEncoderCriticEnsemble
        critic_def = SharedEncoderCriticEnsemble(
            encoder=encoders["critic"],
            network=MLP(**critic_network_kwargs),
            ensemble_size=critic_ensemble_size,
            name="critic",
        )

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        if encoder_type == "voxnet-pretrained":
            from serl_launcher.utils.train_utils import load_pretrained_VoxNet_params

            freeze_weights = encoder_kwargs.get("fix_pretrained_gradient", True)
            agent = load_pretrained_VoxNet_params(agent, freeze_weights, image_keys)

        if encoder_type == "voxnet-pretrained-color":
            from serl_launcher.utils.train_utils import load_pretrained_VoxNet_params

            freeze_weights = encoder_kwargs.get("fix_pretrained_gradient", True)
            agent = load_pretrained_VoxNet_params(agent, freeze_weights, image_keys, color=True)

        return agent

    def rotation_augmentation_fn(self, observations, next_observations, actions, rng, activated=False):
        if not activated:
            return observations, next_observations, actions
        for pixel_key in self.config["image_keys"]:
            if not "pointcloud" in pixel_key:
                continue  # skip if not pointcloud
            only_180 = self.config["activate_batch_rotation"] == 180

            # rotation of state, action and voxel grid (use the same rng for all of them, so same rotation)
            # jax.debug.print("before {}  {}  {}", observations["state"][0, 0, :], next_observations["state"][0, 0, :], actions[0, :])
            # jax.debug.print("voxel: \n{}", jnp.mean(observations[pixel_key][0, 0, ...].reshape((5, 10, 5, 10, 40)), axis=(1, 3, 4)))
            # jax.debug.print("action: {}", actions[0, :])
            observations = observations.copy(
                add_or_replace={
                    "state": batched_random_rot90_state(
                        observations["state"], rng, num_batch_dims=2, only_180=only_180
                    ),
                    pixel_key: batched_random_rot90_voxel(
                        observations[pixel_key], rng, num_batch_dims=2, only_180=only_180
                    ),
                }
            )
            next_observations = next_observations.copy(
                add_or_replace={
                    "state": batched_random_rot90_state(
                        next_observations["state"], rng, num_batch_dims=2, only_180=only_180
                    ),
                    pixel_key: batched_random_rot90_voxel(
                        next_observations[pixel_key], rng, num_batch_dims=2, only_180=only_180
                    )
                }
            )
            actions = batched_random_rot90_action(actions, rng, only_180=only_180)
            # jax.debug.print("after {}  {}  {}\n", observations["state"][0, 0, :], next_observations["state"][0, 0, :], actions[0, :])
            # jax.debug.print("voxel after: \n{}", jnp.mean(observations[pixel_key][0, 0, ...].reshape((5, 10, 5, 10, 40)), axis=(1, 3, 4)))
            # jax.debug.print("action after: {}", actions[0, :])

        return observations, next_observations, actions

    def image_augmentation_fn(self, obs_rng, observations, next_obs_rng, next_observations):
        for pixel_key in self.config["image_keys"]:
            # pointcloud augmentation
            if "pointcloud" in pixel_key:
                observations = observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_shift_voxel(
                            observations[pixel_key], obs_rng, padding=3, num_batch_dims=2
                        )
                    }
                )
                next_observations = next_observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_shift_voxel(
                            next_observations[pixel_key], next_obs_rng, padding=3, num_batch_dims=2
                        )
                    }
                )

            # image augmentation
            else:
                observations = observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_crop(
                            observations[pixel_key], obs_rng, padding=4, num_batch_dims=2
                        )
                    }
                )
                next_observations = next_observations.copy(
                    add_or_replace={
                        pixel_key: batched_random_crop(
                            next_observations[pixel_key], next_obs_rng, padding=4, num_batch_dims=2
                        )
                    }
                )
        return observations, next_observations

    def augmentation_fn(self, rng, observations, next_observations, actions):
        """
        Apply image, rotation (disabled), and state Gaussian noise augmentations.
        Expects observations/next_observations with key "state" and image keys.
        """
        # Image augmentation
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs, next_obs = self.image_augmentation_fn(
            obs_rng=obs_rng,
            observations=observations,
            next_obs_rng=next_obs_rng,
            next_observations=next_observations,
        )

        # Rotation augmentation (disabled for now)
        # rng, rot90_rng = jax.random.split(rng)
        # obs, next_obs, actions = self.rotation_augmentation_fn(
        #     observations=obs,
        #     next_observations=next_obs,
        #     actions=actions,
        #     rng=rot90_rng,
        #     activated=False,
        # )

        # State Gaussian noise (on normalized state)
        # Build std_vec once and cache in config if not present. If no assignments, skip.
        std_vec = self.config.get("state_noise_std_vec")
        if std_vec is None:
            total_dim = observations["state"].shape[-1]
            assignments = self.config.get("state_noise_std_assignments")
            if assignments is None:
                assignments = get_box_handover_state_noise_assignments(total_dim)
            if assignments is not None:
                std_vec = build_std_vec_from_slices(total_dim, assignments, default_std=0.0)
                self = self.replace(config={**self.config, "state_noise_std_vec": std_vec})  # cache

        apply_prob = float(self.config.get("state_noise_apply_prob", 1.0))
        if std_vec is not None and apply_prob > 0.0:
            rng, n1, n2 = jax.random.split(rng, 3)
            obs = obs.copy(
                add_or_replace={
                    "state": add_gaussian_noise_state(
                        obs["state"], n1, std_vec, apply_prob=apply_prob, num_batch_dims=2
                    )
                }
            )
            next_obs = next_obs.copy(
                add_or_replace={
                    "state": add_gaussian_noise_state(
                        next_obs["state"], n2, std_vec, apply_prob=apply_prob, num_batch_dims=2
                    )
                }
            )

        return obs, next_obs, actions

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
            self,
            batch: Batch,
            *,
            utd_ratio: int,
            pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.

        It also performs data augmentation on the observations and next_observations
        before updating the network.
        """
        new_agent = self
        if len(self.config["image_keys"]) and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, aug_rng = jax.random.split(rng)
        obs, next_obs, actions = self.augmentation_fn(
            aug_rng, batch["observations"], batch["next_observations"], batch["actions"]
        )
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
                "actions": actions,
            }
        )

        # TODO implement K=2 and M=2

        new_state = self.state.replace(rng=rng)

        new_agent = self.replace(state=new_state)
        return SACAgent.update_high_utd(
            new_agent, batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_critics(
            self,
            batch: Batch,
            *,
            pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        new_agent = self
        if len(self.config["image_keys"]) and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        # TODO implement K=2 and M=2

        rng = new_agent.state.rng
        rng, aug_rng = jax.random.split(rng)
        obs, next_obs, actions = self.augmentation_fn(
            aug_rng, batch["observations"], batch["next_observations"], batch["actions"]
        )

        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
                "actions": actions,
            }
        )

        new_state = self.state.replace(rng=rng)
        new_agent = self.replace(state=new_state)
        new_agent, critic_infos = new_agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"critic"}),
        )
        del critic_infos["actor"]
        del critic_infos["temperature"]

        return new_agent, critic_infos

    def test_augmentation(self, batch: Batch):
        if len(self.config["image_keys"]) and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        obs_cp, next_obs_cp, actions_cp = batch["observations"].copy(), batch["next_observations"].copy(), batch[
            "actions"].copy()

        rng, rot90_rng = jax.random.split(self.state.rng, 2)
        obs, next_obs, actions = self.rotation_augmentation_fn(
            observations=batch["observations"],
            next_observations=batch["next_observations"],
            actions=batch["actions"],
            rng=rot90_rng,
            activated=True,
        )

        for _ in range(3):
            obs, next_obs, actions = self.rotation_augmentation_fn(
                observations=obs,
                next_observations=next_obs,
                actions=actions,
                rng=rot90_rng,
                activated=True,
            )

        # each instance of the batch is rotated by 0, 90, 180 or 270 degrees,
        # 4 times these should all return to the original one
        print("-" * 35, "\nAsugmentation check here!\n", "-" * 35)
        jax.debug.print("obs {}", jnp.sum(jnp.abs(obs["state"] - obs_cp["state"])))
        jax.debug.print("next_obs {}", jnp.sum(jnp.abs(next_obs["state"] - next_obs_cp["state"])))
        jax.debug.print("actions {}", jnp.sum(jnp.abs(actions - actions_cp)))
        # they are all in range 1e-6 to 1e-7, check done
