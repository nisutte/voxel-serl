import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


"""
sparse convolution implementation from deepseek
TODO test, has never been tested (could yield performance increases)
"""


"""                             Usage:
# Example sparse data
N = 1000  # Number of non-empty voxels
indices = np.random.randint(0, 32, (N, 3))  # Random coordinates in 32x32x32 grid
features = np.random.randn(N, 1)            # 1 input channel

# Initialize module
model = SparseConv3D(features=64, kernel_size=3, spatial_size=(32, 32, 32))
params = model.init(random.PRNGKey(0), indices, features)

# Apply convolution
new_indices, new_features = model.apply(params, indices, features)
"""


class SparseConv3D(nn.Module):
    features: int
    kernel_size: int = 3
    spatial_size: Tuple[int, int, int] = (32, 32, 32)
    padding: str = 'same'  # or 'valid'

    def setup(self):
        in_channels = 1  # Adjust based on your input channels
        self.kernel = self.param(
            'kernel',
            nn.initializers.he_normal(),
            (self.kernel_size, self.kernel_size, self.kernel_size, in_channels, self.features)
        )
        self.pad = self.kernel_size // 2 if self.padding == 'same' else 0

    def __call__(self, indices, features):
        """
        Args:
            indices: [N, 3] int32 array of sparse voxel coordinates
            features: [N, in_channels] float32 array of features

        Returns:
            new_indices: [M, 3] output coordinates
            new_features: [M, out_channels] output features
        """
        return sparse_conv3d(indices, features, self.kernel, self.spatial_size, self.pad)


def sparse_conv3d(indices, features, kernel, spatial_size, pad):
    K = kernel.shape[0]
    spatial_size = jnp.array(spatial_size)

    # Generate kernel offsets
    grid = jnp.arange(K) - pad
    offsets = jnp.stack(jnp.meshgrid(grid, grid, grid, indexing='ij'), axis=-1).reshape(-1, 3)

    # Calculate output indices [N*K^3, 3]
    expanded_indices = indices[:, None, :] - offsets[None, :, :]
    expanded_indices = expanded_indices.reshape(-1, 3)

    # Validity check
    valid = jnp.all((expanded_indices >= 0) & (expanded_indices < spatial_size), axis=1)
    output_indices = expanded_indices[valid]

    # Gather corresponding features and kernel weights
    input_idx = jnp.repeat(jnp.arange(indices.shape[0]), K ** 3)[valid]
    kernel_idx = jnp.tile(jnp.arange(K ** 3), indices.shape[0])[valid]

    gathered_feat = features[input_idx]
    gathered_kernel = kernel.reshape(-1, *kernel.shape[-2:])[kernel_idx]

    # Compute contributions [M, out_channels]
    contributions = jnp.einsum('...i,...ij->...j', gathered_feat, gathered_kernel)

    # Aggregate contributions using unique indices
    unique_indices, inverse = jnp.unique(output_indices, axis=0, return_inverse=True, size=output_indices.shape[0])
    output_features = jnp.zeros((unique_indices.shape[0], kernel.shape[-1]))
    output_features = output_features.at[inverse].add(contributions, indices_are_sorted=True)

    return unique_indices, output_features