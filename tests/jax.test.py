#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python3 python3Packages.jax python3Packages.jaxlibWithCuda

# See
# * https://github.com/NixOS/nixpkgs/blob/462770166e93a78b1586e8cfe481425b9db91525/pkgs/development/python-modules/jaxlib/bin.nix#L5-L14
# * https://github.com/NixOS/nixpkgs/pull/164176

from jax import random
from jax.lib import xla_bridge

# Check that jaxlib can find the GPU
assert xla_bridge.get_backend().platform == "gpu"

# Check that we can generate PRNGKey's. This has been broken before. TODO: source?
rng = random.PRNGKey(0)

# Check that we can multiply matrices via cuDNN.
x = random.normal(rng, (100, 100))
x @ x
