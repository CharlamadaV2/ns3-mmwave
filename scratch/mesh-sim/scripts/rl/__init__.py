"""Gymnasium registration for the mesh-sim RL environment."""

from gymnasium.envs.registration import register, registry

ENV_ID = "mesh_sim/MeshEnv-v0"

# Importing this package twice (or alongside an explicit registration) must not raise.
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="scripts.rl.env.mesh_env:MeshRlEnv",
    )
