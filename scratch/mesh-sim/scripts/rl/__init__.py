from gymnasium.envs.registration import register

register(
    id="mesh_sim/MeshEnv-v0",
    entry_point="mesh_sim.scripts.rl.env.mesh_env:MeshEnv",
)
