from gymnasium.envs.registration import register

# Register the custom rubiks cube environment
register(
    id="gymnasium_env/RubiksCube-v0",
    entry_point="gymnasium_env.envs:RubiksCubeEnv"
)