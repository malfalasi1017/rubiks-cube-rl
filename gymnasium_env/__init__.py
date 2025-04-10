from gymnasium.envs.registration import register

register(
    id="RubiksCube-v0",
    entry_point="gymnasium_env.env.rubiks_cube:RubiksCubeEnv",
)