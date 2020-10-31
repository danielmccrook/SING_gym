from gym.envs.registration import register

# Meta
# ----------------------------------------
register(
    id='SINGRL-v0',
    entry_point='singRL.envs:SINGEnv',
    max_episode_steps=200,
    reward_threshold=90.0,
)
