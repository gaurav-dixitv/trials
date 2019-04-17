from gym.envs.registration import register

register(
    id='trials-v0',
    entry_point='trials.envs:Trials',
)
