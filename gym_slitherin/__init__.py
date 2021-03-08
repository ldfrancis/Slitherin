from gym.envs.registration import register
from gym_slitherin.envs.slitherin_env import SlitherinEnv

register(id="Slitherin2-v0",
    entry_point="gym_slitherin.envs:SlitherinEnv",
    kwargs={"n_agents":2

    },
)

register(id="Slitherin4-v0",
    entry_point="gym_slitherin.envs:SlitherinEnv",
    kwargs={"n_agents":4

    },
)

register(id="Slitherin1-v0",
    entry_point="gym_slitherin.envs:SlitherinEnv",
    kwargs={"n_agents":1

    },
)
