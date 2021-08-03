from domus_gym.envs.domus_env import DomusEnv


def test_domus_env():
    env = DomusEnv()

    env.step(action=1)
