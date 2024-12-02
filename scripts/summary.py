from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import domus_gym
from domus_mlsim import load_scenarios

N_UCS = 28


def main(envname: str, alg: str, log_path: str, run_number: str) -> np.ndarray:
    log_path = Path(log_path)
    assert log_path.exists()

    ppo_path = log_path / alg.lower() / (envname + "_" + run_number)
    assert ppo_path.exists()
    model_file = ppo_path / (envname + ".zip")
    vecnormalize = ppo_path / envname / "vecnormalize.pkl"

    sc = load_scenarios()
    averages = np.zeros((N_UCS, 4))
    env = DummyVecEnv(
        [
            lambda: gym.make(
                envname, use_scenario=1, fixed_episode_length=sc.time[1] * 60
            )
        ]
    )
    env = VecNormalize.load(vecnormalize, env)
    env.norm_reward = False
    model = PPO.load(model_file, env=env)
    for i in range(1, N_UCS + 1):
        env.set_attr("use_scenario", i)
        env.set_attr("fixed_episode_length", sc.time[i] * 60)

        obs = env.reset()

        done = False
        count = 0
        totals = np.zeros((4))

        while not done:
            action, state = model.predict(obs)
            obs, reward, done, infos = env.step(action)
            reward = reward[0]
            done = done[0]
            infos = infos[0]
            count += 1
            totals += np.array(
                [reward, infos["comfort"], infos["energy"], infos["safety"]]
            )

        averages[i - 1] = totals / count

    return averages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="summarise performance using assessment framework"
    )
    parser.add_argument("-a", "--algo", help="Algorithm (e.g., ppo)", type=str)
    parser.add_argument("-e", "--env", help="Environment", type=str)
    parser.add_argument(
        "-r",
        "--run-number",
        help="Suffix on environment for run",
        type=str,
        nargs="+",
    )
    parser.add_argument("-f", "--exp-folder", help="Folder", type=str)
    parser.add_argument(
        "outputfile",
        help="csv to write results to",
        type=argparse.FileType("w"),
    )

    args = parser.parse_args()

    args.algo = args.algo.upper()

    all_runs = []
    for run in args.run_number:
        averages = main(
            envname=args.env,
            alg=args.algo,
            log_path=args.exp_folder,
            run_number=run,
        )

        rces = ["reward", "comfort", "energy", "safety"]
        avgdf = pd.DataFrame(averages, index=range(1, N_UCS + 1), columns=rces)
        avgdf = avgdf.assign(run=run)
        all_runs.append(avgdf)

    alldf = pd.concat(all_runs)
    alldf.to_csv(args.outputfile)
    print(alldf.groupby("run").aggregate(["mean", "std"]))
