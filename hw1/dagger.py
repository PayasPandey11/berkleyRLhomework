"""
Trains Dagger on behavioral cloning model with expert data by berkely cs234 
Example usage:
python dagger.py Humanoid-v2
"""
import os
import argparse
import numpy as np
from keras.models import load_model
import gym
import pickle
import load_policy

import tf_util
import tensorflow as tf
from gym import wrappers


def dagger(
    exp_observations,
    exp_actions,
    model,
    max_steps=1000,
    roll_outs=30,
    model_epochs=1,
    render=False,
):
    returns = []
    observations = []
    actions = []

    for episode in range(roll_outs):
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        dagger_obs = []
        dagger_actions = []
        while not done:
            action = model.predict(np.reshape(np.array(obs), (1, len(obs))))
            corrected_action = policy_fn(obs[None, :])
            obs, r, done, _ = env.step(action)

            totalr += r
            steps += 1
            if render:
                env.render()
            if steps >= max_steps:
                break

            dagger_obs.append(obs)
            dagger_actions.append(corrected_action)

        exp_observations = np.concatenate((exp_observations, dagger_obs), axis=0)
        exp_actions = np.concatenate((exp_actions, dagger_actions), axis=0)
        actions = [act.flatten() for act in exp_actions]
        model.fit(np.array(exp_observations), np.array(actions), epochs=model_epochs)

        returns.append(totalr)
    return returns


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)

    args = parser.parse_args()
    env_name = args.env_name

    env = gym.make(env_name)
    env = wrappers.Monitor(
        env,
        f"Saved_Videos/hw1/dagger/{env_name}/",
        resume=True,
        force=True,
        video_callable=lambda episode: episode % 10 == 0,
    )

    model = load_model(f"./models/hw1/{env_name}.h5")

    file = open(f"./expert_data/{env_name}.pkl", "rb")
    data = pickle.load(file)
    exp_observations, exp_actions = data["observations"], data["actions"]
    policy_fn = load_policy.load_policy(f"./experts/{env_name}.pkl")

    returns = dagger(exp_observations, exp_actions, model, max_steps=1000)

    print(f"returns = {returns}")
    print(f"mean return = {np.mean(returns)}")
    print(f"std of return = {np.std(returns)}")
