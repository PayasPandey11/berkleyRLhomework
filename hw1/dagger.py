import os
import argparse
import numpy as np
from keras.models import load_model
import gym
import pickle
import load_policy

import tf_util
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("env_name", type=str)

args = parser.parse_args()
env_name = args.env_name


agent_name = env_name

model = load_model(f"./models/hw1/{agent_name}.h5")


file = open(f"./expert_data/{agent_name}.pkl", "rb")
data = pickle.load(file)
exp_observations, exp_actions = data["observations"], data["actions"]
policy_fn = load_policy.load_policy(f"./experts/{agent_name}.pkl")
env = gym.make(agent_name)


max_steps = 500 or env.spec.timestep_limit
returns = []
observations = []
actions = []
render = True

for i in range(10):
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
    model.fit(np.array(exp_observations), np.array(actions), epochs=1)

    returns.append(totalr)


print(f"returns = {returns}")
print(f"mean return = {np.mean(returns)}")
print(f"std of return = {np.std(returns)}")
