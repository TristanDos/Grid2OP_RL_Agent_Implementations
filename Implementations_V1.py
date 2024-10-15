import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt

import os

import pickle

import grid2op
from grid2op import gym_compat
from grid2op.gym_compat import BoxGymObsSpace
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.Parameters import Parameters

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

from tqdm import tqdm

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self, max_steps):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Combines L2RPN and N1 rewards

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep


        # Create Grid2Op environment with specified parameters
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.max_steps = max_steps 
        self.curr_step = 0 

        self.actions_to_keep = ['change_bus', 'change_line_status', 'curtail']
        self.observations_to_keep = ["load_p", "load_q",
                            "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                            "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

        print(self._gym_env.action_space)
        print(self._gym_env.observation_space)

        self.setup_observations()
        self.setup_actions()

        print(self.observation_space)

    def setup_observations(self):
        # Get the original observation space
        original_space = self._gym_env.observation_space

        # Create a new dictionary to hold the filtered spaces
        filtered_spaces = {}

        for key in self.observations_to_keep:
            if key in original_space.spaces:
                filtered_spaces[key] = original_space.spaces[key]

        # Create a new observation space with only the filtered spaces
        self.observation_space = gym.spaces.Dict(filtered_spaces)


    def setup_actions(self):

        low = []
        high = []
        for key, space in self._gym_env.action_space.spaces.items():
            if key not in self.actions_to_keep:
                continue

            if isinstance(space, gym.spaces.MultiBinary):
                low.extend([0] * space.n)
                high.extend([1] * space.n)
            elif isinstance(space, gym.spaces.Box):
                low.extend(space.low.tolist())
                high.extend(space.high.tolist())
            else:
                raise NotImplementedError(f"Unsupported action space type: {type(space)}")

        self.action_space =  gym.spaces.Box(np.array(low), np.array(high), dtype=np.int32)
        # self.action_space =  gym.spaces.flatten_space(self._gym_env.action_space)

    def unflatten_action(self, action):

        original_action = {}
        idx = 0
        for key, space in self._gym_env.action_space.spaces.items():
            if key not in self.actions_to_keep:
                continue

            if isinstance(space, gym.spaces.MultiBinary):
                size = space.n
                original_action[key] = action[idx:idx + size]
                idx += size
            elif isinstance(space, gym.spaces.Box):
                size = space.shape[0]
                original_action[key] = action[idx:idx + size]
                idx += size
        return original_action

    def step(self, action):
        original_action = self.unflatten_action(action)

        self.curr_step += 1 
        obs, reward, terminated, truncated, info = self._gym_env.step(original_action)

        # Filter the observation
        filtered_obs = {key: obs[key] for key in self.observations_to_keep if key in obs}

        if self.curr_step >= self.max_steps:
            terminated = True

        return filtered_obs, reward, terminated, truncated, info


    def reset(self, seed=None):  # Add seed argument here
        self.curr_step = 0 
        obs, info = self._gym_env.reset(seed=seed)

        # Filter the observation
        filtered_obs = {key: obs[key] for key in self.observations_to_keep if key in obs}

        return (filtered_obs, info)

    def render(self, mode="human"):
        return self._gym_env.render(mode=mode)

class RewardLoggerCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        # Get the reward for the current step
        reward = self.locals['rewards'][0]
        self.current_rewards.append(reward)

        # Check if the episode is done, then log the reward
        done = self.locals['dones']
        if done:
            episode_reward = np.sum(self.current_rewards)
            self.episode_rewards.append(episode_reward)
            self.current_rewards = []
            if self.verbose > 0:
                print(f"Episode reward: {episode_reward}")
        return True

    def get_rewards(self):
        return self.episode_rewards
    
class EpisodeLengthLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeLengthLoggerCallback, self).__init__(verbose)
        self.episode_lengths = []
        self.current_length = 0

    def _on_step(self) -> bool:
        # Increment the step count
        self.current_length += 1

        # Check if the episode is done, then log the episode length
        done = self.locals['dones']
        if done:
            self.episode_lengths.append(self.current_length)
            self.current_length = 0  # Reset for the next episode
            if self.verbose > 0:
                print(f"Episode length: {self.episode_lengths[-1]}")
        return True

    def get_lengths(self):
        return self.episode_lengths

def create_env(max_steps):
    return Monitor(Gym2OpEnv(max_steps))

def train(model_class, model_name, env, total_timesteps=10000):
    print('Training ' + model_name)

    reward_logger = RewardLoggerCallback()
    length_logger = EpisodeLengthLoggerCallback()

    callback_list = CallbackList([reward_logger, length_logger])

    model = model_class("MultiInputPolicy", env, verbose=1, learning_rate=0.1, gamma=0.9)
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    model.save(f"models/{model_name}")

    print('Completed Training ' + model_name)
    
    # Plotting rewards and Episode lengths after training
    rewards = reward_logger.get_rewards()
    episode_lengths = length_logger.get_lengths()

    return model, rewards, episode_lengths

def evaluate(env, model, n_episodes=30, random_agent=False):
    
    print('Evaluating agent')

    rewards = []
    episode_lengths = []

    for _ in tqdm(range(n_episodes), desc=f"Evaluating over {n_episodes} episodes"):
        episode_reward = 0
        steps = 0
        done = False
        obs = env.reset()[0]
        while not done:
            steps += 1
            if (random_agent):
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    mean_r_reward = np.mean(rewards)
    std_r_reward = np.std(rewards)
    mean_l_reward = np.mean(episode_lengths)
    std_l_reward = np.std(episode_lengths)

    print('Completed evaluating agent')

    return mean_r_reward, std_r_reward, mean_l_reward, std_l_reward

def plot_returns(returns, iter="1"):
    ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std, ppo_reward, ppo_length = returns['ppo']
    a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std, a2c_reward, a2c_length = returns['a2c']
    random_r_mean, random_r_std, random_l_mean, random_l_std = returns['random']

    agents = ['Random', 'PPO', 'A2C']
    r_means = [random_r_mean, ppo_r_mean, a2c_r_mean]
    r_stds = [random_r_std, ppo_r_std, a2c_r_std]
    l_means = [random_l_mean, ppo_l_mean, a2c_l_mean]
    l_stds = [random_l_std, ppo_l_std, a2c_l_std]

    plt.figure(figsize=(10, 6))
    plt.bar(agents, r_means, yerr=r_stds, capsize=10)
    plt.title('Final Agent Return Comparison')
    plt.ylabel('Mean Return')
    plt.ylim(bottom=0)
    for i, v in enumerate(r_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'plots/v{iter}/agent_r_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(agents, l_means, yerr=l_stds, capsize=10)
    plt.title('Final Agent Length Comparison')
    plt.ylabel('Mean Length')
    plt.ylim(bottom=0)
    for i, v in enumerate(r_means):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'plots/v{iter}/agent_l_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_reward, label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_reward, label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Comparison of PPO, A2C, and Random Agent')
    plt.legend()
    plt.savefig(f'plots/v{iter}/agent_reward.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_length, label='PPO', marker='o', linestyle='-')
    plt.plot(a2c_length, label='A2C', marker='s', linestyle='-')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.title('Episode Length Over Time for PPO, A2C and Random Agents')
    plt.legend()
    plt.savefig(f'plots/v{iter}/episode_length_over_time.png')
    plt.close()

def main():
    max_steps = 500
    env = create_env(max_steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    ppo_reward = 0
    a2c_reward = 0
    iteration = "1"
    
    # Train PPO
    if not os.path.exists(f'models/ppo_v{iteration}.zip'):
        ppo_model, ppo_reward, ppo_length = train(PPO, f"ppo_v{iteration}", vec_env)

        combo : tuple[list, list] = (ppo_reward, ppo_length)

        # Pickle a simple object
        with open(f'models/ppo_meta_v{iteration}.pickle', 'wb') as f:
            pickle.dump(combo, f)
    else:
        ppo_model = PPO.load(f'models/ppo_v{iteration}.zip', env=env)

        # Unpickle the object
        with open(f'models/ppo_meta_v{iteration}.pickle', 'rb') as f:
            ppo_reward, ppo_length = pickle.load(f)

    # Train A2C
    if not os.path.exists(f'models/a2c_v{iteration}.zip'):
        a2c_model, a2c_reward, a2c_length = train(A2C, f"a2c_v{iteration}", vec_env)
        combo : tuple[list, list] = (a2c_reward, a2c_length)

        # Pickle a simple object
        with open(f'models/a2c_meta_v{iteration}.pickle', 'wb') as f:
            pickle.dump(combo, f)

    else:
        a2c_model = A2C.load(f'models/a2c_v{iteration}.zip', env=env)

        # Unpickle the object
        with open(f'models/a2c_meta_v{iteration}.pickle', 'rb') as f:
            a2c_reward, a2c_length = pickle.load(f)
    
    return_dict = {}

    # Evaluate PPO
    ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std = evaluate(env, ppo_model)

    return_dict['ppo'] = (ppo_r_mean, ppo_r_std, ppo_l_mean, ppo_l_std, ppo_reward, ppo_length)

    # Evaluate A2C
    a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std = evaluate(env, a2c_model)

    return_dict['a2c'] = (a2c_r_mean, a2c_r_std, a2c_l_mean, a2c_l_std, a2c_reward, a2c_length)

    # Evaluate Random
    random_r_mean, random_r_std, random_l_mean, random_l_std = evaluate(env, None, random_agent=True)

    return_dict['random'] = (random_r_mean, random_r_std, random_l_mean, random_l_std)

    # Plot returns
    plot_returns(return_dict, iteration)

if __name__ == "__main__":
    main()
