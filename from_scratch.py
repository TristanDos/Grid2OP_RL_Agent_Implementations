import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import json

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace, MultiDiscreteActSpace
from lightsim2grid import LightSimBackend

from typing import Dict, Literal, Any
import copy

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
import numpy as np
from tqdm import tqdm

class Gym2OpEnv(gym.Env):
    def __init__(self,
                 env_config: Dict[Literal["obs_attr_to_keep",
                                          "act_type",
                                          "act_attr_to_keep"],
                                  Any]= None):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        if env_config is None:
            env_config = {}

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        # customize observation space
        obs_attr_to_keep = [key for key in self._gym_env.observation_space.keys()]
        obs_check = ['a_or',
                    'active_alert',
                    'actual_dispatch',
                    'alert_duration',
                    'attack_under_alert',
                    'attention_budget',
                    'current_step',
                    'curtailment',
                    'curtailment_limit',
                    'curtailment_limit_effective',
                    'curtailment_limit_mw',
                    'curtailment_mw',
                    'day',
                    'day_of_week',
                    'delta_time',
                    'duration_next_maintenance',
                    'gen_margin_down',
                    'gen_margin_up',
                    'gen_p',
                    'gen_p_before_curtail',
                    'gen_q',
                    'gen_theta',
                    'gen_v',
                    'hour_of_day',
                    'is_alarm_illegal',
                    'last_alarm',
                    'line_status',
                    'load_p',
                    'load_q',
                    'load_theta',
                    'load_v',
                    'max_step',
                    'minute_of_hour',
                    'month',
                    'p_ex',
                    'p_or',
                    'prod_p',
                    'prod_q',
                    'prod_v',
                    'q_ex',
                    'q_or',
                    'rho',
                    'storage_charge',
                    'storage_power',
                    'storage_power_target',
                    'storage_theta',
                    'target_dispatch',
                    'thermal_limit',
                    'theta_ex',
                    'theta_or',
                    'time_before_cooldown_line',
                    'time_before_cooldown_sub',
                    'time_next_maintenance',
                    'time_since_last_alarm',
                    'time_since_last_alert',
                    'time_since_last_attack',
                    'timestep_overflow',
                    'topo_vect',
                    'total_number_of_alert',
                    'v_ex',
                    'v_or',
                    'was_alarm_used_after_game_over',
                    'was_alert_used_after_attack',
                    'year']

        if "obs_attr_to_keep" in env_config:
            obs_attr_to_keep = copy.deepcopy(env_config["obs_attr_to_keep"])
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space,
                                                         attr_to_keep=obs_attr_to_keep
                                                         )
        # export observation space for the Grid2opEnv
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)

        # customize the action space
        act_type = "discrete"
        if "act_type" in env_config:
            act_type = env_config["act_type"]

        self._gym_env.action_space.close()

        act_attr_to_keep = [key for key in self._gym_env.action_space.keys()]
        
        if act_type == "discrete":
            # user wants a discrete action space
            if "act_attr_to_keep" in env_config:
                act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space,
                                                          attr_to_keep=act_attr_to_keep)
            self.action_space = Discrete(self._gym_env.action_space.n)
        elif act_type == "box":
            # user wants continuous action space
            if "act_attr_to_keep" in env_config:
                act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = BoxGymActSpace(self._g2op_env.action_space,
                                                        attr_to_keep=act_attr_to_keep)
            self.action_space = Box(shape=self._gym_env.action_space.shape,
                                    low=self._gym_env.action_space.low,
                                    high=self._gym_env.action_space.high)
        elif act_type == "multi_discrete":
            # user wants a multi-discrete action space
            if "act_attr_to_keep" in env_config:
                act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space,
                                                               attr_to_keep=act_attr_to_keep)
            self.action_space = MultiDiscrete(self._gym_env.action_space.nvec)
        else:
            raise NotImplementedError(f"action type '{act_type}' is not currently supported.")
        
        # self.setup_observations()
        # self.setup_actions()

    def flatten_action_space(self):
        """
        Flatten the dictionary of actions into a MultiBinary action space.
        This will map each binary decision into a separate element in the MultiBinary space.
        """
        action_space_dict = self._gym_env.action_space.spaces
        binary_lengths = []

        # Flatten all MultiBinary and Discrete actions into a single binary vector
        for key, space in action_space_dict.items():
            if isinstance(space, gym.spaces.MultiBinary):
                binary_lengths.append(space.n)
            elif isinstance(space, gym.spaces.Box):
                # Flatten Box space into binary format
                flattened_length = np.prod(space.shape)
                binary_lengths.append(int(flattened_length))

        # Return a MultiBinary space representing the flattened action space
        return gym.spaces.MultiBinary(sum(binary_lengths))

    def unflatten_action(self, action):
        """
        Convert the flattened action space back into the dictionary format
        that the environment expects.
        """
        action_space_dict = self._gym_env.action_space.spaces
        start_idx = 0
        action_dict = {}

        for key, space in action_space_dict.items():
            if isinstance(space, gym.spaces.MultiBinary):
                end_idx = start_idx + space.n
                action_dict[key] = action[start_idx:end_idx]
                start_idx = end_idx
            elif isinstance(space, gym.spaces.Box):
                flattened_length = int(np.prod(space.shape))
                end_idx = start_idx + flattened_length
                action_dict[key] = np.reshape(action[start_idx:end_idx], space.shape)
                start_idx = end_idx

        return action_dict

    def setup_observations(self):
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        self.action_space = self.flatten_action_space()

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        # Map the flattened action space back to the original action space dictionary
        action_dict = action

        return self._gym_env.step(action_dict)

    def render(self):
        return self._gym_env.render()

# Training PPO or A2C on the Grid2Op Environment
def train_agent(agent_type="PPO"):
    env = Gym2OpEnv()
    
    # Check if the environment is compatible
    check_env(env)

    # Train either PPO or A2C
    if agent_type == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=1)
    elif agent_type == "A2C":
        model = A2C("MultiInputPolicy", env, verbose=1)
    else:
        raise ValueError("Unsupported agent type. Choose either 'PPO' or 'A2C'.")

    # Train the model
    model.learn(total_timesteps=10000, progress_bar=True)

    # Save the model
    model.save(f"{agent_type}_grid2op")

    return model

def evaluate_agent(model, env, num_episodes=10, random_agent=False):
    total_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes), desc=f"Evaluating over {num_episodes} episodes"):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if random_agent:
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # Compute average reward and episode length
    avg_reward = sum(total_rewards) / num_episodes
    avg_length = sum(episode_lengths) / num_episodes

    return avg_reward, avg_length

if __name__ == "__main__":
    agent_type = "PPO"  # Change this to "A2C" if you want to train A2C
    trained_model = train_agent(agent_type)
    
    # Evaluate the trained agent
    env = Gym2OpEnv()
    avg_reward, avg_length = evaluate_agent(trained_model, env)
    print(f"Average Reward: {avg_reward}, Average Episode Length: {avg_length}")

    agent_type = "Random"

    # Evaluate the trained agent
    env = Gym2OpEnv()
    avg_reward, avg_length = evaluate_agent(model=None, env=env, random_agent=True)
    print(f"Average Reward: {avg_reward}, Average Episode Length: {avg_length}")
