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

from stable_baselines3 import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from tqdm import tqdm
import os

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
        if "act_type" in env_config:
            act_type = env_config["act_type"]
        else:
            act_type = "discrete"

        self._gym_env.action_space.close()
        
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
            print(act_type)
            raise NotImplementedError(f"action type '{act_type}' is not currently supported.")
        
        # self.setup_observations()
        # self.setup_actions()

    def setup_observations(self):
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        self.action_space = self.flatten_action_space()

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()

class RandomAgent():
    def __init__(self, env: Gym2OpEnv):
        self.env = env

    def predict(self, obs, deterministic: bool):
        return self.env.action_space.sample(), None
    
    def learn(self, total_timesteps: int, progress_bar: bool):
        pass

    def save(self, path):
        pass

def evaluate_agent(model, env, num_episodes=10):
    total_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes), desc=f"Evaluating over {num_episodes} episodes"):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
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

def main():

    act_attr_to_keep = ['change_bus', 'change_line_status', 'curtail', 'curtail_mw', 'redispatch', 'set_bus', 'set_line_status', 'set_storage']
    obs_attr_to_keep = ['a_or',
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

    env_configs = {
        'DQN': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"discrete", "act_attr_to_keep": act_attr_to_keep},
        'PPO': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"discrete", "act_attr_to_keep": act_attr_to_keep},
        'A2C': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"discrete", "act_attr_to_keep": act_attr_to_keep},
        'SAC': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"box", "act_attr_to_keep": act_attr_to_keep},
        'HER': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"discrete", "act_attr_to_keep": act_attr_to_keep},
        'DDPG':{"obs_attr_to_keep": obs_attr_to_keep,'act_type': "box", "act_attr_to_keep": act_attr_to_keep},
        'TD3': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"box", "act_attr_to_keep": act_attr_to_keep},
        'Random': {"obs_attr_to_keep": obs_attr_to_keep,'act_type':"discrete", "act_attr_to_keep": act_attr_to_keep},
    }

    models = {
        'Random': RandomAgent(Gym2OpEnv(env_configs['Random'])),
        'DQN': DQN("MlpPolicy", Gym2OpEnv(env_configs['DQN']), verbose=0),
        'PPO': PPO("MlpPolicy", Gym2OpEnv(env_configs['PPO']), verbose=0),
        'A2C': A2C("MlpPolicy", Gym2OpEnv(env_configs['A2C']), verbose=0),
        'SAC': SAC("MlpPolicy", Gym2OpEnv(env_configs['SAC']), verbose=0),
        # 'HER': HerReplayBuffer("MlpPolicy", Gym2OpEnv(env_configs['HER']), verbose=1),
        'DDPG': DDPG("MlpPolicy", Gym2OpEnv(env_configs['DDPG']), verbose=0),
        'TD3': TD3("MlpPolicy", Gym2OpEnv(env_configs['TD3']), verbose=0),
    }
    
    for model_name, model in models.items():
        if not os.path.exists(f'models/{model_name}.zip'):
            print("Training: ", model_name)
            model.learn(total_timesteps=1024, progress_bar=True)
            model.save(f"models/{model_name}")
        else:
            model.load(f'models/{model_name}.zip')
    
    final_out = ""
    for model_name, model in models.items():
        env_config = {"obs_attr_to_keep": env_configs[model_name]['obs_attr_to_keep'], "act_type": env_configs[model_name]['act_type'], "act_attr_to_keep": env_configs[model_name]['act_attr_to_keep']}
        gym_env = Gym2OpEnv(env_config)
        
        out = f"Evaluating for {model_name}:\n"
        print(out)
        avg_reward, avg_length = evaluate_agent(model, gym_env, 50)
        out += f"Average reward: {avg_reward}\nAverage length: {avg_length}\n"
        print(f"Average reward: {avg_reward}\nAverage length: {avg_length}\n")

        final_out += out + "\n"

    results = open("results.txt", "w")
    results.write(final_out)
    results.close()

if __name__ == "__main__":
    main()