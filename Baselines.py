import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from collections import OrderedDict

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        # Setup reward
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        # Define the observations we want to keep
        self.obs_to_keep = [
            'rho', 'line_status', 'topo_vect', 'actual_dispatch',
            'target_dispatch', 'gen_p', 'load_p'
        ]
        
        # Calculate the flattened observation space size
        self.flat_obs_size = sum(
            np.prod(self._gym_env.observation_space[k].shape)
            for k in self.obs_to_keep
        )
        
        # Define our simplified observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.flat_obs_size,)
        )

    def setup_actions(self):
        # Simplify the action space to focus on 'change_bus' and 'redispatch'
        low = []
        high = []
        for key, space in self._gym_env.action_space.spaces.items():
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

    def _flatten_observation(self, obs):
        return np.concatenate([
            obs[k].flatten() for k in self.obs_to_keep
        ])
    
    def unflatten_action(self, action):
        original_action = {}
        idx = 0
        for key, space in self._gym_env.action_space.spaces.items():
            if isinstance(space, gym.spaces.MultiBinary):
                size = space.n
                original_action[key] = action[idx:idx + size]
                idx += size
            elif isinstance(space, gym.spaces.Box):
                size = space.shape[0]
                original_action[key] = action[idx:idx + size]
                idx += size
        return original_action

    def reset(self, seed=None):
        self.curr_step = 0 
        return self._gym_env.reset(seed=seed)

    def step(self, action):
        # Convert the action from our simplified space to the full Grid2Op action
        full_action = self._gym_env.action_space({
            'change_bus': action['change_bus'],
            'redispatch': action['redispatch']
        })
        obs, reward, terminated, truncated, info = self._gym_env.step(full_action)
        return self._flatten_observation(obs), reward, terminated, truncated, info

    def render(self):
        return self._gym_env.render()

# A3C Neural Network
class A3CNetwork(nn.Module):
    def __init__(self, input_size, n_actions):
        super(A3CNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

# A3C Agent
class A3CAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.gamma = gamma

        input_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.network = A3CNetwork(input_size, n_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        logits, _ = self.network(state)
        probs = F.softmax(logits, dim=-1)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        logits, values = self.network(states)
        _, next_values = self.network(next_states)

        probs = F.softmax(logits, dim=-1)
        action_dist = Categorical(probs)
        log_probs = action_dist.log_prob(actions)

        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        entropy = -(probs * probs.log()).sum(1).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(agent, n_episodes=1000, max_steps=100):
    for episode in range(n_episodes):
        state = agent.env.reset()
        total_reward = 0
        done = False
        step = 0

        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not done and step < max_steps:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward
            step += 1

            if len(states) == 32 or done:  # Update every 32 steps or at episode end
                agent.update(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {step}")

def main():
    env = Gym2OpEnv()
    agent = A3CAgent(env)
    train(agent)

if __name__ == "__main__":
    main()