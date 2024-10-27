import os
import pickle
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v2_spaces


def plot_single(metrics_dict: Dict[str, Dict[str, list]], version="v0"):
    """Plots a set of metrics using a metrics dictionary.

    Plots 4 different graphs including: evaluation returns/episode lengths and training rewards/episode lengths.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, list]]
        Maps model names as keys to a second dictionary. The second dict then maps different metric types to floats.
        i.e. PPO -> Training rewards, Training lengths, Eval rewards, Eval lengths -> floats
    version : str, optional
        String for version/iteration number, by default "v0"
    """    
    model_names = metrics_dict.keys()
    eval_rewards_arr = []
    eval_lengths_arr = []
    training_rewards_dict = {}
    training_lengths_dict = {}

    for model_name in model_names:
        metrics = metrics_dict[model_name]

        eval_reward = metrics['eval_reward']
        eval_length = metrics['eval_length']
        training_rewards = metrics['training_rewards']
        training_lengths = metrics['training_episode_lengths']

        eval_rewards_arr.append(eval_reward)
        eval_lengths_arr.append(eval_length)

        if model_name != "Random":
            training_rewards_dict[model_name] = training_rewards
            training_lengths_dict[model_name] = training_lengths

    plt.figure(figsize=(6, 4))
    plt.bar(model_names, eval_rewards_arr, capsize=10)
    plt.title('Average Model Returns During Evaluation')
    plt.ylabel('Average Return')
    plt.ylim(bottom=0)
    for i, v in enumerate(eval_rewards_arr):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'plots/{version}/eval_rewards.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(model_names, eval_lengths_arr, capsize=10)
    plt.title('Average Model Episode Lengths During Evaluation')
    plt.ylabel('Average Episode Length (Number of Steps)')
    plt.ylim(bottom=0)
    for i, v in enumerate(eval_lengths_arr):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'plots/{version}/eval_lengths.png')
    plt.close()

    plt.figure(figsize=(6, 4))

    counter = 0
    for model_name, training_rewards in training_rewards_dict.items():
        plt.plot(training_rewards, label=model_name, linestyle='-')
        
        counter += 1
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Rewards Accrued During Training for {[model_name for model_name in training_rewards_dict.keys()]}')
    plt.legend()
    plt.savefig(f'plots/{version}/training_rewards.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    
    counter = 0
    for model_name, training_lengths in training_lengths_dict.items():
        plt.plot(training_lengths, label=model_name, linestyle='-')
        
        counter += 1

    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')
    plt.title(f'Episode Length During Training for {[model_name for model_name in training_lengths_dict.keys()]}')
    plt.legend()
    plt.savefig(f'plots/{version}/training_lengths.png')
    plt.close()

def plot_comparisons(metrics_dicts, variations, window_size=30, version="v1", rotation=45, new_folder=""):
    """
    Metrics Dict : model names -> Dict2 : TrainingRewards, EvalRewards etc. -> floats

    Function to plot training and evaluation metrics for PPO and A2C models across different
    observation/action space variations with optional smoothing.
    
    Parameters:
        metrics_dicts (list): A list of metrics dictionaries, each corresponding to a variation of the observation/action space.
        variations (list): A list of strings representing the names of each variation.
        window_size (int): The size of the rolling window for smoothing the line plots.
        version (str): The version of the plots for saving them to a unique directory.
    """

    # Ensure the directory for saving plots exists
    save_dir = f'plots/{version}/{new_folder}'
    os.makedirs(save_dir, exist_ok=True)

    # Models to compare (PPO and A2C)
    models = ['PPO', 'A2C']
    
    # Initialize structures to hold training and evaluation data
    training_rewards_ppo = []
    training_rewards_a2c = []
    eval_rewards_ppo = []
    eval_rewards_a2c = []
    eval_lengths_ppo = []
    eval_lengths_a2c = []
    
    # Iterate over the metrics dictionaries for each variation
    for i, metrics_dict in enumerate(metrics_dicts):
        # Extract PPO data
        ppo_rewards_smoothed = pd.Series(metrics_dict['PPO']['training_rewards']).rolling(window=window_size).mean()
        training_rewards_ppo.append(ppo_rewards_smoothed)
        eval_rewards_ppo.append(np.mean(metrics_dict['PPO']['eval_reward']))
        eval_lengths_ppo.append(np.mean(metrics_dict['PPO']['eval_length']))

        # For version 3, the LSTM implementation was done with only PPO since stable baselines does not have a version for A2C
        if (version == "v3" and i == 3):
            # For the LSTM, we only have info for PPO then, so I appended zeroes for A2C
            eval_rewards_a2c.append(0)
            eval_lengths_a2c.append(0)

            continue

        # Extract A2C data
        a2c_rewards_smoothed = pd.Series(metrics_dict['A2C']['training_rewards']).rolling(window=window_size).mean()
        training_rewards_a2c.append(a2c_rewards_smoothed)
        eval_rewards_a2c.append(np.mean(metrics_dict['A2C']['eval_reward']))
        eval_lengths_a2c.append(np.mean(metrics_dict['A2C']['eval_length']))

    # Plot 1: Line plot comparing PPO training rewards across variations with smoothing
    plt.figure(figsize=(6, 4))
    for i, rewards in enumerate(training_rewards_ppo):
        plt.plot(rewards, label=f'{variations[i]}')
    plt.title(f'PPO Training Rewards Over Time (Window Size: {window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_ppo_training.png')
    plt.close()

    # Plot 2: Line plot comparing A2C training rewards across variations with smoothing
    plt.figure(figsize=(6, 4))
    for i, rewards in enumerate(training_rewards_a2c):
        plt.plot(rewards, label=f'{variations[i]}')
    plt.title(f'A2C Training Rewards Over Time (Window Size: {window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_a2c_training.png')
    plt.close()

    # Bar width for grouped bar plots
    bar_width = 0.35
    index = np.arange(len(variations))
    
    ## Plot 3: Grouped bar graph comparing avg eval rewards for PPO and A2C across variations
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(index, eval_rewards_ppo, bar_width, label='PPO')
    bars2 = plt.bar(index + bar_width, eval_rewards_a2c, bar_width, label='A2C')

    # Adding value labels above the bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    plt.title('Average Evaluation Rewards')
    plt.xlabel('Observation/Action Space Variations')
    plt.ylabel('Avg Rewards')
    plt.xticks(index + bar_width / 2, variations, rotation=rotation)
    # plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_eval_rewards.png')
    plt.close()

    # Plot 4: Grouped bar graph comparing avg eval lengths for PPO and A2C across variations
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(index, eval_lengths_ppo, bar_width, label='PPO')
    bars2 = plt.bar(index + bar_width, eval_lengths_a2c, bar_width, label='A2C')

    # Adding value labels above the bars
    autolabel(bars1)
    autolabel(bars2)

    plt.title('Average Evaluation Episode Lengths')
    plt.xlabel('Observation/Action Space Variations')
    plt.ylabel('Avg Episode Lengths')
    plt.xticks(index + bar_width / 2, variations, rotation=rotation)
    # plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_eval_lengths.png')
    plt.close()

def collectV0():
    """Collects metrics to plot iteration 0 or baseline results
    """    

    version = "v0"

    metrics_dict = {}
    with open(f'fromClusterRun4/models/{version}/metricsseed4321.pkl', 'rb') as f:
        metrics_dict = pickle.load(f) # deserialize using load()]
    
    plot_single(metrics_dict, version=version)

def collectV1():
    """Collects metrics to plot iteration 1 results
    """

    version = "v1"

    var_names = ["REMOVE_REDUNDANT U REMOVE_ADVERSARIAL", "REMOVE_ADVERSARIAL",
                  "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL", "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT",
                  "REMOVE_REDUNDANT", "REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL", "REMOVE_TIME_DEPENDENT"]
    
    name_mapping = {
        "REMOVE_REDUNDANT U REMOVE_ADVERSARIAL": "RR + RA",
        "REMOVE_ADVERSARIAL": "RA",
        "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL": "RR + RTD + RA",
        "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT": "RR + RTD",
        "REMOVE_REDUNDANT": "RR",
        "REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL": "RTD + RA",
        "REMOVE_TIME_DEPENDENT": "RTD"
    }

    short_names = [name_mapping[name] for name in var_names]
    
    metrics_list = []
    for var in var_names:
        with open(f'fromClusterRun4/models/{version}/{var}/metrics.pkl', 'rb') as f:
            metrics_dict = pickle.load(f) # deserialize using load()]
            metrics_list.append(metrics_dict)

    plot_comparisons(metrics_list, short_names, version=version, rotation=0, new_folder="act_vs_obs/")

    var_names = ["CHANGE_ACTION_REMOVE", "SET_ACTION_REMOVE"]

    metrics_list = []
    for var in var_names:
        with open(f'fromClusterRun4/models/{version}/{var}/metrics.pkl', 'rb') as f:
            metrics_dict = pickle.load(f) # deserialize using load()]
            metrics_list.append(metrics_dict)
    
    plot_comparisons(metrics_list, var_names, version=version, rotation=0, new_folder="change_vs_set/")

def collectV2():
    """Collects metrics to plot iteration 2 results
    """

    version = "v2"

    var_names = list(v2_spaces.REWARDS)
    metrics_list = []
    for var in var_names:
        with open(f'fromClusterRun4/models/{version}/{var}/metrics.pkl', 'rb') as f:
            metrics_dict = pickle.load(f) # deserialize using load()]
            metrics_list.append(metrics_dict)
    
    plot_comparisons(metrics_list, var_names, version="v2")

def collectV3():
    """Collects metrics to plot iteration 3 results
    """

    version = "v3.2"

    stack_var_names = ["2_stack", "3_stack", "4_stack"]
    metrics_list = []

    for var in stack_var_names:
        with open(f'fromClusterRun4/models/{version}/{var}/metrics.pkl', 'rb') as f:
            metrics_dict = pickle.load(f) # deserialize using load()]
            metrics_list.append(metrics_dict)
    
    version = "v3.1"
    var = "LSTM"
    stack_var_names.append(var)

    with open(f'fromClusterRun4/models/{version}/{var}/metrics.pkl', 'rb') as f:
            metrics_dict = pickle.load(f) # deserialize using load()]
            metrics_list.append(metrics_dict)
    
    var_names = stack_var_names
    
    plot_comparisons(metrics_list, var_names, version="v3", rotation=0)

if __name__=="__main__":
    collectV0()

    collectV1()

    collectV2()

    collectV3()
