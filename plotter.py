import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_obs_act_combos_grouped(metrics_dicts, variations, window_size=10):
    """
    Function to plot training and evaluation metrics for PPO and A2C models across different
    observation/action space variations with optional smoothing.
    
    Parameters:
        metrics_dicts (list): A list of metrics dictionaries, each corresponding to a variation of the observation/action space.
        variations (list): A list of strings representing the names of each variation.
        window_size (int): The size of the rolling window for smoothing the line plots.
    """

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
    for metrics_dict in metrics_dicts:
        # Extract PPO data
        ppo_rewards_smoothed = pd.Series(metrics_dict['PPO']['training_rewards']).rolling(window=window_size).mean()
        training_rewards_ppo.append(ppo_rewards_smoothed)
        eval_rewards_ppo.append(np.mean(metrics_dict['PPO']['eval_reward']))
        eval_lengths_ppo.append(np.mean(metrics_dict['PPO']['eval_length']))

        # Extract A2C data
        a2c_rewards_smoothed = pd.Series(metrics_dict['A2C']['training_rewards']).rolling(window=window_size).mean()
        training_rewards_a2c.append(a2c_rewards_smoothed)
        eval_rewards_a2c.append(np.mean(metrics_dict['A2C']['eval_reward']))
        eval_lengths_a2c.append(np.mean(metrics_dict['A2C']['eval_length']))
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Line plot comparing PPO training rewards across variations with smoothing
    for i, rewards in enumerate(training_rewards_ppo):
        axs[0, 0].plot(rewards, label=f'{variations[i]}')
    axs[0, 0].set_title(f'PPO Training Rewards Over Time (Window Size: {window_size})')
    axs[0, 0].set_xlabel('Episodes')
    axs[0, 0].set_ylabel('Rewards')
    axs[0, 0].legend()

    # Plot 2: Line plot comparing A2C training rewards across variations with smoothing
    for i, rewards in enumerate(training_rewards_a2c):
        axs[0, 1].plot(rewards, label=f'{variations[i]}')
    axs[0, 1].set_title(f'A2C Training Rewards Over Time (Window Size: {window_size})')
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Rewards')
    axs[0, 1].legend()

    # Bar width for grouped bar plots
    bar_width = 0.35
    index = np.arange(len(variations))
    
    # Plot 3: Grouped bar graph comparing avg eval rewards for PPO and A2C across variations
    axs[1, 0].bar(index, eval_rewards_ppo, bar_width, label='PPO')
    axs[1, 0].bar(index + bar_width, eval_rewards_a2c, bar_width, label='A2C')
    axs[1, 0].set_title('Average Evaluation Rewards')
    axs[1, 0].set_xlabel('Observation/Action Space Variations')
    axs[1, 0].set_ylabel('Avg Rewards')
    axs[1, 0].set_xticks(index + bar_width / 2)
    axs[1, 0].set_xticklabels(variations, rotation=45)
    axs[1, 0].legend()

    # Plot 4: Grouped bar graph comparing avg eval lengths for PPO and A2C across variations
    axs[1, 1].bar(index, eval_lengths_ppo, bar_width, label='PPO')
    axs[1, 1].bar(index + bar_width, eval_lengths_a2c, bar_width, label='A2C')
    axs[1, 1].set_title('Average Evaluation Episode Lengths')
    axs[1, 1].set_xlabel('Observation/Action Space Variations')
    axs[1, 1].set_ylabel('Avg Episode Lengths')
    axs[1, 1].set_xticks(index + bar_width / 2)
    axs[1, 1].set_xticklabels(variations, rotation=45)
    axs[1, 1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'plots/v1/comparisons.png')

def plot_obs_act_combos_separate(metrics_dicts, variations, window_size=10, version="v1"):
    """
    Function to plot training and evaluation metrics for PPO and A2C models across different
    observation/action space variations with optional smoothing.
    
    Parameters:
        metrics_dicts (list): A list of metrics dictionaries, each corresponding to a variation of the observation/action space.
        variations (list): A list of strings representing the names of each variation.
        window_size (int): The size of the rolling window for smoothing the line plots.
        version (str): The version of the plots for saving them to a unique directory.
    """

    # Ensure the directory for saving plots exists
    save_dir = f'plots/{version}/'
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
    for metrics_dict in metrics_dicts:
        # Extract PPO data
        ppo_rewards_smoothed = pd.Series(metrics_dict['PPO']['training_rewards']).rolling(window=window_size).mean()
        training_rewards_ppo.append(ppo_rewards_smoothed)
        eval_rewards_ppo.append(np.mean(metrics_dict['PPO']['eval_reward']))
        eval_lengths_ppo.append(np.mean(metrics_dict['PPO']['eval_length']))

        # Extract A2C data
        a2c_rewards_smoothed = pd.Series(metrics_dict['A2C']['training_rewards']).rolling(window=window_size).mean()
        training_rewards_a2c.append(a2c_rewards_smoothed)
        eval_rewards_a2c.append(np.mean(metrics_dict['A2C']['eval_reward']))
        eval_lengths_a2c.append(np.mean(metrics_dict['A2C']['eval_length']))

    # Plot 1: Line plot comparing PPO training rewards across variations with smoothing
    plt.figure(figsize=(20, 16))
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
    plt.figure(figsize=(20, 16))
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
    plt.figure(figsize=(20, 24))
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
    plt.xticks(index + bar_width / 2, variations, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_eval_rewards.png')
    plt.close()

    # Plot 4: Grouped bar graph comparing avg eval lengths for PPO and A2C across variations
    plt.figure(figsize=(20, 24))
    bars1 = plt.bar(index, eval_lengths_ppo, bar_width, label='PPO')
    bars2 = plt.bar(index + bar_width, eval_lengths_a2c, bar_width, label='A2C')

    # Adding value labels above the bars
    autolabel(bars1)
    autolabel(bars2)

    plt.title('Average Evaluation Episode Lengths')
    plt.xlabel('Observation/Action Space Variations')
    plt.ylabel('Avg Episode Lengths')
    plt.xticks(index + bar_width / 2, variations, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_eval_lengths.png')
    plt.close()


if __name__=="__main__":
    version = "v1"

    var_names = ["CHANGE_ACTION_REMOVE", "SET_ACTION_REMOVE", "REMOVE_REDUNDANT U REMOVE_ADVERSARIAL", "REMOVE_ADVERSARIAL",
                  "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL", "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT",
                  "REMOVE_REDUNDANT", "REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL", "REMOVE_TIME_DEPENDENT"]
    
    metrics_list = []
    for var in var_names:
        with open(f'fromCluster/models/{version}/{var}/metrics.pkl', 'rb') as f:
            metrics_dict = pickle.load(f) # deserialize using load()]
            metrics_list.append(metrics_dict)

    plot_obs_act_combos_separate(metrics_list, var_names)