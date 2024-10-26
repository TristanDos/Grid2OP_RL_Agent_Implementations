import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import v2_spaces

def plot_v0_average(metrics_dicts):
    model_names = metrics_dict.keys()
    eval_rewards_arr = []
    eval_lengths_arr = []
    training_rewards_dict = {}
    training_lengths_dict = {}

    for model_name in model_names:
        eval_reward = 0
        eval_length = 0
        training_rewards = [0] * 2000
        training_lengths = [0] * 2000

        for met in metrics_dicts:
            metrics = met[model_name]

            eval_reward += metrics['eval_reward']
            eval_length += metrics['eval_length']
            training_rewards += metrics['training_rewards']
            training_lengths += metrics['training_episode_lengths']
        
        eval_reward /= 3
        eval_length /= 3

        new_training_rewards = []
        new_training_lengths = []

        for step in range(len(training_rewards)):
            if training_lengths[step] != 0:
                new_training_rewards.append(training_rewards[step] / 3)
                new_training_lengths.append(training_lengths[step] / 3)
            else:
                break

        eval_rewards_arr.append(eval_reward)
        eval_lengths_arr.append(eval_length)

        if model_name != "Random":
            training_rewards_dict[model_name] = new_training_rewards
            training_lengths_dict[model_name] = new_training_lengths

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, eval_rewards_arr, capsize=10)
    # sns.barplot(model_names, eval_rewards_arr, capsize=10)
    plt.title('Average Model Returns During Evaluation')
    plt.ylabel('Average Return')
    plt.ylim(bottom=0)
    for i, v in enumerate(eval_rewards_arr):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'plots/{version}/eval_rewards.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, eval_lengths_arr, capsize=10)
    # sns.barplot(model_names, eval_lengths_arr, capsize=10)
    plt.title('Average Model Episode Lengths During Evaluation')
    plt.ylabel('Average Episode Length (Number of Steps)')
    plt.ylim(bottom=0)
    for i, v in enumerate(eval_lengths_arr):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.savefig(f'plots/{version}/eval_lengths.png')
    plt.close()

    plt.figure(figsize=(10, 6))

    counter = 0
    markers = ['o', 's']
    for model_name, training_rewards in training_rewards_dict.items():
        plt.plot(training_rewards, label=model_name, marker=markers[counter], linestyle='-')
        # sns.lineplot(training_rewards, label=model_name, marker=markers[counter], linestyle='-')
        
        counter += 1
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Rewards Accrued During Training for {[model_name for model_name in training_rewards_dict.keys()]}')
    plt.legend()
    plt.savefig(f'plots/{version}/training_rewards.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    
    counter = 0
    markers = ['o', 's']
    for model_name, training_lengths in training_lengths_dict.items():
        plt.plot(training_lengths, label=model_name, marker=markers[counter], linestyle='-')
        # sns.lineplot(training_lengths, label=model_name, marker=markers[counter], linestyle='-')
        
        counter += 1

    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')
    plt.title(f'Episode Length During Training for {[model_name for model_name in training_lengths_dict.keys()]}')
    plt.legend()
    plt.savefig(f'plots/{version}/training_lengths.png')
    plt.close()

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

def plot_obs_act_combos_separate(metrics_dicts, variations, window_size=30, version="v1"):
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
    plt.figure(figsize=(16, 10))
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
    plt.figure(figsize=(16, 10))
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
    plt.figure(figsize=(18, 10))
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
    plt.xticks(index + bar_width / 2, variations, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_eval_rewards.png')
    plt.close()

    # Plot 4: Grouped bar graph comparing avg eval lengths for PPO and A2C across variations
    plt.figure(figsize=(18, 10))
    bars1 = plt.bar(index, eval_lengths_ppo, bar_width, label='PPO')
    bars2 = plt.bar(index + bar_width, eval_lengths_a2c, bar_width, label='A2C')

    # Adding value labels above the bars
    autolabel(bars1)
    autolabel(bars2)

    plt.title('Average Evaluation Episode Lengths')
    plt.xlabel('Observation/Action Space Variations')
    plt.ylabel('Avg Episode Lengths')
    plt.xticks(index + bar_width / 2, variations, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparisons_eval_lengths.png')
    plt.close()

if __name__=="__main__":
    # version = "v2"

    # var_names = list(v2_spaces.REWARDS)
    # metrics_list = []
    # for var in var_names:
    #     with open(f'fromCluster(Iterations0,1,2)/models/{version}/{var}/metrics.pkl', 'rb') as f:
    #         metrics_dict = pickle.load(f) # deserialize using load()]
    #         metrics_list.append(metrics_dict)
    
    # plot_obs_act_combos_separate(metrics_list, var_names, version="v2")

    # version = "v1"

    # var_names = ["REMOVE_REDUNDANT U REMOVE_ADVERSARIAL", "REMOVE_ADVERSARIAL",
    #               "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL", "REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT",
    #               "REMOVE_REDUNDANT", "REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL", "REMOVE_TIME_DEPENDENT"]
    
    # metrics_list = []
    # for var in var_names:
    #     with open(f'fromCluster(Iterations0,1,2)/models/{version}/{var}/metrics.pkl', 'rb') as f:
    #         metrics_dict = pickle.load(f) # deserialize using load()]
    #         metrics_list.append(metrics_dict)

    # plot_obs_act_combos_separate(metrics_list, var_names)

    version = "v0"

    metrics_list = []
    for i in range(3):
        with open(f'fromCluster/models/{version}/metrics{i}.pkl', 'rb') as f:
            metrics_list.append(pickle.load(f))
    
    plot_v0_average(metrics_list)

