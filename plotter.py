import matplotlib.pyplot as plt
import numpy as np
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_obs_act_space_metrics(metrics_dicts, variations, window_size=10):
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

# Example usage:
# metrics_dict = load from a pkl file or defined earlier
# plot_grid2op_metrics(metrics_dict)

if __name__=="__main__":
    version = "v1"

    metrics_list = []
    
    with open(f'models/{version}/metrics.pkl', 'rb') as f:
        metrics_dict = pickle.load(f) # deserialize using load()]
        metrics_list.append(metrics_dict)

    var_names = ["Baseline", "Variation 1", "Variation 2", "Variation 3", "Variation 4",]
    plot_obs_act_space_metrics(metrics_list, var_names)