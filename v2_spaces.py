import itertools

# action difference sets
SET_ACTION_REMOVE = {'set_bus', 'set_line_status'}

CHANGE_ACTION_REMOVE = {'change_bus', 'change_line_status'}

# observation difference sets
REMOVE_REDUNDANT = {
    'curtailment_limit_effective',
    'curtailment_limit_mw',
    'curtailment_mw',
    'gen_margin_down',
    'gen_margin_up',
    'gen_p_before_curtail',
    'gen_q',
    'gen_theta',
    'gen_v',
    'load_q',
    'load_theta',
    'load_v',
    'prod_q',
    'prod_v',
    'storage_power_target',
    'storage_theta',
}

REMOVE_TIME_DEPENDENT =  {
    'current_step',
    'day',
    'day_of_week',
    'delta_time',
    'duration_next_maintenance',
    'hour_of_day',
    'minute_of_hour',
    'month',
    'time_before_cooldown_line',
    'time_before_cooldown_sub',
    'time_next_maintenance',
    'time_since_last_alarm',
    'time_since_last_alert',
    'time_since_last_attack',
    'year',
}

REMOVE_ADVERSARIAL = {
    'active_alert',
    'actual_dispatch',
    'alert_duration',
    'attack_under_alert',
    'time_before_cooldown_line',
    'time_before_cooldown_sub',
    'time_next_maintenance',
    'time_since_last_alarm',
    'time_since_last_alert',
    'time_since_last_attack',
    'total_number_of_alert',
    'was_alarm_used_after_game_over',
    'was_alert_used_after_attack',
}


REWARDS = {
    'stability', 'economic', 'comprehensive'
}

class Variation:
    def __init__(self, act_attr_to_rmv=set(), obs_attr_to_rmv=set(), reward_type='default'):
        self.playable_actions = {'change_bus', 'change_line_status', 'curtail', 'redispatch', 'set_bus', 'set_line_status'}
        self.act_attr_to_rmv = act_attr_to_rmv
        self.act_attr_to_keep = self.playable_actions - act_attr_to_rmv
        
        self.complete_observation = {
            'a_or',
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
            'year'
        }
        self.obs_attr_to_rmv = obs_attr_to_rmv
        self.obs_attr_to_keep = self.complete_observation - obs_attr_to_rmv

        self.env_configs = {
            'Random': {"obs_attr_to_keep": list(self.obs_attr_to_keep), 
                       'act_type': "discrete", 
                       "act_attr_to_keep": list(self.act_attr_to_keep),
                       "reward_type": reward_type},
            'PPO': {"obs_attr_to_keep": list(self.obs_attr_to_keep), 
                    'act_type': "discrete", 
                    "act_attr_to_keep": list(self.act_attr_to_keep),
                    "reward_type": reward_type},
            'A2C': {"obs_attr_to_keep": list(self.obs_attr_to_keep), 
                    'act_type': "discrete", 
                    "act_attr_to_keep": list(self.act_attr_to_keep),
                    "reward_type": reward_type}
            }

    def get_attributes(self):
        return self.env_configs


def combinatorial_unions(named_sets):
    combinations = {}
    keys = list(named_sets.keys())
    sets = list(named_sets.values())
    
    # Loop over all possible lengths of combinations (0 to N)
    for r in range(1, len(sets) + 1):
        # For each combination of size r, compute the union
        for combination in itertools.combinations(enumerate(sets), r):
            indices = [i for i, _ in combination]  # Get the indices of the sets in the combination
            names = [keys[i] for i in indices]     # Get the corresponding names
            
            # Generate a name for the combination (join with " U ")
            combination_name = " U ".join(names) if names else "empty"
            
            # Compute the union of the sets in the combination
            union_result = set().union(*(s for _, s in combination))
        
            combinations[combination_name] = union_result
    
    return combinations


action_subspaces = {
    'SET_ACTION_REMOVE': SET_ACTION_REMOVE,
    'CHANGE_ACTION_REMOVE': CHANGE_ACTION_REMOVE
}

observation_subspaces = combinatorial_unions({
    'REMOVE_REDUNDANT': REMOVE_REDUNDANT,
    'REMOVE_TIME_DEPENDENT': REMOVE_TIME_DEPENDENT, 
    'REMOVE_ADVERSARIAL': REMOVE_ADVERSARIAL
})
