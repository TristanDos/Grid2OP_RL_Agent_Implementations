import json

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

# Convert into JSON
# File name is mydata.json
with open("kept_observations.json", "w") as final:
	json.dump(obs_attr_to_keep, final)
	
# File name is mydata.json
with open("kept_actions.json", "w") as final:
	json.dump(act_attr_to_keep, final)

# Download the file
# files.download('mydata.json')