# Grid2OP_RL_Agent_Implementations

## Authors

Tristan Dos Remendos (2465830)

Gabriel Drew (2362193)

## File Structure

The following files refer to the implementations for each iteration. (models_v{iteration_num}.py)

```
models_v0.py
models_v1.py
models_v2.py
models_v3.1.py
models_v3.2.py
```

`plotter.py` , `callbacks.py` , `v1_spaces.py` and `v2_spaces.py` are supplementary files containing classes used in models_v*.py files.

`run.sh` and `job.batch` are files used for running these files on the Wits Cluster.

---

Subfolders are locations for outputs from the various python files:

* models (Contains zipped sb3 models and pkl files for evaluation outcomes)

  * v0 - Baseline outputs
  * v1 - Iteration 1 outputs (Comparing different action and observation spaces)
    * CHANGE_ACTION_REMOVE
    * REMOVE_ADVERSARIAL
    * REMOVE_REDUNDANT
    * REMOVE_REDUNDANT U REMOVE_ADVERSARIAL
    * REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT
    * REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL
    * REMOVE_TIME_DEPENDENT
    * REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL
    * SET_ACTION_REMOVE
  * v2 - Iteration 2 outputs (Comparing different reward functions)
    * comprehensive
    * economic
    * stability
  * v3.1 - Iteration 3 outputs (Assessing LSTM Policy Network)
    * LSTM
  * v3.2 - Iteration 3 outputs (Comparing stack sizes for vector frame stacking)
    * 2_stack
    * 3_stack
    * 4_stack
* plots (Contains images generated using matplotlib)

  * v0 - Plots of baseline results
  * v1 - Plots of iteration 1 results (Comparing different action and observation spaces)

    * act_vs_obs (plots comparing observation spaces using a selected action space)
    * change_vs_set (plots comparing different action space reductions - set vs change actions removed)
    * CHANGE_ACTION_REMOVE (singular plots)
    * REMOVE_ADVERSARIAL (singular plots)
    * REMOVE_REDUNDANT (singular plots)
    * REMOVE_REDUNDANT U REMOVE_ADVERSARIAL (singular plots)
    * REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT (singular plots)
    * REMOVE_REDUNDANT U REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL (singular plots)
    * REMOVE_TIME_DEPENDENT (singular plots)
    * REMOVE_TIME_DEPENDENT U REMOVE_ADVERSARIAL (singular plots)
    * SET_ACTION_REMOVE (singular plots)
  * v2 - Iteration 2 outputs (Comparing different reward functions)

    * comprehensive (singular plots)
    * economic (singular plots)
    * stability (singular plots)
  * v3 - Combines all iteration 3 outputs into 4 comparison plots
  * v3.1 - Iteration 3 outputs (Assessing LSTM Policy Network)

    * LSTM (singular plots)
  * v3.2 - Iteration 3 outputs (Comparing stack sizes for vector frame stacking)

    * 2_stack (singular plots)
    * 3_stack (singular plots)
    * 4_stack (singular plots)

* results (Contains text files of the evaluation results for every iteration)
  
## Running all iterations

`install_imports.bash` contains the pip commands for installing all of the necessary libraries for running the files.

---

To run everything, use:

```
chmod +rwx *.bash
./run_models.bash
```

---

This will generate all of the .pkl files and plots in the folders specified in [File Structure](https://github.com/TristanDos/Grid2OP_RL_Agent_Implementations/tree/main?tab=readme-ov-file#file-structure "Specifies directory layout").

Text files containing all of the evaluation results for each iteration will also be generated in the results folder in the following format:

`results_{version number}_{variation name}.txt`
