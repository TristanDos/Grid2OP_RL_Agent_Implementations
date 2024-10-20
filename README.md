# Grid2OP_RL_Agent_Implementations

## Authors

Tristan Dos Remendos (2465830)

Gabriel Drew (2362193)

## File Structure

The following files refer to the implementations for each iteration. (models_v{iteration_num})

```
models_v0.py
models_v1.py
models_v2.py
```

`plotter.py` , `callbacks.py` and `v1_spaces.py` are supplementary files containing classes used in models_v*.py files.

Subfolders are locations for outputs from the various python files:

* models (Contains zipped sb3 models and pkl files for evaluation outcomes)

  * v0 - Baseline outputs
  * v1 - Iteration 1 outputs
  * v2 - Iteration 2 outputs
* plots (Contains images generated using matplotlib)

  * v0 - Baseline outputs
  * v1 - Iteration 1 outputs
  * v2 - Iteration 2 outputs
