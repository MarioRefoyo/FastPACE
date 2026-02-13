# FastPACE
This repository contains the code of "FastPACE: Fast PlAnning of Counterfactual Explanations for Time Series Classification".

FastPACE is a model-agnostic counterfactual generation method designed to mitigate the trade-off between explanation quality and execution times.
FastPACE always returns a valid counterfactual by design and is computationally efficient, while also delivering high quality CFEs, 
even surpassing the current state-of-the-art methods in terms of explanation quality. 
As a result, FastPACE makes counterfactual explanations more practical in real world settings with long and multi dimensional inputs. 

FastPACE casts counterfactual generation as an episodic Markov Decision Process (MDP), solving it with the Cross-Entropy Method (CEM), 
a lightweight look-ahead planning approach. 
On the other hand, it targets efficiency by applying aggregated modifications to the input instance, jointly modifying contiguous temporal segments 
and groups of similar channels, thus encouraging coherent and plausible interventions. This is done in a hierarchical, 
multi-granularity manner: starting from coarse edits that quickly locate influential regions and progressively refining them to obtain sparse,
localized modifications. Finally, FastPACE enforces validity by design, ensuring that generated counterfactuals consistently achieve
the desired target prediction.

![FastPACE_diagram.png](FastPACE_diagram.png)

# Source Code
The source code of FastPACE can be found in `./methods/FastPaCECF.py`. The class within this file extends
`CounterfactualMethod` in `./methods/counterfactual_common.py`, a wrapper class to normalize the interfaces of all methods
included in the experiments. Additionally, `FastPACECF.py` utilizes auxiliary classes for environment definition, 
hierarchical clustering, and planning, which can be found under `./methods/FastPACE/*`.

# Experiments
To reproduce the experiments, train the black-box classifiers and Outlier Detection models by running 
`./experiments/models/train_pytorch.py`,  `./experiments/models/train_ae_model_pytorch.py`, `./experiments/models/train_if_model.py` 
and `./experiments/models/train_lof_model.py`. This will create a folder per dataset in `./experiments/models/` containing the models and the parameters.

Then, run `./experiments/ab-cf.py`, `./experiments/comte.py`, `./experiments/discox.py`, `./experiments/multispace.py`, `./experiments/ng.py` to 
obtain the results of the baseline methods, and `./experiments/fastpace.py` to obtain the results of FastPACE. 
Counterfactual Explanations will be stored in `./experiments/results/{DATASET}/{EXPERIMENT_NAME}` using pickle format.

To visualize results tables, execute `./experiments/evaluation/visualize_counterfactuals_mo_multivariate.ipynb`
