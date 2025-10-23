# Fitting calcium imaging ground truth
This is an attempt to fit different models to calcium imaging data with simultaneously juxtacellularly recorded action potentials from the same neuron.

## The ground truth
Ground truth from an example neuron is included (`CAttached_jGCaMP8s_472182_2_mini.mat`). More ground truth is available via the [Cascade repository](https://github.com/HelmchenLabSoftware/Cascade/tree/master/Ground_truth) and described in [Rupprecht et al. (2021)](https://www.nature.com/articles/s41593-021-00895-5) and [Rupprecht et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.03.03.641129v3). How to use and load this ground truth data in Python or Matlab is described in the [repository's FAQ](https://github.com/HelmchenLabSoftware/Cascade?tab=readme-ov-file#how-can-i-inspect-the-ground-truth-datasets). For questions about how to use and interpret this dataset, raise an issue or reach out via [e-mail](mailto:p.t.r.rupprecht+cascade@gmail.com).

## The models
I have implemented three models that describe the relationship between action potentials and the resulting fluorescence of the calcium indicator: 1) simple Hill nonlinearity, 2) Greenberg model, 3) Broussard model.

1. The simple nonlinear model was adapted from [Deneux et al. (2016)](https://www.nature.com/articles/ncomms12190). It is a simple linear model with a Hill-type nonlinearity on top. It is simple, easy and fast to fit, and readily interpretable.

2. The Greenberg model was implemented following [Greenberg et al. (2018)](https://www.biorxiv.org/content/10.1101/479055v1). Instead of a simple instantaneous nonlinearity, it implements a history dependent sequential binding model with (semi-)hidden states. It is difficult to fit (due to parameter redundancy) and takes some time to gain an intuition.

3. The Broussard model was implemented based on a recent preprint by [Broussard et al. (2025)](https://www.biorxiv.org/content/10.1101/2024.12.31.630967v2). The model implementation is not yet publicly released, and so this implementation of their model might be off to some extent. The idea behin this model is to have a biophysically detailed model which, in particular, models a slowing down of the indicator kinetics when calcium ions are bound ("use-dependent slowing"). The model is difficult to fit, and I have the impressions that the parameters are also redundant and therefore there might not be best-fit parameter combinations.

## The scripts
To fit the models to the ground truth, I have tried three different approaches. First, I used [simulation-based inference](https://sbi-dev.github.io/sbi/0.22/) (SBI), a powerful framework to not only fit a model to data, but also to obtain uncertainty measures of the fits. Second, I used a simpler approach inspired by the [Jaxley preprint](https://www.biorxiv.org/content/10.1101/2024.08.21.608979v2.full) to fit the model using differentiable optimization, resulting in a point estimate of model parameters. Third, I used an interactive graphical user interface where the model parameters can be adjusted manually to explore the best parameters and the fits to the data.

The scripts should be pretty self-explanatory. I would recommend using the simplest model for exploration:

`python interactive_Hill_nonlinearity.py`

or

`python jax_Hill_nonlinearity.py --mat CAttached_jGCaMP8s_472182_2_mini.mat`

## Dependencies and environment
To run the scripts, a few packages need to be installed, which can be figured out relatively easily from the first lines of each script. I ran all scripts on a simple laptop CPU.

## Observations
In the beginning, the results look relatively good. Here's the fit for the `python jax_Hill_nonlinearity.py` command as noted above:

<img alt="image" src="https://github.com/user-attachments/assets/89cbb0f6-d19c-4f11-9971-afb445b89e93"  width="85%"/>

However, upon closer inspection, there are many problems with the reliable fitting of small events or the timescales of small events. It seemed in my hands very challenging to provide a good fit for both small, intermediate, and large events. The model (and also the more complex models) seemed unable to capture these different kinetics and amplitudes for small events vs. during events when the cell was in a state of high firing rates.

<img alt="image" src="https://github.com/user-attachments/assets/a2f946c7-5e52-44bf-9620-33bab0d4b560"  width="85%"/>

As can be seen, a very large fraction of the variance could be explained for this specific neuron (`Corr 0.974`). However, this large values occludes the fact that the model is a poor fit for small events (which contribute only little to the overall variance of the calcium signal but are important from a neurophysiological perspective).

## Possible next steps
I have the impression that the models I used here are either too unflexible (`Hill_nonlinearity`) or too unconstrained. It could be a good idea to use a procedure to find a heuristic model that is a good fit to the data. (The approach shown by [Aygün et al. (2025)](https://arxiv.org/abs/2509.06503) could be a good way to do so, but I'm not an expert for this method.) Such a model could then be used to generate artificial and specific data, which in turn could be further used to derive measures to quantify nonlinearity, like the Hill exponent and the [Kd](https://www.scientifica.uk.com/learning-zone/how-to-compute-%CE%B4f-f-from-calcium-imaging-data) value.

