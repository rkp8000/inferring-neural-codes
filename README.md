# Source code for Pang et al

"Inferring neural codes from natural behavior in *Drosophila* social communication."

(in preparation)


## Summary of directory organization

Modules:

disp.py: Plotting functions.

aux.py: Auxiliary functions, e.g. data loading, basic reformatting, etc.

my_stats.py: Simple statistical functions.

time_stats.py: Functions for temporal statistics.

my_torch.py: Functions for fitting neural -> behavior models.

record_0_main.py: Simulation functions to record artificial neural activity from MA and MA-matched LN model.

record_1_ma_ext.py: Simulation functions to record artificial neural activity from MA variations.

record_2_lin_ln.py: Simulation functions to record artificial neural activity from lin/LN models.


## Notebook organization

All figures made in notebooks, grouped by functionality.

00 - Raw data visualization and reformatting.

01 - Basic song and movement statistics.

02 - Non-neural models for predicting behavior.

03 - Fitting encoding models to neural calcium imaging data.

04 - Recording artificial neural activity from fit neural models.

05 - Predicting behavior from artificial neural recordings.

06 - Comparing behavioral predictions from different models.

07 - Behavioral predictions vs artificial neural population size.

08 - Comparisons of single-neuron predictions of behavior vs parameters.

09 - Compression/information-theoretic analyses.

10 - Analyses of natural-song accumulation dynamics in MA model.

21 - Auxiliary analyses/datasets for training artificial RNN to predict behavior from song.

A - Equations, etc.

N - Analyses of initial neuroimaging data of female brain activity in response to natural song.

S - Supplementary figures/analyses.

Z - Some figures for specific meetings/talks/etc.

