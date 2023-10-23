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



## Manuscript figure key

### Main

Fig 1A (Song/locomotion example): 00_A5

Fig 1B (Multiple song examples): 10_A, 1_AM

Fig 1C (Raw Baker et al imaging data): 03_C1A

Fig 1D (LN/MA fits to baker data): 03_CM

Fig 1E (Example LN/MA naturalistic continuations): 03_DM

Fig 1F (SCHEMATIC)

Fig 1G (Example LN/MA predictions of female walking): 06_A1

Fig 1H (Bar chart of encoding model scores): 06_B


Fig 2A (Perturbed MA encoding model scores): 06_B

Fig 2B-D (Example perturbed MA predictions of female walking): 06_A2

Fig 2E (Female walking speed var expl vs number of neurons): 07_B

Fig 2F (Neural and female walking speed var expl vs num PCs): 07_C

Fig 2G (Example fast-adapt-slow-int MA pop trajectory): 08_B

Fig 2H (Example PC projections of fast-adapt-slow-int MA pop): 08_B

Fig 2I (Reconstruction of female walking speed from PCs): 07_E

Fig 2J (Female walking speed regression weights on PCs): 07_A3

Fig 2K (Example songs driving activity on top PCs): 07_A3

Fig 2L-M (Heterogeneity of adaptation vs integration): 07_G


Fig 3A (Accumulator response example): 10_A

Fig 3B (Example responses to many songs): 10_A

Fig 3C (Correlations of accumulation w song feats): 10_A

Fig 3D (Accumulator corrs vs MA params): 10_A

Fig 3E (Response distributions of MA neurons): 09_C

Fig 3F (Song info vs female locomotion predictability): 09_D


Fig 4A (SCHEMATIC)

Fig 4B-C (Population responses to many songs): 10_E

Fig 4D (Song separation over multiple timescales): 10_C

Fig 4E-F (Song trajectories in PC space): 10_E

Fig 4I (Song separation of PC projections): 10_D1



### Supplementary

Fig S1 (Example Baker et al vs Pacheco et al imaging data and MA params): 03_C1A, 03_C1B, 03_E, 03_F

Fig S2 (MA, LN, and Linear fit examples and error distributions): 03_CM

Fig S3 (Sine-offset responses): 03_CM

Fig S4 (Song examples and statistics): 01_A, 01_AM

Fig S5 (Predictions of alternate female behavioral variables and smoothing windows): 06_A1, 06_B, 06_B1

Fig S6 (Songs driving activity on top neural PCs): 07_A3

Fig S7 (PC interpretations via hand-picked song feats): 07_A3

Fig S8 (Greedily built MA population): 07_D, 08_C

Fig S9 (Response distributions/entropies across MA params): 09_C

Fig S10 (Response entropies vs MA params + extra example): 09_D, 09_C

Fig S11 (Greedily population response entropy): 09_F

Fig S12 (Pop trajectory distance traveled and song code variance): 10_C

Fig S13 (Linear projections of song-evoked MA activity): 11_A

Fig S14 (Strain-specific song and movement statistics): 01_K

Fig S15 (MA/LN comparison vs num trials): 06_D

Fig S16 (Female walking prediction from hand-picked features): 06_B

Fig S17 (Female walking prediction lternate LN formulations/fitting): 06_B2

Fig S18 (Linear filters and basis functions): 02_E

Fig S19 (Sine-offset fits and walking speed predictions): 06_B
