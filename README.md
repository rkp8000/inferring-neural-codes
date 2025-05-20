# Source code for "Inferring neural population codes for *Drosophila* acoustic communication."

Pang, Rich, Christa A. Baker, Mala Murthy, Jonathan Pillow.

https://www.pnas.org/doi/10.1073/pnas.2417733122

## Summary of directory organization

Modules:

disp.py: Plotting functions.

aux.py: Auxiliary functions, e.g. data loading, basic reformatting, etc.

my_stats.py: Simple statistical functions.

time_stats.py: Functions for temporal statistics.

my_torch.py: Functions for fitting neural -> behavior models.

record_0_main.py: Simulation functions to record artificial neural activity from NA and NA-matched LN model.

record_1_ma_ext.py: Simulation functions to record artificial neural activity from NA variations.

record_3_rsvr.py: Simulation functions to record artificial neural activity from reservoir network.

Note: For historical reasons many instances of "NA" are written as "MA" in the codebase.


## Notebook organization

All figures made in notebooks, grouped by functionality.

00 - Raw data visualization and reformatting.

01 - Basic song and movement statistics.

02 - Non-neural models for predicting behavior.

03 - Fitting encoding models to neural calcium imaging data.

04 - Simulating artificial neural activity from fit neural models.

05 - Predicting behavior from artificial neural recordings.

06 - Comparing behavioral predictions from different models.

07 - Behavioral predictions vs artificial neural population size and neural PCs.

08 - Example model neuron time-series in response to song.

09 - Info-theoretic analyses.

10 - Analyses of natural-song accumulation dynamics in MA model.

20 - Analyses for selecting best natural songs to present for Ca imaging experiments.

A - Equations, etc.

N - Analyses of initial neuroimaging data of female brain activity in response to natural song.

S - Supplementary figures/analyses.


## Manuscript figure key

The project contains several auxiliary notebooks/plots that were not included in the manuscript. The notebooks used to produce the specific figures in the main text and supplement are given below.

### Main

Fig 1A (Song/locomotion example): 00_A5

Fig 1B (Song durations, mode durations, song mode frequencies): 01_A

Fig 1C (Multiple song examples): 10_A

Fig 1D (SCHEMATIC)

Fig 1E (Baker et al imaging data with LN/NA fits): 03_C2

Fig 1G (Example LN/NA naturalistic continuations): 03_D

Fig 1H (Example LN/NA predictions of female walking): 06_A1

Fig 1I (Bar chart of encoding model scores): 06_B


Fig 2A (Perturbed MA encoding model scores): 06_B

Fig 2B-D (Example perturbed MA predictions of female walking): 06_A2

Fig 2E (Female walking speed var expl vs number of neurons): 07_B

Fig 2F (Neural and female walking speed var expl vs num PCs): 07_C

Fig 2G (Example fast-adapt-slow-int NA pop trajectory): 08_B

Fig 2H (Example PC projections of fast-adapt-slow-int MA pop): 08_B

Fig 2I (Reconstruction of female walking speed from PCs): 07_E


Fig 3A (SCHEMATIC + female walking speed rgr weights on PCs): 07_A3

Fig 3B (Example songs driving activity on top PCs): 07_A3


Fig 4A (Example single-neuron response distributions): 09_C

Fig 4B (Song info vs NA params): 09_D

Fig 4C (Female walking speed var expl vs NA params): 09_D

Fig 4D (Song info vs female walking speed var expl): 09_D


Fig 5A (SCHEMATIC)

Fig 5B (State space representation of responses to many songs): 10_E

Fig 5C (State space represnation of example song-evoked trajs): 10_E

Fig 5D (Song separation over multiple timescales): 10_C

Fig 5E (Example song-output transformations): 11_C

Fig 5F (Spatial signatures of motifs): 11_A

Fig 5G (SCHEMATIC)

Fig 5H (Selective accumulation of temporal motifs): 11_B



### Supplementary

Fig S1A (Example Baker et al raw responses): 03_C1A

Fig S1B (Example Pacheco et al raw responses): 03_C1B

Fig S1C (Baker et al fit NA params): 03_E

Fig S1D (Pacheco et al fit NA params): 03_F


Fig S2A-D (More LN and NA fit examples): 03_C3

Fig S2E-F (Neural data fit errors for alternate LN fitting schemes): 03_C2

Fig S2G (Female walking speed var explained from alternate LN models): 06_B2


Fig S3 (LN and NA fits to sine-offset responses): 03_C3


Fig S4A (1-min Female walking speed var explained vs neural models): 06_B

Fig S4B-C (Female forward and lateral speed var explained vs neural models): 06_B1

Fig S4E (Example LN vs NA predictions of 1-min smoothed female walking speed): 06_A1

Fig S4F (1-min smoothed female walking speed var expl vs perturbations): 06_B


Fig S5A (More song examples): 01_A

Fig S5B (Singing/quiet durations): 01_A

Fig S5C (Mode segment durations): 01_A

Fig S5D (Singing/quiet autocovariance): 01_A3

Fig S5E (Sine/pulse cross covariance): 01_A4


Fig S6 (Cross-spectral densities between true and predicted female walking speed): 06_A5


Fig S7A (Example fast-adapt/slow-int response): 03_G

Fig S7B (Fast-adapt/slow-int regime): 03_F


Fig S8 (Heterogeneity of adaptation vs integration): 07_G


Fig S9 (More song examples for each neural PC + stats): 07_A3


Fig S10 (Response distributions + entropies for more example NA neurons): 09_C


Fig S11 (Mutual info vs time lag for example NA neurons): 09_E


Fig S12A-B (Song-evoked trajecs along neural PCs): 10_E

Fig S12C (Song-evoked trajec separation along neural PCs): 10_C


Fig S13 (Param sweep for song-to-output-sine-wave transformations): 11_C


Fig S14 (Temporal pattern accum w traditional reservoir computer): 11_B1


Fig S15A (NA-Sine-Rebound fits to sine-offset responses): 03_C2A

Fig S15B (Behavioral prediction including sine-offset neurons): 06_B


Fig S16 (Example accumulation of natural but not block song): 10_A


Fig S17 (NA vs LN fits vs number of courtship sessions used): 06_D


Fig S18 (Female walking speed var explained vs hand-picked features): 06_B


Fig S19 (Song->female walking speed lin filters + basis functions): 02_E


Fig S20A (Female walking speed var explained by state space model [S5]): 02_J

Fig S20B (Female walking speed var explained by feed-fwd network): 02_K

Fig S20C (Female walking speed var explained by LSTM): 02_L


Fig S21 (Female walking speed var explained by reservoir computer): 06_B
