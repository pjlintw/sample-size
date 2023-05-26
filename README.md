# Revisiting Sample Size Determination in Natural Language Understanding
--

The code for replicating the experiments conducted in the ACL paper  "Revisiting Sample Size Determination in Natural Language Understanding". 

Note: The official code is a work in progress.


## Installation

### Python version

* Python == 3.8.8


## Quick Start

We provide the scripts and sample commands for reproducing our results in the paper.


### Training the models 

To assess the empirical performance, we trained our models on a range of data sizes, from 1% to 100%. The command below can be used to train models on different data sizes from the IMDb dataset, using a Transformer encoder with 6 layers.


```
python3 sample_selector.py 1% 2% 3% 4% 5% 6% 7% 8% 9% 10% 13% 17% 21% 25% 28% 34% 40% 43% 47% 50% 55% 60% 65% 70% 80% 85% 90% 95% 100% \
  --nlayers 6 \
  --n_epochs 200 \
  --output_file results/test.txt \
  --train_model True
```

To use datasets other than IMDb, you can modify the value of the 'dataset' variable in the 'datautils.py' file on line 27. Simply change the value to 'SST2', 'AG_NEWS', or 'DBpedia' depending on the dataset you wish to use.

### Learning curve fitting on 10%

We fit learning curves for EXP, Inverse, POW4, and Ensemble by running the script 'sample_selector.py' with the argument `--extrapolating_training_index` set to 10. The script fits the learning curve for data sizes ranging from 1% to 10%, and plots results using a 10% sample size. To run the script, enter the following command:


```
python3 sample_selector.py 1% 2% 3% 4% 5% 6% 7% 8% 9% 10% 13% 17% 21% 25% 28% 34% 40% 43% 47% 50% 55% 60% 65% 70% 80% 85% 90% 95% 100% \
  --nlayers 6 \
  --n_epochs 200 \
  --output_file results/test.txt \
  --plot_extrapolating True \
  --extrapolating_training_index 10 \
  --show_percentage_labels True
```



### Learning curve fitting on 50%

Similarly, the script support learning curve fitting on various sample sizes. One can use `extrapolating_training_index` as `20` for fitting the curves on 50% data. For plotting using 50% sample size: 

```
python3 sample_selector.py 1% 2% 3% 4% 5% 6% 7% 8% 9% 10% 13% 17% 21% 25% 28% 34% 40% 43% 47% 50% 55% 60% 65% 70% 80% 85% 90% 95% 100% \
  --nlayers 6 \
  --n_epochs 200 \
  --output_file results/test.txt \
  --plot_extrapolating True \
  --extrapolating_training_index 20 \
  --show_percentage_labels True
```

### Plotting MAE 

Our evaluation metric for learning curves is the mean absolute error (MAE). The following command can be used to plot the MAE on different sample sizes, ranging from 1% to 50%:

```
python3 sample_selector.py 1% 2% 3% 4% 5% 6% 7% 8% 9% 10% 13% 17% 21% 25% 28% 34% 40% 43% 47% 50% 55% 60% 65% 70% 80% 85% 90% 95% 100% \
  --nlayers 6 \
  --n_epochs 200 \
  --output_file results/test.txt \
  --plot_mae True \
  --show_percentage_labels True
```





