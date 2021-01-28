[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4469437.svg)](https://doi.org/10.5281/zenodo.4469437)

# Introduction
The `lensing_haloes` module provides all the ingredients to reproduce the results from [arXiv:2101.07800](https://arxiv.org/abs/2101.07800).

Downloading the cluster samples from
[Zenodo](https://zenodo.org/record/4469437) is the fastest way to have
a look at the data. There you can find the generated cluster samples
for both stage III and stage IV-like surveys and their best-fitting
cosmologies.

## Installation
In the desired location, type
```
git clone https://github.com/StijnDebackere/lensing_haloes
cd lensing_haloes
pip install .
```

This should automatically install all requirements for the package to
work.

## Quickstart

If you want to generate data yourself, you should start by running
`lensing_haloes.results.save_halo_model()`, which will save the halo
model density profiles to the specified files. Then, running
`lensing_haloes.results.save_halo_model_lensing()` will generate the
lensing signal for the specified model and determine the best-fitting
NFW parameters.

This information is all that is needed to generate the biased mock
cluster samples. Start of by creating unbiased samples by calling
`lensing_haloes.cosmo.generate_mock_cluster_sample.generate_many_samples()`.
Then, these haloes can be biased by calling
`lensing_haloes.cosmo.generate_mock_cluster_sample.bias_samples_fbar()`.
Finally, the best-fitting cosmology can be determined by calling
`lensing_haloes.cosmo.fit_mock_cluster_sample.fit_maps_gaussian_mp()`
or
`lensing_haloes.cosmo.fit_mock_cluster_sample.fit_maps_poisson_mp()`,
depending on the preferred likelihood.

The plotting scripts for the different figures in the paper, can be
found in `lensing_haloes.plots.paper_plots`.
