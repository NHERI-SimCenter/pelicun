<img src="https://raw.githubusercontent.com/NHERI-SimCenter/pelicun/master/docs/figures/logo.PNG" alt="pelicun" height="100"/>

[![Documentation Status](https://readthedocs.org/projects/pelicun/badge/?version=latest)](http://pelicun.readthedocs.io/en/latest/?badge=latest)
[![TravisCI](https://travis-ci.org/NHERI-SimCenter/pelicun.svg?branch=master)](https://travis-ci.org/NHERI-SimCenter/pelicun)
[![Coverage Status](https://coveralls.io/repos/github/NHERI-SimCenter/pelicun/badge.svg?branch=master)](https://coveralls.io/github/NHERI-SimCenter/pelicun?branch=master)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1489230.svg)](https://doi.org/10.5281/zenodo.1489230)

Probabilistic Estimation of Losses, Injuries, and Community resilience Under Natural disasters

## What is it?

`pelicun` is a Python package that provides tools for assessment of damage and losses due to natural hazards. It uses a stochastic damage and loss model that is based on the methodology described in FEMA P58 (FEMA, 2012). While FEMA P58 aims to assess the seismic performance of a building, with `pelicun` we want to provide a more versatile, hazard-agnostic tool that will eventually provide loss estimates for other types of assets (e.g. bridges, facilities, pipelines) and lifelines. The underlying loss model was designed with these objectives in mind and it will be gradually extended to have such functionality.

Currently, the scenario assessment from the FEMA P58 methodology is built-in the tool. Detailed documentation of the available methods and their use is available at http://pelicun.readthedocs.io

## What can I use it for?

The current version of `pelicun` can be used to quantifiy lossess from an earthquake scenario in the form of *decision variables*. This functionality is typically utilized for performance based engineering or seismic risk assessment. There are several steps of seismic performance assessment that `pelcicun` can help with:

- **Describe the joint distribution of seismic response.** The response of a structure or other type of asset to an earthquake is typically described by so-called *engineering demand parameters* (EDPs). `pelicun` provides methods that take a finite number of EDP vectors and find a multivarite distribution that describes the joint distribution of EDP data well.

- **Define the damage and loss model of a building.** The component damage and loss data from FEMA P58 is provided with `pelicun`. This makes it easy to define building components without having to provide all the data manually. The stochastic damage and loss model is designed to facilitate modeling correlations between several parameters of the damage and loss model.

- **Estimate component damages.** Given a damage and loss model and the joint distribution of EDPs, `pelicun` provides methods to estimate the quantity of damaged components and collapses.

- **Estimate consequences.** Using information about collapses and component damages, the following consequences can be estimated with the loss model: reconstruction cost and time, unsafe placarding (red tag), injuries and fatalities. 

## Why should I use it?

1. It is free and it always will be. 
2. It is open source. You can always see what is happening under the hood.
3. It is efficient. The loss assessment calculations in `pelicun` use `numpy` and `scipy` libraries to efficiently propagate uncertainties and provide detailed results quickly.
4. You can trust it. Every function in `pelicun` is tested after every commit. See the Travis-CI and Coveralls badges at the top for more info. 
5. You can extend it. If you have other methods that you consider better than the ones we already offer, we encourage you to fork the repo, and extend `pelicun` with your approach. You do not need to share your extended version with the community, but if you are interested in doing so, contact us and we are more than happy to merge your version with the official release.

## Requirements

The following packages are required for `pelicun`:

- `numpy` >= 1.15.1

- `scipy` >= 1.1

- `pandas` >= 0.20

We recommend installing the Anaconda Python distribution because these packages and many other useful ones are available there.

## Installation

`pelicun` is available for Python 2.7 and Python 3.5+ at the Python Package Index (PyPI). You can simply install it using `pip` as follows:

```
pip install pelicun
```

## License

`pelicun` is distributed under the BSD 3-Clause license, see LICENSE.

## Contact

Adam Zsarnóczay, NHERI SimCenter, Stanford University, adamzs@stanford.edu
