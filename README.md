<p align="center">
	<img src="https://raw.githubusercontent.com/NHERI-SimCenter/pelicun/master/doc/source/_static/pelicun-Logo-white.png"
		alt="pelicun" align="middle" height="200"/>
</p>

<p align="center">
	<a href="https://doi.org/10.5281/zenodo.2558558", alt="DOI">
		<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.2558558.svg" /></a>
</p>

<p align="center">
	<b>Probabilistic Estimation of Losses, Injuries, and Community resilience Under Natural hazard events</b>
</p>

[![Latest Release](https://img.shields.io/github/v/release/NHERI-SimCenter/pelicun?color=blue&label=Latest%20Release)](https://github.com/NHERI-SimCenter/pelicun/releases/latest)
![Tests](https://github.com/NHERI-SimCenter/pelicun/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/github/NHERI-SimCenter/pelicun/branch/master/graph/badge.svg?token=W79M5FGOCG)](https://codecov.io/github/NHERI-SimCenter/pelicun/tree/master)
[![Ruff](https://img.shields.io/badge/ruff-linted-blue)](https://img.shields.io/badge/ruff-linted-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue)](https://raw.githubusercontent.com/NHERI-SimCenter/pelicun/master/LICENSE)

## What is it?

`pelicun` is a Python package that provides tools for assessment of damage and losses due to natural hazard events. It uses a stochastic damage and loss model that is an extension of the high-resolution PEER performance assessment methodology described in FEMA P58 (FEMA, 2012). While FEMA P58 aims to assess the seismic performance of a building, with `pelicun` we provide a more versatile, hazard-agnostic tool to assess the performance of several types of assets in the built environment.

Detailed documentation of the available methods and their use is available at http://nheri-simcenter.github.io/pelicun

## What can I use it for?

`pelicun` quantifies losses from an earthquake or hurricane scenario in the form of decision variables. This functionality is typically utilized for performance-based engineering and regional risk assessment. There are several steps of performance assessment that `pelicun` can help with:

- **Describe the joint distribution of asset response.** The response of a structure or other type of asset to an earthquake or hurricane wind is typically described by so-called engineering demand parameters (EDPs). `pelicun` provides methods that take a finite number of EDP vectors and find a multivariate distribution that describes the joint distribution of EDP data well. You can control the type of target distribution, apply truncation limits and censor part of the data to consider detection limits in your analysis. Alternatively, you can choose to use your EDP vectors as-is without resampling from a fitted distribution.

- **Define the damage and loss model of a building.** The component damage and loss data from the first two editions of FEMA P58 and the HAZUS earthquake and hurricane models for buildings are provided with pelicun. This makes it easy to define building components without having to collect and provide all the data manually. The stochastic damage and loss model is designed to facilitate modeling correlations between several parameters of the damage and loss model.

- **Estimate component damages.** Given a damage and loss model and the joint distribution of EDPs, `pelicun` provides methods to estimate the amount of damaged components and the number of cases with collapse.

- **Estimate consequences.** Using information about collapse and component damages, the following consequences can be estimated with the loss model: reconstruction cost and time, unsafe placarding (red tag), injuries and fatalities.

## Why should I use it?

1. It is free and it always will be.
2. It is open source. You can always see what is happening under the hood.
3. It is efficient. The loss assessment calculations in `pelicun` use `numpy`, `scipy`, and `pandas` libraries to efficiently propagate uncertainties and provide detailed results quickly.
4. You can trust it. Every function in `pelicun` is tested after every commit. See the Travis-CI and Coveralls badges at the top for more info.
5. You can extend it. If you have other methods that you consider better than the ones we already offer, we encourage you to fork the repo, and extend `pelicun` with your approach. You do not need to share your extended version with the community, but if you are interested in doing so, contact us and we are more than happy to merge your version with the official release.

## Installation

`pelicun` is available at the [Python Package Index (PyPI)](https://pypi.org/project/pelicun/). You can simply install it using `pip` as follows:

```shell
pip install pelicun
```

If you are interested in using an earlier version, you can install it with the following command: 

```shell
pip install pelicun==2.6.0
```

Note that 2.6.0 is the last minor version before the v3.0 release. Other earlier versions can be found [here](https://pypi.org/project/pelicun/#history).

## Documentation and usage examples

The documentation for pelicun can be accessed [here](https://NHERI-SimCenter.github.io/pelicun/).
It includes information for users, instructions for developers and usage examples.

## Changelog

The release notes are available in the [online documentation](https://nheri-simcenter.github.io/pelicun/release_notes/index.html)

## License

`pelicun` is distributed under the BSD 3-Clause license, see LICENSE.

## Acknowledgment

This material is based upon work supported by the National Science Foundation under Grants No. 1612843 2131111. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## Contact

Adam Zsarnóczay, NHERI SimCenter, Stanford University, adamzs@stanford.edu
