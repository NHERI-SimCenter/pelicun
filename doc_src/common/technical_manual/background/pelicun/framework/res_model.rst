.. _lbl-tb_framework_response:

**************
Response Model
**************

The response model is based on the samples in the raw EDP file and provides a probabilistic description of the structural response. The samples can include an arbitrary number of EDP types (EDPt in Fig. 4) that describe the structural response at pre-defined locations and directions (EDPt,l,d). In buildings, locations typically correspond to floors or stories, and two directions are assigned to the primary and secondary horizontal axes. However, one might use more than two directions to collect several responses at each floor of an irregular building and locations can refer to other parts of structures, such as the piers of a bridge or segments of a pipeline.

EDPs can be resampled either after fitting a probability distribution function to the raw data or by bootstrapping the raw EDPs. Besides the widely used multivariate lognormal distribution, its truncated version is also available. This allows the consideration, for example, that PID values above a pre-defined truncation limit are not reliable. Another option, using the raw EDPs as-is, is useful in regional simulations to preserve the order of samples and maintain the spatial dependencies introduced in random characteristics of the building inventory or the seismic hazard.