.. _lbl-tb_framework_loss:

**********
Loss Model
**********

Each Damage State has a corresponding set of consequence descriptions in the Damage & Loss Database. These are used to define a consequence model that identifies a set of decision variables (DVs) and corresponding consequence functions that link the amount of damaged components to the value of the DV. The constant and quantity-dependent stepwise consequence functions from FEMA P58 are available in pelicun.

Collapses and their consequences are also handled by the damage and the loss models. The collapse model describes collapse events using the concept of collapse modes introduced in FEMA P58. Collapse is either triggered by EDP values exceeding a collapse limit or it can be randomly triggered based on a collapse probability prescribed in the AIM file. The latter approach allows for external collapse fragility models. Each collapse mode has a corresponding collapse consequence model that describes the corresponding injuries and losses.

Similarly to the performance model, the randomness in damage and losses is handled with a few high-dimensional random variables. This allows researchers to experiment with various correlation structures between damages of components, and consequences of those damages. Among the consequences, the repair costs and times and the number of injuries of various severities are also linked; allowing, for example, to consider that repairs that cost more than expected will also take longer time to finish.

Once the damage and loss models are assembled, the previously sampled EDPs are used to evaluate the Damage Measures (Fig. 3). These DMs identify the Damage State of each Component Group in the structure. This information is used by the loss simulation to generate the Decision Variables. The final step of the calculation in pelicun is to aggregate results into a Damage and Loss (DL) file that provides a concise overview of the damage and losses. All intermediate data generated during the calculation (i.e., EDPs, DMs, DVs) are also saved in CSV files.
