# DB Damage and Loss

### SimCenter Damage and Loss DataBase

This database provides damage and loss model parameters intended for a broad range of applications in Natural Hazards Engineering. In this initial release, it is seeded with collections of models for earthquake damage and loss assessment. Future updates will add models to support studies focused on hurricane wind, storm surge, flood, and tsunami impacts.

The model parameters are stored using the damage and loss model schema introduced in SimCenter's Pelicun framework. Models are grouped into collections; each collection is stored in a pair of CSV and JSON files. The CSV files contain model parameters that are required for the calculations, while the JSON files contain corresponding metadata that describe the components and their damage states. The `DB` folder stores the available collections; and, for the sake of transparency and reproducibility, the `data_sources` folder provides the raw data that was used to generate the data in `DB`.

Researchers and practitioners are encouraged to comment on the available data and provide feedback on what additional data they would like to see in the database. Those who have relevant data available are encouraged to contact us and contribute to the database.


### ChangeLog

#### v1.0

- Initial release, includes the following collections:
	+ FEMA P-58 2nd edition
	+ Hazus Earthquake Model for Buildings
	+ Hazus Earthquake Model for Transportation

### Documentation

Every model available in the database is documented under Damage and Loss DB in: https://nheri-simcenter.github.io/PBE-Documentation/

### Acknowledgement

This material is based upon work supported by the National Science Foundation under Grants No. 1621843 and No. 2131111. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

### Contact

NHERI-SimCenter nheri-simcenter@berkeley.edu