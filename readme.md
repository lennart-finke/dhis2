# DHIS2 Coverage Analysis
We aim to predict ground-truth coverage rates globally and in Ethiopia from high-quality datasets like DHS, calibrating based on Ethiopia facility-level data from DHIS2.

## Data Sources
- Exports from DHIS2 Ethiopia (District Health Information Software, Version 2) & MFR (Master Facility Registry). This is maintained by the Ministry of Health and not publicly available.
- United Nations population data, available [here](https://population.un.org/wpp/).
- WorldPopo population data, available [here](https://hub.worldpop.org/geodata/listing?id=135).
- DHS (Demographic and Health Surveys) Ethiopia, available [here](https://www.dhsprogram.com/methodology/survey/survey-display-586.cfm).
- PMA (Performance Monitoring for Action) Ethiopia, available [here](https://www.pmadata.org/data/available-datasets).
- Administrative boundary data from CSA (Central Statistics Agency) and BoFED (Regional Bureau of Finance and Economic Development), available [here](https://data.humdata.org/dataset/cod-ab-eth).
- (Optional:) Travel time data from the Data for Children Collaborative, available [here](https://datashare.ed.ac.uk/handle/10283/3898).

## Replication
For the order and dependencies of the main analysis, see `dependency_graph/`. We recommend `uv` to run the code.