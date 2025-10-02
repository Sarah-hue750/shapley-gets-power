# Shapley Gets Power
This project is licensed under the Mozilla Public License 2.0.

This code coresponds to the paper #TODO: add reference.
Shapley gets Power uses shapley values to attribute power flow change in case of multiple outage scenarios.
As an application example the Scandinavian transmission grid provided by PyPSA is studied. 
An approximation scheme of the shapley value is introduced. 
It is based on only considering the k outage lines that have the highest mutual LODF with a given outage line. 
This approximation is compared with ground thruth shapley values.




## Setup

1. **Clone the repository:**

   Clone the repository.

2. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate shapley-gets-power
   ```



3. **Download precomputed networks:**

   Generate them yourself

   1. Navigate into the main Snakemake workflow directory of `PyPSA-Eur`:

   ```bash
    cd submodules/pypsa-eur
   ```
   2. Make sure to copy the custom powerplants to the right place in pypsa-eur


   ```bash
   cp ../../data/custom_powerplants.csv data/
   cp ../../data/config.yaml configs/
   ```

   3. Run the following command
   ```bash
   snakemake -call -j1 solve_elec_networks --configfile ../configs/config.yaml
   ```

4. **Create the file utils/config_local.py:**
   This file should contain:
   ```
   root_path = absolute path to the repository
   ```

## Code run through

To create the plots given in the paper. 
Run through the files contained in the scripts folder in the following order. 

1. **Extract the isolated synchronous grids:**

   This file creates the necessary grid files of the Scandinavian grid to run all other files.
   ```
   /scripts/script_01_extract_testgrid.py
   ```
2. **Find braess paradox cases:**

   Extracts cases of braess paradox in the Scandinavian grid and saves them in ``/results/braess_application_data/``.

   ```
   /scripts/script_02_find_braess_cases.py
   ```
3. **Find application cases:**

   Extracts cases where 4 outage lines lead to an overloaded line. 
   And the overload is most reduced by line l, but line l does not have the greatest direct effect.
   Line l reduces overload because of interaction with the other outages.

   ```
   /scripts/script_03_find_application.py
   ```
4. **Create plots:**

   Creates introductionary and motivational plots of the paper. Including:

   - Plot of a braess example
   - Plot showcase of the shapley taylor indice
   - Plot showing the application example for shapley taylor indices
   
   ```
   /scripts/script_04_create_plots.py
   ```
5. **Run approximations**

   This is a file that calculates approximations of shapley value and true shapley values to allow for comparison.
   This code takes significant computational resources.

   ```
   /scripts/script_05_run_approximations.py
   ```
6. **Create approximation plots**

   Creates the plots comparing approximations of shapley values and ground thruth. Including:

   - Plot comparison for one outage line set and k compared with random selection
   - Plot for multiple outage line sets showing runtime, share above threshold and approximation errors dependent on k

   ```
   /scripts/script_06_create_approximation_plots.py
   ```


## Authors and acknowledgment
Sarah Schreyer - software developement 

Jan Lange - software developement

Philipp Böttcher - software developement

