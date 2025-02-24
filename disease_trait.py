# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:03:18 2025

@author: ecakir
"""

# Import necessary libraries
import obonet  # For reading OBO files into network graphs
import networkx as nx  # For working with network/graph data structures
import pandas as pd  # For handling and analyzing tabular data

# Specify the path where the data files are located
dataPath = "data/"

# Define the path to the ontology file (efo.obo), downloaded from https://www.ebi.ac.uk/efo/efo.obo
# The file should be saved in the specified directory under the name "www.ebi.ac.uk.txt".
url = dataPath + "www.ebi.ac.uk.txt"

# Read the ontology file using obonet, which returns a NetworkX graph object
efo_graph = obonet.read_obo(open(url, "r", encoding="utf8"))

# Extract all ancestor terms (parent terms) of the specific node "EFO:0000408" (represents "disease")
# Using `nx.ancestors`, we get all parent nodes recursively in the graph.
allDiseases = list(set(nx.ancestors(efo_graph, "EFO:0000408")))  # List of all parent terms related to "disease"
print(len(allDiseases))
# Load the GWAS Catalog data file
# Replace the file name with the version you are using. Ensure it matches your dataset version.
gwasCatalogEns = pd.read_csv(
    dataPath + "gwas_catalog_v1.0.2-associations_e113_r2024-12-19.tsv",
    low_memory=False,  # Avoid warnings for mixed types in columns
    sep="\t"  # Specify tab as the column separator
)

# Extract EFO IDs from the "MAPPED_TRAIT_URI" column in the GWAS Catalog
# Each URI ends with the EFO ID, which is extracted after the last "/" and formatted correctly.
gwasCatalogEns["efoID"] = (
    gwasCatalogEns["MAPPED_TRAIT_URI"]
    .str.split("/")  # Split the URI string by "/"
    .str[-1]  # Take the last segment (EFO ID)
    .str.replace("_", ":")  # Replace underscores with colons for correct formatting
)
gwasCatalogEns["name"] = (
    gwasCatalogEns["MAPPED_TRAIT"]
)
print(len(gwasCatalogEns))

# Identify traits present in the GWAS catalog that are NOT classified as diseases
# This is done by finding the difference between the set of EFO IDs and the set of disease-related terms.
notDiseasesInGWAS = list(set(gwasCatalogEns.efoID).difference(set(allDiseases)))

# Identify diseases present in the GWAS catalog
efoID_intersection = set(gwasCatalogEns.efoID).intersection(set(allDiseases))
filtered_df = gwasCatalogEns[gwasCatalogEns['efoID'].isin(efoID_intersection)]
allDiseasesInGWAS = dict(zip(filtered_df['efoID'], filtered_df['name']))
try:
    with open("allDiseases.txt", "w") as file:
        for efo_id, name in allDiseasesInGWAS.items():
            file.write(f"{efo_id}\t{name}\n")
    print("Data successfully written to allDiseases.txt")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
    
print(len(allDiseasesInGWAS))
    
#You now have two lists of disease codes and trait codes that appear in the GWAS catalog: allDiseasesInGWAS and allTraitsInGWAS.