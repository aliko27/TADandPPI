from dataclasses import dataclass, field
import requests
from requests.exceptions import HTTPError
from http import HTTPStatus
import time
import pandas as pd
import random
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


RETRY_CODES = [
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
]

class StringAPI:
    ALLOWED_METHODS = ['network', 'get_string_ids', 'ppi_enrichment']
    ALLOWED_OUPUTS = ['image', 'json', 'tsv-no-header']
    
    def __init__(self, config_file="config.json") -> None:
        with open(config_file, "r") as f:
            self.config = json.load(f)
            
        self.STRING_API_URL = self.config['STRING_API_URL']
        self.SPECIES = self.config['SPECIES']
        self.REQUIRED_SCORE = self.config['REQUIRED_SCORE']
    
    def build_request(self, output_format, method, identifiers):
        identifiers = "%0d".join(identifiers)
        data = {'identifiers': identifiers,
                'species': self.SPECIES,
                'required_score': self.REQUIRED_SCORE
            }
        url = "/".join([self.STRING_API_URL, output_format, method])
        return url, data
        

class Disease:
    def __init__(self, name, efo_id, config_file="config.json"):
        self.name = name
        self.efo_id = efo_id
        self.genes = []
        self.proteins = []
        self.interactions = []
        self.num_edges = 0
        self.pvalue = 0
        self.z_score = 0
        
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.num_random_edges = self.config["NUM_RANDOM_EDGES"]
        aliases_df = pd.read_csv(self.config['ALIAS_FILE'], sep='\t')
        self.ALL_PROTEINS = set(aliases_df['string_protein_id'].unique().tolist())
        self.MIN_NUM_NODES = self.config['MIN_NUM_NODES']
        
        self.gwas_path = self.config['GWAS_FILE']
        self.gwas_data = pd.read_csv(self.gwas_path, low_memory=False, sep="\t")
        self.api = StringAPI()
    
    def __str__(self):
        return self.name

    def send_request(self, url, method="GET", data=None, retries=3):
        response = None
        for n in range(0, retries):
            try:
                if method == "GET":
                    response = requests.get(url)
                if method == "POST":
                    if data is not None:
                        response = requests.post(url, data=data)
                    else:
                        print("Parameter Data is empty\n")
                        return None
                response.raise_for_status()
                return response
            except HTTPError as exc:
                code = exc.response.status_code
                if code in RETRY_CODES:
                    time.sleep(2**n)
                    continue
                print(f"Failed to send get request: {exc}\n")
                return None
            except requests.RequestException as exc:
                if n == (retries - 1):
                    print(f"Failed to send get request: {exc}\n")
                    return None
                time.sleep(n)
        print("Max retries exceeded\n")
        return None


    def get_genes(self):
        filtered_data = self.gwas_data[
            self.gwas_data['MAPPED_TRAIT'].str.strip().str.casefold() == self.name.strip().casefold()
        ]
        # Extract MAPPED_GENE column from the GWAS catalog
        # remove duplicates, double genes
        if not filtered_data.empty:
            self.genes = (
                filtered_data['MAPPED_GENE']
                .dropna()
                .apply(lambda genes: [g.strip() for g in genes.split(',')])
                .explode()
                .drop_duplicates()
                .tolist()
            )
            print(f"Genes mapped to {self.name}: {self.genes}")
        else:
            print(f"No genes found for {self.name}.")
            self.genes = []
        return self.genes

    
    def get_ppi_enrichment(self, output_format='tsv-no-header', method='ppi_enrichment'):
        url, data = self.api.build_request(output_format, method, self.proteins)
        response = self.send_request(url, 'POST', data)
        
        if response:
            try:
                for line in response.text.strip().split("\n"):
                    columns = line.split("\t")
                    if len(columns) > 5:
                        self.pvalue = columns[5]
                        print("P-value:", self.pvalue)
                    else:
                        print("Unexpected response format:", line)
            except Exception as e:
                print(f"Error parsing response: {e}")
        else:
            print("Failed to retrieve PPI enrichment data.")
        
        if self.pvalue == 0:
            raise ValueError("P-value could not be extracted from response.")
        
        return self.pvalue

    
    def get_proteins(self):
        url, data = self.api.build_request("tsv-no-header", "get_string_ids", self.genes)
        response = self.send_request(url, "POST", data)
        
        if response:
            protein_mappings = response.text.split("\n")
            for line in protein_mappings:
                if line.strip(): 
                    fields = line.split("\t")
                    if len(fields) > 1:
                        protein_id = fields[1]  
                        self.proteins.append(protein_id)
            self.proteins = list(dict.fromkeys(self.proteins))
            print(f"Proteins: {len(self.proteins)}\n")
        else:
            print("Failed to retrieve protein mappings.")
        return self.proteins
    
    def get_clustering(self, proteins, modify_num_edges=True):
        url, data = self.api.build_request("tsv-no-header", "network", proteins)
        response = self.send_request(url, "POST", data)
        num_edges = 0
        if response:
            interactions = response.text.split("\n")
            for interaction in interactions:
                if interaction.strip():
                    self.interactions.append(interaction)
            interaction_data = [line.split("\t") for line in interactions if line]
            num_edges += len(interaction_data)
        else:
            print("Failed to retrieve clustering data.")
        
        if modify_num_edges:
            self.num_edges = num_edges
            print(f"Number of edges in the disease network: {self.num_edges}\n")
            return self.num_edges
        else:
            return num_edges
    
    def get_network_image(self):
        url, data = self.api.build_request('image', 'network', self.proteins)
        response = self.send_request(url, "POST", data)
        image = Image.open(BytesIO(response.content))
        image.show()
        
    def get_random_edges_parallel(self, iterations):
        random_edge_counts = []

        # function to handle a single random edge count calculation
        def get_random_edges_single():
            random_proteins = random.sample(list(self.ALL_PROTEINS), len(self.proteins))
            return self.get_clustering(random_proteins, False)
        
        # use ThreadPoolExecutor to parallelize the random edge retrieval
        with ThreadPoolExecutor() as executor:
            # submit all tasks (iterations) to the pool
            futures = [executor.submit(get_random_edges_single) for _ in range(iterations)]

            # collect results as they finish
            for future in as_completed(futures):
                try:
                    random_edge_counts.append(future.result())
                except Exception as e:
                    print(f"Error in iteration: {e}")

        print(f"Random edges calculation completed\n")
        return random_edge_counts
    
    def compare_statistically(self, random_edges):
        self.z_score = stats.zscore(random_edges, self.num_edges)
        print(f"z score for a random nodes: {self.z_score}\n")

    def run(self):
        '''To run all the commands at once.'''
        
        print(self.name, self.efo_id)
        self.get_genes()
        if len(self.genes) == 0:
            print("No genes associated with the disease trait\n")
            return None
        
        self.get_proteins()
        if len(self.proteins) == 0:
            print(f"No proteins mapped to the disease trait\n")
            return None
        
        self.get_clustering(self.proteins)
        if self.num_edges < self.MIN_NUM_NODES:
            print(f"The number of edges in the network is less than the minimum requirement{self.MIN_NUM_NODES}\n")
            return None

        self.get_ppi_enrichment()
        
        
        
        
        
        return self.num_edges
        