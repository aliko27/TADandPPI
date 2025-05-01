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
from scipy.stats import norm, zscore
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from scipy.stats import shapiro
import re


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
    
    def __str__(self) -> str:
        return 'string'
    
    def build_request(self, output_format, method, identifiers):
        identifiers = "%0d".join(identifiers)
        data = {'identifiers': identifiers,
                'species': self.SPECIES,
                'required_score': self.REQUIRED_SCORE
            }
        url = "/".join([self.STRING_API_URL, output_format, method])
        return url, data
    
    def get_clustering(self, proteins):
        url, data = self.build_request("tsv", "network", proteins)
        response = Disease.send_request(url, "POST", data)
        num_edges = 0
        interaction_data = []
        
        if response:
            interactions = response.text.split("\n")
            interaction_data = [line.split("\t") for line in interactions if line]
            #num_edges += len(interaction_data)
        else:
            print("Failed to retrieve clustering data.")
            
        
        if len(interaction_data) > 1:
            final_interactions = []
            for i in range(1, len(interaction_data)):
                if interaction_data[i][0] in proteins and interaction_data[i][1] in proteins:
                    if {interaction_data[i][0], interaction_data[i][1]} not in final_interactions:
                        final_interactions.append({interaction_data[i][0], interaction_data[i][1]})
            
            num_edges = len(final_interactions)
        else:
            num_edges = 0
            final_interactions = []
        
        return final_interactions, num_edges

class BioGridAPI:
    ALLOWED_METHODS = ['interactions']

    
    def __init__(self, config_file="config.json") -> None:
        with open(config_file, "r") as f:
            self.config = json.load(f)
        
        self.ACCESSKEY = self.config['ACCESSKEY']
        self.BIOGRID_URL = self.config['BIOGRID_URL']
    
    def __str__(self) -> str:
        return 'biogrid'
    
    def build_request(self, method, identifiers):
        identifiers = "|".join(identifiers)
        data = {'geneList': identifiers,
                'includeInteractors': 'false',
                'interactionTypes': 'physical',
                'searchNames': 'true',
                'searchSynonyms': 'true',
                'interSpeciesExcluded': 'true',
                'selfInteractionsExcluded': 'true',
                'throughputTag': 'low',
                #'includeEvidence': 'true',
                #'evidenceList': 'Affinity Capture-Western|Co-crystal Structure|FRET|PCA|Far Western|Cross-Linking-MS',
                'accessKey': self.ACCESSKEY
                
            }
        url = f"{self.BIOGRID_URL}/{method}"
        return url, data
    
    def get_clustering(self, genes):
        url, data = self.build_request("interactions", genes)
        response = Disease.send_request(url, "GET", data)
        num_edges = 0
        interaction_data = []
        
        if response:
            interactions = response.text.split("\n")
            interaction_data = [line.split("\t") for line in interactions if line]
            #num_edges += len(interaction_data)
        else:
            print("Failed to retrieve clustering data.")
        
        if len(interaction_data) > 0:
            final_interactions = []
            for row in interaction_data:
                gene1, gene2 = row[7].upper(), row[8].upper()
                syn1 = set(row[9].split('|')) | {gene1}
                syn2 = set(row[10].split('|')) | {gene2}

                if syn1 & set(genes) and syn2 & set(genes):
                    final_interactions.append(set([gene1, gene2]))
                num_edges = len(final_interactions)
        else:
            num_edges = 0
            final_interactions = []
        
        return final_interactions, num_edges
    

class Disease:
    def __init__(self, name, efo_id, api = 'string', config_file="config.json"):
        self.name = name
        self.efo_id = efo_id
        self.genes = []
        self.proteins = []
        self.interactions = []
        self.random_edges = []
        self.num_edges = 0
        self.pvalue = None
        self.z_score = 0
        self.normal = False
        self.p_val_empirical = None
        
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.iterations = self.config["RANDOM_EDGES_ITERATIONS"]
        aliases_df = pd.read_csv(self.config['ALIAS_FILE'], sep='\t')
        self.ALL_PROTEINS = set(aliases_df['string_protein_id'].unique().tolist())
        self.MIN_NUM_NODES = self.config['MIN_NUM_NODES']
        all_genes_df = pd.read_csv(self.config["ENSEMBL_GENES"], sep='\t')
        self.all_genes = set(all_genes_df['Gene name'].unique().tolist())
        
        self.gwas_path = self.config['GWAS_FILE']
        self.gwas_data = pd.read_csv(self.gwas_path, low_memory=False, sep="\t")
        if api.lower() == 'biogrid':
            self.api = BioGridAPI()
        else:
            self.api = StringAPI()
            
    
    def __str__(self):
        return self.name

    @staticmethod
    def send_request(url, method="GET", data=None, retries=10):
        response = None
        for n in range(0, retries):
            try:
                if method == "GET":
                    response = requests.get(url, params=data)
                    #print(response.text)
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
        '''filtered_data = self.gwas_data[
            self.gwas_data['MAPPED_TRAIT'].str.strip().str.casefold() == self.name.strip().casefold()
        ]'''
        
        mask = self.gwas_data["MAPPED_TRAIT_URI"].apply(
            lambda x: False if pd.isna(x) else any(uri.split("/")[-1] == self.efo_id for uri in x.split("\\,"))
        )
        filtered_data = self.gwas_data[mask]
        # Extract MAPPED_GENE column from the GWAS catalog
        # remove duplicates, double genes
        if not filtered_data.empty:
            self.genes = (
                filtered_data['MAPPED_GENE']
                .dropna()
                .apply(lambda genes: re.split(r'\s*[-,;]\s*|\s+', genes))
                .explode()
                .drop_duplicates()
                .tolist()
            )
            print(f"Genes mapped to {self.name}: ")
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
        url, data = self.api.build_request("tsv", "network", proteins)
        response = self.send_request(url, "POST", data)
        num_edges = 0
        interaction_data = []
        
        if response:
            interactions = response.text.split("\n")
            interaction_data = [line.split("\t") for line in interactions if line]
            #num_edges += len(interaction_data)
        else:
            print("Failed to retrieve clustering data.")
        
        if len(interaction_data) > 1:
            final_interactions = []
            for i in range(1, len(interaction_data)):
                if interaction_data[i][0] in proteins and interaction_data[i][1] in proteins:
                    if {interaction_data[i][0], interaction_data[i][1]} not in final_interactions:
                        final_interactions.append({interaction_data[i][0], interaction_data[i][1]})
            
            num_edges = len(final_interactions)
        else:
            num_edges = 0
            final_interactions = []
        
        if modify_num_edges:
            self.num_edges = num_edges
            self.interactions = final_interactions
            print(f"Number of edges in the disease network: {self.num_edges}\n")
            return self.num_edges
        else:
            return num_edges
    
    def get_network_image(self):
        url, data = self.api.build_request('image', 'network', self.proteins)
        response = self.send_request(url, "POST", data)
        image = Image.open(BytesIO(response.content))
        image.show()
        
    def get_random_edges_parallel(self, identifiers, all_ids):

        # function to handle a single random edge count calculation
        def get_random_edges_single():
            random_ids = [str(rid) for rid in random.sample(list(all_ids), len(identifiers))]

            result = self.api.get_clustering(random_ids)
            return result
        
        futures_to_ids = {}
        # use ThreadPoolExecutor to parallelize the random edge retrieval
        with ThreadPoolExecutor() as executor:
            # submit all tasks (iterations) to the pool
            #futures = [executor.submit(get_random_edges_single) for _ in range(self.iterations)]
            for i in range(self.iterations):
                future = executor.submit(get_random_edges_single)
                futures_to_ids[future] = i  # Associate the future with its ID (here i is used as an example)

        # Collect results as they finish
        for future in as_completed(futures_to_ids.keys()):

            # collect results as they finish
            #for future in as_completed(futures):
                try:
                    l, num = future.result()
                    self.random_edges.append(num)
                except Exception as e:
                    print(f"Error in iteration: {e}")
                    print(f"Type of error: {type(e)}")
        print(f"Random edges calculation completed\n")
        return self.random_edges
    
    def compare_statistically(self):
        mean = np.mean(self.random_edges)
        std = np.std(self.random_edges)
        stat, p = shapiro(self.random_edges)
        
        if p > 0.05:
            self.normal = True
            if std == 0:
                self.z_score = 0
            else:
                self.z_score = (self.num_edges - mean) / std
            self.p_val_empirical = np.sum(np.array(self.random_edges) >= self.num_edges) / self.iterations
        elif p <= 0.05:
            self.normal = False
            self.p_val_empirical = (np.sum(np.array(self.random_edges) >= self.num_edges) + 1) / (self.iterations + 1)
            self.z_score = (self.num_edges - mean) / std

        
        print(f"z score: {self.z_score}\n")
        
        x = np.linspace(min(self.random_edges), max(self.random_edges), 1000)
        pdf = norm.pdf(x, mean, std)
        plt.hist(self.random_edges, bins=30, density=True, alpha=0.6, color='blue', label="Random Edge Counts")
        plt.plot(x, pdf, 'k-', lw=2, label=f'Normal Dist. (μ={mean:.2f}, σ={std:.2f})')
        plt.axvline(self.num_edges, color='red', linestyle='dashed', linewidth=2, label="Observed Edges")

        plt.title('Distribution of Random Edge Counts')
        plt.xlabel('Number of Edges')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'disease_hist/{self.name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.p_val_empirical

    def run(self):
        '''To run all commands at once.'''
        
        print(self.name, self.efo_id)
        self.get_genes()
        if len(self.genes) == 0:
            print("No genes associated with the disease trait\n")
            return None
        
        if str(self.api) == 'string':
            
            self.get_proteins()
            if len(self.proteins) == 0:
                print(f"No proteins mapped to the disease trait\n")
                return None
            
            self.interactions, self.num_edges = self.api.get_clustering(self.proteins)
            if self.num_edges < self.MIN_NUM_NODES:
                print(f"The number of edges in the network is less than the minimum requirement: {self.MIN_NUM_NODES}\n")
                return None
            
            self.get_ppi_enrichment()
            self.get_random_edges_parallel(self.proteins, self.ALL_PROTEINS)
        
        else:
            print(self.genes)
            self.interactions, self.num_edges = self.api.get_clustering(self.genes)
            if self.num_edges < self.MIN_NUM_NODES:
                print(f"The number of edges in the network is less than the minimum requirement: {self.MIN_NUM_NODES}\n")
                return None
            self.get_random_edges_parallel(self.genes, self.all_genes)

        
        
        self.compare_statistically()
        return self.num_edges, self.pvalue, self.z_score, self.p_val_empirical
        