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


def get_z_score(random_edges, observed_edges):
    z_score = stats.zscore(random_edges)
    plot(random_edges=random_edges, observed_edges=observed_edges)
    return z_score

def plot(random_edges, observed_edges):
    z_score = stats.zscore(random_edges)
    x = np.linspace(min(random_edges), max(random_edges), 1000)
    mean = stats.mean(random_edges)
    std = stats.std(random_edges)
    
    pdf = norm.pdf(x, mean, std)
    plt.hist(random_edges, bins=20, density=True, alpha=0.6, color='blue', label="Random Edge Counts")
    plt.plot(x, pdf, 'k-', lw=2, label=f'Normal Dist. (μ={mean:.2f}, σ={std:.2f})')
    plt.axvline(observed_edges, color='red', linestyle='dashed', linewidth=2, label="Observed Edges")

    plt.title('Distribution of Random Edge Counts')
    plt.xlabel('Number of Edges')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


class StringAPI:
    def __init__(self, output_format, config_file="config.json") -> None:
        self.output_format = output_format
        self.STRING_API_URL = self.config['STRING_API_URL']
        self.SPECIES = self.config['SPECIES']
        self.REQUIRED_SCORE = self.config['REQUIRED_SCORE']
    
    def build_request(self, identifiers):
        identifiers = "%0d".join(identifiers)
        data = {'identifiers': identifiers,
                'species': self.SPECIES,
                'required_score': self.REQUIRED_SCORE
            }
        url = "/".join([self.STRING_API_URL, self.OUTPUT_FORMAT, method])
        return url, data
        
    


class Disease:
    def __init__(self, name, efo_id, config_file="config.json"):
        self.name = name
        self.efo_id = efo_id
        self.genes = []
        self.proteins = []
        self.interactions = []
        self.num_edges = 0
        
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.num_random_edges = self.config["NUM_RANDOM_EDGES"]
        aliases_df = pd.read_csv(self.config['ALIAS_FILE'], sep='\t')
        self.ALL_PROTEINS = set(aliases_df['string_protein_id'].unique().tolist())
        self.MIN_NUM_NODES = self.config['MIN_NUM_NODES']
        
        self.gwas_path = self.config['GWAS_FILE']
        self.gwas_data = pd.read_csv(self.gwas_path, low_memory=False, sep="\t")
    
    def __str__(self):
        return self.name

    def send_request(self, url, method="GET", data=None, retries=3):
        for n in range(0, retries):
            try:
                if method.upper == "GET":
                    response = requests.get(url)
                if method.upper == "POST":
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

    def get_proteins(self, genes, method = 'get_string_ids'):
        identifiers = "%0d".join(genes)
        data = {'identifiers': identifiers,
                'species': self.SPECIES,
                'required_score': self.REQUIRED_SCORE
            }
        url = "/".join([self.STRING_API_URL, self.OUTPUT_FORMAT, method])
        return url, data
    
    def get_clustering(self, proteins, method = 'network'):
        identifiers = "%0d".join(proteins)
        data = {'identifiers': identifiers,
                'species': self.SPECIES,
                'required_score': self.REQUIRED_SCORE
            }
        request_url = "/".join([self.STRING_API_URL, self.OUTPUT_FORMAT, method])
        return request_url, data
    
    def get_clustering_image(self, proteins, method = 'network'):
        identifiers = "%0d".join(proteins)
        data = {'identifiers': identifiers,
                'species': self.SPECIES,
                'required_score': self.REQUIRED_SCORE
            }
        request_url = "/".join([self.STRING_API_URL, 'image', method])
        return request_url, data
    
    def get_ppi_enrichment(self, method='ppi_enrichment'):
        identifiers = "%0d".join(self.proteins)
        data = {
            'identifiers': identifiers,
            'species': self.SPECIES,
            'required_score': self.REQUIRED_SCORE
        }
        url = "/".join([self.STRING_API_URL, self.OUTPUT_FORMAT, method])
        
        response = self.send_post_request(url, data)
        
        pvalue = None  # Initialize to avoid unbound variable error
        
        if response:
            try:
                for line in response.text.strip().split("\n"):
                    columns = line.split("\t")
                    if len(columns) > 5:
                        pvalue = columns[5]
                        print("P-value:", pvalue)
                    else:
                        print("Unexpected response format:", line)
            except Exception as e:
                print(f"Error parsing response: {e}")
        else:
            print("Failed to retrieve PPI enrichment data.")
        
        if pvalue is None:
            raise ValueError("P-value could not be extracted from response.")
        
        return pvalue


    def get_random_edges_parallel(self, num_nodes, iterations=100, required_score=700):
        random_edge_counts = []  # Collect results from all iterations

        # Function to handle a single random edge count calculation
        def get_random_edges_single(iteration):
            random_num_edges = 0
            random_interactions = []
            random_proteins = random.sample(list(self.ALL_PROTEINS), num_nodes)
            url, data = self.get_clustering(random_proteins)
            response = self.send_post_request(url, data)
            
            if response:
                interactions = response.text.split("\n")
                for interaction in interactions:
                    if interaction.strip():
                        random_interactions.append(interaction)
                interaction_data = [line.split("\t") for line in interactions if line]
                random_num_edges += len(interaction_data)
            else:
                print("Failed to retrieve clustering data for random nodes.")

            return random_num_edges

        # use ThreadPoolExecutor to parallelize the random edge retrieval
        with ThreadPoolExecutor() as executor:
            # submit all tasks (iterations) to the pool
            future_to_iteration = {executor.submit(get_random_edges_single, i): i for i in range(iterations)}

            # collect results as they finish
            for future in as_completed(future_to_iteration):
                try:
                    result = future.result()
                    random_edge_counts.append(result)
                except Exception as e:
                    print(f"Error in iteration: {e}")

        print(f"Random edges calculation completed.\n")
        return random_edge_counts
    

    def quantify_clustering(self):
        print(self.name, self.efo_id)
        # Retrieve genes
        genes = self.get_genes()
        print(f"Genes: {len(genes)}\n")
        if len(self.genes) == 0:
            return None
        
        # Retrieve protein mappings
        url, data = self.get_proteins(self.genes)
        response = self.send_post_request(url, data)
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

        if len(self.proteins) == 0 or len(self.proteins) == 2:
            return None
        
        # For clustering the proteins
        url, data = self.get_clustering(self.proteins)
        response = self.send_post_request(url, data)
        if response:
            interactions = response.text.split("\n")
            for interaction in interactions:
                if interaction.strip():
                    self.interactions.append(interaction)
            interaction_data = [line.split("\t") for line in interactions if line]
            self.num_edges += len(interaction_data)
        else:
            print("Failed to retrieve clustering data.")
        print(f"Number of edges: {self.num_edges}\n")
        
        if self.num_edges == 1:
            return None
        
        # For retrieveing the network image
        url, data = self.get_clustering_image(self.proteins)
        response = self.send_post_request(url, data)
        image = Image.open(BytesIO(response.content))
        image.show()
        
        # Compare to random set of protein clustering
        #random_edge_counts = self.get_random_edges_parallel(len(self.proteins), self.num_random_edges, self.REQUIRED_SCORE)
        #z_score = get_z_score(random_edge_counts, self.num_edges)
        #print(f"z score for a random nodes: {z_score}\n")
        
        
        
        return self.num_edges
        