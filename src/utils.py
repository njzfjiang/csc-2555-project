import json
import numpy as np
import yaml

def load_config(config_path='configs/experiment_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_sweep_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    results = data['results']
    alphas = np.array(data['alphas'])
    gammas = np.array(data['gammas'])
    betas = np.array(data['betas'])
    return results, alphas, gammas, betas