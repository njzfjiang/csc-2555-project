import numpy as np
import os
import pickle


def get_cache_key(num_samples=1000, 
                  num_features=2,
                  prior_a=0.5,
                  base_rate_a=0.3,
                  base_rate_b=0.3,
                  cov_scale_a=1.0,
                  cov_scale_b=1.0,
                  flip_noise_a=0.0,
                  seed=42):
    """Generate a cache key based on data generation parameters."""
    params = f"n{num_samples}_f{num_features}_pa{prior_a:.2f}_bra{base_rate_a:.2f}_brb{base_rate_b:.2f}_csa{cov_scale_a:.2f}_csb{cov_scale_b:.2f}_fna{flip_noise_a:.2f}_s{seed}"
    return f"data_{params}.pkl"


def load_cached_data(cache_dir='data/cached', **kwargs):
    """
    Load cached data if it exists.
    
    Parameters:
    -----------
    cache_dir : str, optional
        Directory to load cached data from. Default: 'data/cached'
    **kwargs : dict
        Data generation parameters (used to generate cache key)
    
    Returns:
    --------
    tuple or None
        (X, y, s) if cached data exists, None otherwise
    """
    cache_key = get_cache_key(**kwargs)
    cache_path = os.path.join(cache_dir, cache_key)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache from {cache_path}: {e}")
            return None
    return None


def save_cached_data(X, y, s, cache_dir='data/cached', **kwargs):
    """
    Save generated data to cache.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Labels
    s : array-like
        Group memberships
    cache_dir : str, optional
        Directory to save cached data. Default: 'data/cached'
    **kwargs : dict
        Data generation parameters (used to generate cache key)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = get_cache_key(**kwargs)
    cache_path = os.path.join(cache_dir, cache_key)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y, s), f)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")


def generate_data(num_samples=1000, 
                  num_features=2,
                  prior_a=0.5,           # P(A) = 0.5, P(B) = 0.5
                  base_rate_a=0.3,       # P(Y=1|A)
                  base_rate_b=0.3,       # P(Y=1|B)
                  cov_scale_a=1.0,       # Covariance scale for group A (γ)
                  cov_scale_b=1.0,       # Covariance scale for group B
                  flip_noise_a=0.0,      # flip noise for group A (β) [0, 1]
                  seed=42,
                  use_cache=False,
                  cache_dir='data/cached'):
    """
    Generate synthetic data with controlled shifts.
    
    Parameters:
    -----------
    use_cache : bool, optional
        Whether to use cached data if available. Default: False
    cache_dir : str, optional
        Directory for caching. Default: 'data/cached'
    """
    # Try to load from cache if enabled
    if use_cache:
        cached_data = load_cached_data(
            cache_dir=cache_dir,
            num_samples=num_samples,
            num_features=num_features,
            prior_a=prior_a,
            base_rate_a=base_rate_a,
            base_rate_b=base_rate_b,
            cov_scale_a=cov_scale_a,
            cov_scale_b=cov_scale_b,
            flip_noise_a=flip_noise_a,
            seed=seed
        )
        if cached_data is not None:
            return cached_data
    
    np.random.seed(seed)
    
    # 1. generate 2 groups using s (0=A, 1=B) and label y
    s = np.random.choice([0, 1], size=num_samples, p=[prior_a, 1-prior_a])
    base_rates = np.where(s == 0, base_rate_a, base_rate_b)
    y = np.random.binomial(1, base_rates)
    
    # 2. generate X (covariate shift: scaled by group s) 
    # with the first feature correlated with y and s, and the rest as noise
    X = np.zeros((num_samples, num_features))

    mean = np.where(y == 1, 1.0, -1.0)
    scale = np.where(s == 0, cov_scale_a, cov_scale_b)
    X[:, 0] = np.random.normal(loc=mean, scale=scale)

    for j in range(1, num_features):
        X[:, j] = np.random.normal(0, 1)
    
    # 3. introduce label noise (flip) based on group s and label y
    # Group A (s=0) and (y=1)
    mask_a = (s == 0) & (y == 1)
    flip_a = np.random.rand(np.sum(mask_a)) < flip_noise_a
    y[mask_a] = np.where(flip_a, 0, y[mask_a])
    
    # Group B (s=1) and (y=0)
    mask_b = (s == 1) & (y == 0)
    flip_b = np.random.rand(np.sum(mask_b)) < (flip_noise_a / 2)
    y[mask_b] = np.where(flip_b, 1, y[mask_b])
    
    # Save to cache if enabled
    if use_cache:
        save_cached_data(
            X, y, s,
            cache_dir=cache_dir,
            num_samples=num_samples,
            num_features=num_features,
            prior_a=prior_a,
            base_rate_a=base_rate_a,
            base_rate_b=base_rate_b,
            cov_scale_a=cov_scale_a,
            cov_scale_b=cov_scale_b,
            flip_noise_a=flip_noise_a,
            seed=seed
        )
    
    return X, y, s

