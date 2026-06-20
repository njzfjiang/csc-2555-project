import numpy as np

def generate_data(num_samples=1000, 
                  num_features=2,
                  prior_a=0.5,           # P(A) = 0.5, P(B) = 0.5
                  base_rate_a=0.3,       # P(Y=1|A)
                  base_rate_b=0.3,       # P(Y=1|B)
                  cov_scale_a=1.0,       # Covariance scale for group A (γ)
                  cov_scale_b=1.0,       # Covariance scale for group B
                  flip_noise_a=0.0,      # flip noise for group A (β) [0, 1]
                  seed=42):
    
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
    
    return X, y, s

