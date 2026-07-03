import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


def compute_importance_weights_kde(
    X_train,
    X_target,
    bandwidth=0.3,
    clip_min=0.1,
    clip_max=10.0,
    max_fit_samples=None,
    random_state=42,
):
    """Estimate ``p_target(x) / p_train(x)`` for each training sample.

    Features are standardized jointly before fitting the two KDEs so that a
    single bandwidth has the same meaning across feature dimensions. Returned
    weights have mean one and satisfy the requested final clipping bounds.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_target = np.asarray(X_target, dtype=float)

    if X_train.ndim != 2 or X_target.ndim != 2:
        raise ValueError("X_train and X_target must be two-dimensional arrays")
    if len(X_train) == 0 or len(X_target) == 0:
        raise ValueError("X_train and X_target must be non-empty")
    if X_train.shape[1] != X_target.shape[1]:
        raise ValueError("X_train and X_target must have the same number of features")
    if not np.isfinite(X_train).all() or not np.isfinite(X_target).all():
        raise ValueError("X_train and X_target must contain only finite values")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if not (0 < clip_min <= 1 <= clip_max):
        raise ValueError("clip bounds must satisfy 0 < clip_min <= 1 <= clip_max")

    rng = np.random.default_rng(random_state)
    train_fit = X_train
    target_fit = X_target
    if max_fit_samples is not None:
        if max_fit_samples <= 0:
            raise ValueError("max_fit_samples must be positive or None")
        if len(train_fit) > max_fit_samples:
            train_fit = train_fit[
                rng.choice(len(train_fit), size=max_fit_samples, replace=False)
            ]
        if len(target_fit) > max_fit_samples:
            target_fit = target_fit[
                rng.choice(len(target_fit), size=max_fit_samples, replace=False)
            ]

    scaler = StandardScaler().fit(np.vstack([train_fit, target_fit]))
    X_train_scaled = scaler.transform(X_train)
    train_fit_scaled = scaler.transform(train_fit)
    target_fit_scaled = scaler.transform(target_fit)

    kde_train = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(
        train_fit_scaled
    )
    kde_target = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(
        target_fit_scaled
    )

    log_ratio = (
        kde_target.score_samples(X_train_scaled)
        - kde_train.score_samples(X_train_scaled)
    )
    # Work in log space until exponentiation to avoid numerical overflow.
    log_ratio = np.clip(log_ratio, np.log(clip_min), np.log(clip_max))
    raw_weights = np.exp(log_ratio)

    # Find a common scale whose clipped weights have mean one. This preserves
    # both contracts exactly; clipping followed by ordinary normalization does
    # not generally preserve the clipping bounds.
    lower_scale, upper_scale = 0.0, 1.0
    while np.mean(np.clip(raw_weights * upper_scale, clip_min, clip_max)) < 1:
        upper_scale *= 2
    for _ in range(60):
        scale = (lower_scale + upper_scale) / 2
        candidate_mean = np.mean(
            np.clip(raw_weights * scale, clip_min, clip_max)
        )
        if candidate_mean < 1:
            lower_scale = scale
        else:
            upper_scale = scale
    weights = np.clip(
        raw_weights * ((lower_scale + upper_scale) / 2),
        clip_min,
        clip_max,
    )
    return weights


def effective_sample_size(weights):
    """Return the usual importance-weight effective sample size."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.dot(weights, weights))
