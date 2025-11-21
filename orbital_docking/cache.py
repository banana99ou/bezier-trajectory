"""
Caching utilities for optimization results.
"""

import hashlib
import pickle
import numpy as np
from pathlib import Path


CACHE_VERSION = "1.0"  # Increment to invalidate old caches


def get_cache_key(P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1):
    """
    Generate a deterministic cache key from optimization parameters.
    
    Args:
        P_init: Initial control points
        n_seg: Number of segments
        r_e: KOZ radius
        max_iter: Maximum iterations
        tol: Convergence tolerance
        sample_count: Number of samples for cost evaluation
        v0, v1: Velocity boundary conditions
        a0, a1: Acceleration boundary conditions
    
    Returns:
        str: Cache key (hex digest)
    """
    # Create a deterministic hash from all parameters
    hash_input = {
        'P_init': P_init.tobytes() if isinstance(P_init, np.ndarray) else str(P_init),
        'n_seg': n_seg,
        'r_e': float(r_e),
        'max_iter': max_iter,
        'tol': float(tol),
        'sample_count': sample_count,
        'v0': v0.tobytes() if v0 is not None and isinstance(v0, np.ndarray) else str(v0),
        'v1': v1.tobytes() if v1 is not None and isinstance(v1, np.ndarray) else str(v1),
        'a0': a0.tobytes() if a0 is not None and isinstance(a0, np.ndarray) else str(a0),
        'a1': a1.tobytes() if a1 is not None and isinstance(a1, np.ndarray) else str(a1),
        'version': CACHE_VERSION
    }
    
    # Convert to string and hash
    hash_str = str(sorted(hash_input.items()))
    return hashlib.md5(hash_str.encode()).hexdigest()


def get_cache_path(cache_key, n_seg, cache_dir=None):
    """
    Get the cache file path for a given cache key and segment count.
    
    Args:
        cache_key: Cache key (hex digest)
        n_seg: Number of segments
        cache_dir: Cache directory (defaults to cache/ in project root)
    
    Returns:
        Path: Path to cache file
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"opt_{cache_key[:8]}_nseg{n_seg}.pkl"


def load_from_cache(cache_path):
    """
    Load optimization result from cache.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        tuple: (P_opt, info) if cache exists and is valid, None otherwise
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate cache structure
        if isinstance(data, dict) and 'P_opt' in data and 'info' in data:
            return data['P_opt'], data['info']
        else:
            # Old format compatibility
            return data
    except Exception as e:
        # If cache is corrupted, delete it
        try:
            cache_path.unlink()
        except:
            pass
        return None


def save_to_cache(cache_path, P_opt, info):
    """
    Save optimization result to cache.
    
    Args:
        cache_path: Path to cache file
        P_opt: Optimized control points
        info: Info dictionary
    """
    try:
        data = {
            'P_opt': P_opt,
            'info': info,
            'version': CACHE_VERSION
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        # Silently fail if cache write fails
        pass


def clear_cache(cache_key=None, cache_dir=None):
    """
    Clear cache files. If cache_key is provided, only clear that specific cache.
    
    Args:
        cache_key: Optional cache key to clear specific cache, or None to clear all
        cache_dir: Cache directory (defaults to cache/ in project root)
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
    
    if not cache_dir.exists():
        return
    
    if cache_key is None:
        # Clear all cache files
        for cache_file in cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass
    else:
        # Clear specific cache files
        for cache_file in cache_dir.glob(f"opt_{cache_key[:8]}*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass

