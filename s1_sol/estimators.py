"""
Sample estimators for mean and standard deviation.
"""
import numpy as np
from s1_sol import fitting

def run_sample_estimates_analysis(grouped_data):
    """
    Complete Exercise 1(iii): Calculate sample estimates (mean, std) and errors.
    
    Parameters
    ----------
    grouped_data : dict
        Grouped data by energy
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'E0_list': List of energy values
        - 'means': Sample means
        - 'mean_errors': Errors on means
        - 'stds': Sample standard deviations
        - 'std_errors': Errors on stds
        - 'per_energy': Dict with detailed stats per energy
        - 'fitted_params': Parameters from linear fits to these estimates
        - 'fitted_errors': Errors on fitted parameters
    """

    E0_list = sorted(grouped_data.keys())
    means, mean_errs = [], []
    stds, std_errs = [], []
    per_energy = {}
    
    for E0 in E0_list:
        data = grouped_data[E0]['E_rec']
        n = len(data)
        
        # Calculate Sample Statistics
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)  # Unbiased estimator (N-1)
        
        # Calculate Standard Errors
        # Error on mean = sigma / sqrt(N)
        mu_err = sigma / np.sqrt(n)
        
        # Error on std = sigma / sqrt(2(N-1))
        sigma_err = sigma / np.sqrt(2 * (n - 1))
        
   
        means.append(mu)
        stds.append(sigma)
        mean_errs.append(mu_err)
        std_errs.append(sigma_err)
        
        per_energy[E0] = {
            'mean': mu,
            'mean_err': mu_err,
            'std': sigma,
            'std_err': sigma_err,
            'n_events': n
        }
    
    return {
        'E0_list': E0_list,
        'means': means,
        'mean_errors': mean_errs,
        'stds': stds,
        'std_errors': std_errs,
        'per_energy': per_energy
    }
