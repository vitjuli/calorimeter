"""
Individual Maximum Likelihood Estimation fits.
"""
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

def gaussian_pdf(x, mu, sigma):
    """
    Gaussian probability density function.
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def negative_log_likelihood(data, mu, sigma):
    """
    Calculate negative log-likelihood for Gaussian.
    """
    nll = UnbinnedNLL(data, gaussian_pdf)
    return nll(mu, sigma)

def fit_gaussian_for_energy(E_rec, E0_true):
    """
    Fit Gaussian to E_rec data for a single E0 value using UnbinnedNLL.
    
    Parameters
    ----------
    E_rec : array-like
        Reconstructed energy measurements
    E0_true : float
        True energy value (used for initial guess)
        
    Returns
    -------
    minuit : Minuit
        Fitted Minuit object
    params : dict
        {'mu': val, 'sigma': val}
    errors : dict
        {'mu': err, 'sigma': err}
    """
    nll = UnbinnedNLL(E_rec, gaussian_pdf)
    init_mu = np.mean(E_rec)
    init_sigma = np.std(E_rec, ddof=1)
    
    m = Minuit(nll, mu=init_mu, sigma=init_sigma)
    m.limits['sigma'] = (0, None)
    
    m.migrad()
    m.hesse()
    
    params = {
        'mu': m.values['mu'],
        'sigma': m.values['sigma']
    }
    
    errors = {
        'mu': m.errors['mu'],
        'sigma': m.errors['sigma']
    }
    
    return m, params, errors

def run_mle_fits(grouped_data, sample_results=None, verbose=True):
    """
    Run MLE fits for all energy groups.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary of grouped data
    sample_results : dict, optional
        Results from run_sample_estimates_analysis for comparison
        
    Returns
    -------
    results_mle : dict
        Dictionary with results for each E0
    """
    import pandas as pd
    
    results_mle = {}
    
    for E0 in sorted(grouped_data.keys()):
        E_rec = grouped_data[E0]['E_rec']
        m, params, errors = fit_gaussian_for_energy(E_rec, E0)
        
        results_mle[E0] = {
            'mu': params['mu'],
            'sigma': params['sigma'],
            'mu_err': errors['mu'],
            'sigma_err': errors['sigma']
        }

    E0_list = sorted(results_mle.keys())
    
    if sample_results is not None:
        data = []
        for i, E0 in enumerate(E0_list):
            data.append({
                'E0': E0,
                'μ_sample': sample_results['means'][i],
                'μ_mle': results_mle[E0]['mu'],
                'Δμ': results_mle[E0]['mu'] - sample_results['means'][i],
                'σ(μ)_sample': sample_results['mean_errors'][i],
                'σ(μ)_mle': results_mle[E0]['mu_err'],
                'σ_sample': sample_results['stds'][i],
                'σ_mle': results_mle[E0]['sigma'],
                'Δσ': results_mle[E0]['sigma'] - sample_results['stds'][i],
                'σ(σ)_sample': sample_results['std_errors'][i],
                'σ(σ)_mle': results_mle[E0]['sigma_err']
            })
    else:
        data = [{
            'E0': E0,
            'μ_mle': results_mle[E0]['mu'],
            'σ(μ)': results_mle[E0]['mu_err'],
            'σ_mle': results_mle[E0]['sigma'],
            'σ(σ)': results_mle[E0]['sigma_err']
        } for E0 in E0_list]
    
    return results_mle