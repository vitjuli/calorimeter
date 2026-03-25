"""
Profile likelihood analysis for parameter estimation.

This module provides tools to compute and visualize profile likelihoods,
which help understand parameter uncertainties beyond simple Hesse errors.
"""
import numpy as np
from iminuit import Minuit


def profile_likelihood_1d(data, param_name, param_range, fix_params=None):
    """
    Compute 1D profile likelihood for a Gaussian fit parameter.
    
    For each value of param_name in param_range, minimize the NLL
    with respect to all other parameters.
    
    Parameters
    ----------
    data : array-like
        Data to fit
    param_name : str
        Name of parameter to profile ('mu' or 'sigma')
    param_range : array-like
        Values of param_name to scan
    fix_params : dict, optional
        Parameters to fix (not used in current implementation)
        
    Returns
    -------
    param_values : array
        Scanned parameter values
    nll_values : array
        Negative log-likelihood at each point
    """
    from s1_sol.mle_fits import negative_log_likelihood
    
    nll_values = []
    
    for param_val in param_range:
        if param_name == 'mu':
            # Fix mu, fit sigma
            m = Minuit(lambda sigma: negative_log_likelihood(data, param_val, sigma),
                      sigma=np.std(data, ddof=1))
        elif param_name == 'sigma':
            # Fix sigma, fit mu
            m = Minuit(lambda mu: negative_log_likelihood(data, mu, param_val),
                      mu=np.mean(data))
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        m.errordef = Minuit.LIKELIHOOD
        m.limits['mu' if param_name == 'sigma' else 'sigma'] = (0, None)
        
        m.migrad()
        
        if m.valid:
            nll_values.append(m.fval)
        else:
            nll_values.append(np.nan)
    
    return np.array(param_range), np.array(nll_values)


def compute_profile_for_energy(grouped_data, E0, n_points=50):
    """
    Compute profile likelihoods for both mu and sigma at a given energy.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary from group_by_energy()
    E0 : float
        Energy value to analyze
    n_points : int
        Number of points to scan for each parameter
        
    Returns
    -------
    results : dict
        Dictionary with:
        - 'mu_scan': (mu_values, nll_mu)
        - 'sigma_scan': (sigma_values, nll_sigma)
        - 'best_fit': {'mu': mu_hat, 'sigma': sigma_hat, 'nll_min': nll_min}
    """
    from s1_sol.mle_fits import fit_gaussian_for_energy
    
    data = grouped_data[E0]['E_rec']
    
    # Get best fit values
    m, params, errors = fit_gaussian_for_energy(data, E0)
    mu_hat = params['mu']
    sigma_hat = params['sigma']
    nll_min = m.fval
    

    mu_error = errors['mu']
    sigma_error = errors['sigma']
    
    mu_range = np.linspace(mu_hat - 3*mu_error, mu_hat + 3*mu_error, n_points)
    sigma_range = np.linspace(max(0.1, sigma_hat - 3*sigma_error), 
                               sigma_hat + 3*sigma_error, n_points)
    
    # Compute profiles
    mu_vals, nll_mu = profile_likelihood_1d(data, 'mu', mu_range)
    sigma_vals, nll_sigma = profile_likelihood_1d(data, 'sigma', sigma_range)
    
    return {
        'mu_scan': (mu_vals, nll_mu),
        'sigma_scan': (sigma_vals, nll_sigma),
        'best_fit': {
            'mu': mu_hat,
            'sigma': sigma_hat,
            'nll_min': nll_min,
            'mu_error': mu_error,
            'sigma_error': sigma_error
        }
    }


def compute_mle_trend_profiles(mle_results, n_points=30):
    """
    Compute profile chi-squared for trend parameters (λ, Δ, a, b, c)
    by fitting trends to MLE results.
    """
    from s1_sol.fitting import fit_mean_parameters, fit_resolution_parameters
    
    E0_list = np.array(sorted(mle_results.keys()))
    means = np.array([mle_results[E0]['mu'] for E0 in E0_list])
    mean_errs = np.array([mle_results[E0]['mu_err'] for E0 in E0_list])
    stds = np.array([mle_results[E0]['sigma'] for E0 in E0_list])
    std_errs = np.array([mle_results[E0]['sigma_err'] for E0 in E0_list])
    
    m_mean, mean_params, mean_errors = fit_mean_parameters(E0_list, means, mean_errs)
    m_res, res_params, res_errors = fit_resolution_parameters(E0_list, stds, std_errs)
    
    params = {**mean_params, **res_params}
    errors = {**mean_errors, **res_errors}
    
    results = {'params': params, 'errors': errors, 'scans': {}}
    
    # Profile lambda and Delta 
    for p_name in ['lambda_param', 'Delta']:
        display_name = 'lambda' if p_name == 'lambda_param' else p_name
        scan_range, chi2_vals, _ = m_mean.mnprofile(p_name, size=n_points)
        results['scans'][display_name] = (scan_range, chi2_vals)
    
    # Profile a, b, c
    for p_name in ['a', 'b', 'c']:
        scan_range, chi2_vals, _ = m_res.mnprofile(p_name, size=n_points)
        results['scans'][p_name] = (scan_range, chi2_vals)
    
    return results


def run_mle_profiles(mle_results, n_points=30):
    """Run profile analysis for MLE trend parameters and plot."""
    from s1_sol import plotting
    
    results = compute_mle_trend_profiles(mle_results, n_points)
    fig, _ = plotting.plot_mle_trend_profiles(results)
    plotting.save_figure(fig, 'Figure2.3_mle_profiles.pdf')
    
    return results


def compute_simultaneous_profiles(E_true, E_rec, n_points=30):
    """
    Compute 1D profile likelihoods for all simultaneous fit parameters
    using mnprofile (same style as compute_mle_trend_profiles).
    """
    from s1_sol.simultaneous_fit import run_simultaneous_fit
    
    # Run global fit
    m, params, errors = run_simultaneous_fit(E_true, E_rec)
    
    results = {
        'params': params,
        'errors': errors,
        'scans': {}
    }
    
    param_names = ['lambda_param', 'Delta', 'a', 'b', 'c']
    display_names = ['lambda', 'Delta', 'a', 'b', 'c']
    
    for p_name, d_name in zip(param_names, display_names):
        scan_range, nll_vals, _ = m.mnprofile(p_name, size=n_points)
        results['scans'][d_name] = (scan_range, nll_vals)
    
    return results


def run_simultaneous_profiles(E_true, E_rec, n_points=30):
    """Run profile analysis for simultaneous fit parameters and plot."""
    from s1_sol import plotting
    
    results = compute_simultaneous_profiles(E_true, E_rec, n_points)
    fig, _ = plotting.plot_mle_trend_profiles(results)  # Same plotting function
    plotting.save_figure(fig, 'Figure3.3a_simultaneous_profiles.pdf')
    
    return results


def compute_simultaneous_contours(E_true, E_rec, param_x, param_y, cl=0.68):
    """
    Compute 2D likelihood contour for two parameters.
    
    Parameters
    ----------
    E_true, E_rec : array-like
        Data arrays
    param_x, param_y : str
        Parameter names (e.g., 'lambda_param', 'Delta')
    cl : float
        Confidence level (0.68 for 1 sigma)
        
    Returns
    -------
    x, y : arrays
        Contour points
    best_fit : tuple
        (x_hat, y_hat)
    """
    from s1_sol.simultaneous_fit import run_simultaneous_fit
    
    m, params, errors = run_simultaneous_fit(E_true, E_rec)
    
    param_mapping = {
        'lambda_param': 'lambda',
        'Delta': 'Delta',
        'a': 'a',
        'b': 'b',
        'c': 'c'
    }
    
    pts = m.mncontour(param_x, param_y, cl=cl, size=100)
    
    x_name = param_mapping.get(param_x, param_x)
    y_name = param_mapping.get(param_y, param_y)
    
    return pts, (params[x_name], params[y_name])


def run_all_profiles(E_true, E_rec, n_points=30):
    """
    Run profile likelihood analysis for all trend parameters (λ, Δ, a, b, c).
    
    Parameters
    ----------
    E_true : array-like
        True energy values (all data)
    E_rec : array-like
        Reconstructed energy values (all data)
    n_points : int
        Number of points for profile scan
        
    Returns
    -------
    results : dict
        Profile results with scans for each parameter
    """
    from s1_sol import plotting
    
    results = compute_simultaneous_profiles(E_true, E_rec, n_points=n_points)
    
    fig, _ = plotting.plot_simultaneous_profiles(results)
    plotting.save_figure(fig, 'Figure2.3_profiles.pdf')
    
    return results

