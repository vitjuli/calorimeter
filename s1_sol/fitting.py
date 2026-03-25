"""Fitting functions for detector model parameters using iminuit"""
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares


def mean_model(E0, lambda_param, Delta):
    """
    μ_E = λ * E_0 + Δ
    
    Parameters
    ----------
    E0 : array-like
        True energy values
    lambda_param : float
    Delta : float
        
    Returns
    -------
    mu : array-like
        Predicted mean values
    """
    return lambda_param * E0 + Delta


def resolution_model(E0, a, b, c):
    """
    σ_E / E_0 = sqrt(a²/E_0 + b²/E_0² + c²)
    
    Parameters
    ----------
    E0 : array-like
        True energy values
    a : float
    b : float
    c : float
        
    Returns
    -------
    sigma_over_E : array-like
        Predicted σ/E_0 values
    """
    term1 = (a / np.sqrt(E0))**2
    term2 = (b / E0)**2
    term3 = c**2
    return np.sqrt(term1 + term2 + term3)


def fit_mean_parameters(E0_values, means, mean_errors):
    """
    Fit linear model to mean values using iminuit.
    
    Parameters
    ----------
    E0_values : array-like
        True energy values
    means : array-like
        Sample mean values
    mean_errors : array-like
        Errors on means
        
    Returns
    -------
    minuit : Minuit object
        Fitted Minuit object with results
    params : dict
        Dictionary with parameter values
    errors : dict
        Dictionary with parameter errors
    """
    # Create least squares cost function
    least_squares = LeastSquares(E0_values, means, mean_errors, mean_model)
    
    # Create Minuit object
    m = Minuit(least_squares, lambda_param=1.0, Delta=0.0)
    
    # minimization
    m.migrad()  # Find minimum
    m.hesse()   # Calculate errors
    
    # Extract results
    params = {
        'lambda': m.values['lambda_param'],
        'Delta': m.values['Delta']
    }
    
    errors = {
        'lambda': m.errors['lambda_param'],
        'Delta': m.errors['Delta']
    }
    
    return m, params, errors


def fit_resolution_parameters(E0_values, stds, std_errors):
    """
    Fit resolution model to standard deviation values using iminuit.
    
    Parameters
    ----------
    E0_values : array-like
        True energy values
    stds : array-like
        Sample std values
    std_errors : array-like
        Errors on stds
        
    Returns
    -------
    minuit : Minuit object
        Fitted Minuit object with results
    params : dict
        Dictionary with parameter values
    errors : dict
        Dictionary with parameter errors
    """
    # Convert to σ/E_0
    sigma_over_E = np.array(stds) / np.array(E0_values)
    sigma_over_E_err = np.array(std_errors) / np.array(E0_values)
    
    # Create least squares cost function
    least_squares = LeastSquares(
        E0_values, 
        sigma_over_E, 
        sigma_over_E_err, 
        resolution_model
    )
    
    # Create Minuit object with initial guesses
    m = Minuit(least_squares, a=0.15, b=0.5, c=0.01)
    
    # Run minimization
    m.migrad()  # Find minimum
    m.hesse()   # Calculate errors

    # Extract results
    params = {
        'a': m.values['a'],
        'b': m.values['b'],
        'c': m.values['c']
    }
    
    errors = {
        'a': m.errors['a'],
        'b': m.errors['b'],
        'c': m.errors['c']
    }
    
    return m, params, errors


def bootstrap_fit(grouped_data, n_bootstrap=100):
    """
    Bootstrap analysis for fitting uncertainty.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary from group_by_energy()
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    results : dict
        Bootstrap results with parameter distributions
    """
    lambda_vals = []
    Delta_vals = []
    a_vals = []
    b_vals = []
    c_vals = []
    
    E0_list = sorted(grouped_data.keys())
    
    for i in range(n_bootstrap):
        # Resample each energy group
        boot_means = []
        boot_stds = []
        mean_err  = []
        std_err   = []
        
        for E0 in E0_list:
            E_rec = grouped_data[E0]['E_rec']
            # Resample with replacement
            boot_sample = np.random.choice(E_rec, size=len(E_rec), replace=True)
            
            N = len(boot_sample)
            mu_boot = np.mean(boot_sample)
            sigma_boot = np.std(boot_sample, ddof=1)
            
            boot_means.append(mu_boot)
            boot_stds.append(sigma_boot)
            
            # Errors computed from current bootstrap sample
            mean_err.append(sigma_boot / np.sqrt(N))
            std_err.append(sigma_boot / np.sqrt(2 * (N - 1)))
        
        # Fit to bootstrap sample
        try:
            # Fit mean using iminuit
            _, mean_params, _ = fit_mean_parameters(E0_list, boot_means, mean_err)
            
            # Fit resolution using iminuit
            _, res_params, _ = fit_resolution_parameters(E0_list, boot_stds, std_err)
            
            lambda_vals.append(mean_params['lambda'])
            Delta_vals.append(mean_params['Delta'])
            a_vals.append(res_params['a'])
            b_vals.append(res_params['b'])
            c_vals.append(res_params['c'])
        except Exception as e:
            # Skip failed fits
            print(f"Bootstrap sample {i} failed: {e}")
            continue
    
    return {
        'lambda': np.array(lambda_vals),
        'Delta': np.array(Delta_vals),
        'a': np.array(a_vals),
        'b': np.array(b_vals),
        'c': np.array(c_vals)
    }

def bootstrap_mle_trends(grouped_data, n_bootstrap=100):
    """
    Perform full bootstrap analysis for MLE trends (Exercise 2ii).
    
    1. Resample data for each E0
    2. Fit Gaussian to resampled data (get mu, sigma)
    3. Fit trends to these mu, sigma
    4. Repeat
    
    Parameters
    ----------
    grouped_data : dict
        Data dictionary
    n_bootstrap : int
        Number of iterations
        
    Returns
    -------
    boot_results : dict
        Distributions of parameters lambda, Delta, a, b, c
    """
    from s1_sol import mle_fits
    
    boot_results = {'lambda': [], 'Delta': [], 'a': [], 'b': [], 'c': []}
    E0_arr = np.array(sorted(grouped_data.keys()))
    
    for i in range(n_bootstrap):
        # Temporary lists for this iteration
        b_means, b_mean_errs = [], []
        b_stds, b_std_errs = [], []
        
        try:
            for E0 in E0_arr:
                # 1. Resample
                data = grouped_data[E0]['E_rec']
                resample = np.random.choice(data, size=len(data), replace=True)
                
                # 2. Fit Gaussian (MLE)
                # We suppress warnings/prints here for speed
                _, p, e = mle_fits.fit_gaussian_for_energy(resample, E0)
                
                b_means.append(p['mu'])
                b_mean_errs.append(e['mu'])
                b_stds.append(p['sigma'])
                b_std_errs.append(e['sigma'])
            
            # 3. Fit Trends
            _, mp, _ = fit_mean_parameters(E0_arr, b_means, b_mean_errs)
            _, rp, _ = fit_resolution_parameters(E0_arr, b_stds, b_std_errs)
            
            boot_results['lambda'].append(mp['lambda'])
            boot_results['Delta'].append(mp['Delta'])
            boot_results['a'].append(rp['a'])
            boot_results['b'].append(rp['b'])
            boot_results['c'].append(rp['c'])
            
        except Exception:
            continue
            
    return boot_results


def run_jackknife_analysis(grouped_data, sample_params=None, sample_errors=None, verbose=True):
    """
    Perform Jackknife analysis for all energy groups and fit trends.
    Uses the resample package for jackknife calculations.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary from group_by_energy()
    sample_params : dict, optional
        Sample estimates for comparison
    sample_errors : dict, optional
        Errors on sample estimates for comparison
    verbose : bool
        If True and sample_params provided, print comparison table
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - per_energy: dict of jackknife stats for each E0
        - params: fitted parameters (lambda, Delta, a, b, c) based on JK estimates
        - errors: errors on fitted parameters
    """
    from resample import jackknife

    def sample_std(x):
        return np.std(x, ddof=1)

    E0_list = sorted(grouped_data.keys())

    jk_means = []
    jk_mean_errs = []
    jk_stds = []
    jk_std_errs = []
    per_energy_results = {}

    for E0 in E0_list:
        data = grouped_data[E0]['E_rec']

        mean_jk = jackknife.bias_corrected(np.mean, data)
        mean_var = jackknife.variance(np.mean, data)
        std_jk = jackknife.bias_corrected(sample_std, data)
        std_var = jackknife.variance(sample_std, data)

        stats = {
            'mean_jk': mean_jk,
            'mean_err': np.sqrt(mean_var),
            'std_jk': std_jk,
            'std_err': np.sqrt(std_var)
        }

        per_energy_results[E0] = stats
        jk_means.append(mean_jk)
        jk_mean_errs.append(np.sqrt(mean_var))
        jk_stds.append(std_jk)
        jk_std_errs.append(np.sqrt(std_var))

    # Fit 
    _, mean_params, mean_errors = fit_mean_parameters(E0_list, jk_means, jk_mean_errs)
    _, res_params, res_errors = fit_resolution_parameters(E0_list, jk_stds, jk_std_errs)

    params = {**mean_params, **res_params}
    errors = {**mean_errors, **res_errors}

    if verbose and sample_params is not None and sample_errors is not None:
        for p in ['lambda', 'Delta', 'a', 'b', 'c']:
            diff = params[p] - sample_params[p]
            print(f"{p}: sample={sample_params[p]:.4f}, jk={params[p]:.4f}, diff={diff:+.2e}")

    return {
        "per_energy": per_energy_results,
        "params": params,
        "errors": errors
    }


def run_full_bootstrap_analysis(E_true_full, E_rec_full, grouped_data, n_bootstrap=2500):
    """
    Exercise 4: non-parametric bootstrap of the *entire* analysis.

    For each bootstrap replica:
      * resample events (E0, E_rec) with replacement;
      * re-group by E0;
      * calculate three methods:
          - sample estimates + LS trends,
          - individual MLE per energy + trends,
          - global simultaneous fit;
      * save (lambda, Delta, a, b, c) for each method.
    """
    from s1_sol import simultaneous_fit, mle_fits 

    E0_full = np.asarray(E_true_full, dtype=float)
    Erec_full = np.asarray(E_rec_full, dtype=float)
    N = len(E0_full)
    uniq_E0 = np.unique(E0_full)
    params = ['lambda', 'Delta', 'a', 'b', 'c']

    # make grouped
    def make_grouped(E0_arr, Erec_arr):
        grouped = {}
        for E0 in uniq_E0:
            mask = (E0_arr == E0)
            Erecs = Erec_arr[mask]
            if Erecs.size == 0:
                continue
            grouped[E0] = {
                'E0': np.full_like(Erecs, E0, dtype=float),
                'E_rec': np.asarray(Erecs, dtype=float),
            }
        return grouped

    # sample LS-trends
    def sample_trend(grouped):
        E0_list = sorted(grouped.keys())
        means, stds = [], []
        mean_errs, std_errs = [], []
        for E0 in E0_list:
            x = np.asarray(grouped[E0]['E_rec'], dtype=float)
            n = len(x)
            mu = np.mean(x)
            s = np.std(x, ddof=1)
            means.append(mu)
            stds.append(s)
            mean_errs.append(s / np.sqrt(n))
            std_errs.append(s / np.sqrt(2 * (n - 1)))
        _, mp, _ = fit_mean_parameters(E0_list, means, mean_errs)
        _, rp, _ = fit_resolution_parameters(E0_list, stds, std_errs)
        return {**mp, **rp}

    # individual MLE
    def individual_trend(grouped):
        E0_list = sorted(grouped.keys())
        res_mle = mle_fits.run_mle_fits(grouped)
        mus = [res_mle[E]['mu'] for E in E0_list]
        sigmas = [res_mle[E]['sigma'] for E in E0_list]
        mu_errs = [res_mle[E]['mu_err'] for E in E0_list]
        sigma_errs = [res_mle[E]['sigma_err'] for E in E0_list]
        _, mp, _ = fit_mean_parameters(E0_list, mus, mu_errs)
        _, rp, _ = fit_resolution_parameters(E0_list, sigmas, sigma_errs)
        return {**mp, **rp}

    boot_results = {
        'sample_ests':      {p: [] for p in params},
        'individual_fits':  {p: [] for p in params},
        'simultaneous_fit': {p: [] for p in params},
    }

    rng = np.random.default_rng()

    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        E0_b = E0_full[idx]
        Erec_b = Erec_full[idx]
        grouped_b = make_grouped(E0_b, Erec_b)
        if not grouped_b:
            continue

        try:
            vals_s = sample_trend(grouped_b)
            vals_i = individual_trend(grouped_b)
            _, vals_sim, _ = simultaneous_fit.run_simultaneous_fit(E0_b, Erec_b)
        except Exception as e:
            print(f"[bootstrap {b}] failed: {e}")
            continue

        for p in params:
            boot_results['sample_ests'][p].append(vals_s[p])
            boot_results['individual_fits'][p].append(vals_i[p])
            boot_results['simultaneous_fit'][p].append(vals_sim[p])

    for m in boot_results:
        for p in params:
            boot_results[m][p] = np.asarray(boot_results[m][p])

    E0_list = sorted(grouped_data.keys())

    # sample
    means, stds = [], []
    mean_errs, std_errs = [], []
    for E0 in E0_list:
        x = np.asarray(grouped_data[E0]['E_rec'], dtype=float)
        n = len(x)
        mu = np.mean(x)
        s = np.std(x, ddof=1)
        means.append(mu)
        stds.append(s)
        mean_errs.append(s / np.sqrt(n))
        std_errs.append(s / np.sqrt(2 * (n - 1)))
    _, mp_s, me_s = fit_mean_parameters(E0_list, means, mean_errs)
    _, rp_s, re_s = fit_resolution_parameters(E0_list, stds, std_errs)

    # individual MLE
    res_mle_full = mle_fits.run_mle_fits(grouped_data, verbose=False)
    mus = [res_mle_full[E]['mu'] for E in E0_list]
    sigmas = [res_mle_full[E]['sigma'] for E in E0_list]
    mu_errs = [res_mle_full[E]['mu_err'] for E in E0_list]
    sigma_errs = [res_mle_full[E]['sigma_err'] for E in E0_list]
    _, mp_i, me_i = fit_mean_parameters(E0_list, mus, mu_errs)
    _, rp_i, re_i = fit_resolution_parameters(E0_list, sigmas, sigma_errs)

    # simultaneous
    _, p_sim, e_sim = simultaneous_fit.run_simultaneous_fit(E_true_full, E_rec_full)

    methods_results = {
        'sample_ests': {
            'values': {**mp_s, **rp_s},
            'errors': {**me_s, **re_s},
        },
        'individual_fits': {
            'values': {**mp_i, **rp_i},
            'errors': {**me_i, **re_i},
        },
        'simultaneous_fit': {
            'values': p_sim,
            'errors': e_sim,
        },
    }

    boot_stats = {
        m: {
            p: {
                'mean': float(boot_results[m][p].mean()),
                'std': float(boot_results[m][p].std(ddof=1)),
            }
            for p in params
        }
        for m in boot_results
    }

    return {
        'boot_results': boot_results,
        'boot_stats': boot_stats,
        'methods': methods_results,
    }


def print_results(results, title="Parameter Estimates", format_type='params'):
    """
    Beautiful universal output formatter for results.
    
    Parameters
    ----------
    results : dict
        Dictionary with parameter results
    title : str
        Title for the output table
    format_type : str
        'params' - shows value ± error
        'stats' - shows mean, std, median (for bootstrap)
        'comparison' - shows multiple methods side by side
        
    Examples
    --------
    # Single method with errors:
    print_results({'lambda': 1.01, 'Delta': 2.03, ...}, 
                  errors={'lambda': 0.002, ...}, 
                  title="Simultaneous Fit")
    
    # Bootstrap statistics:
    print_results(boot_stats, title="Bootstrap", format_type='stats')
    """
    params = ['lambda', 'Delta', 'a', 'b', 'c']
    param_labels = {
        'lambda': 'λ',
        'Delta': 'Δ',
        'a': 'a',
        'b': 'b',
        'c': 'c'
    }
    
    print(f"\n{title}")
    print("="*70)
    
    if format_type == 'params':
        # Standard format: Parameter | Value ± Error
        print(f"{'Parameter':<12} | {'Value':<15} | {'Error':<15}")
        print("-"*70)
        
        # Handle if results is a dict with 'values' and 'errors' keys
        if 'values' in results and 'errors' in results:
            values = results['values']
            errors = results['errors']
        else:
            # Assume results has direct param keys
            values = results
            errors = results.get('errors', {})
        
        for param in params:
            if param in values:
                val = values[param]
                err = errors.get(param, 0.0)
                label = param_labels.get(param, param)
                print(f"{label:<12} | {val:>15.4f} | ±{err:<14.4f}")
                
    elif format_type == 'stats':
        # Bootstrap statistics format
        print(f"{'Parameter':<12} | {'Mean':<12} | {'Std Dev':<12} | {'Median':<12}")
        print("-"*70)
        
        for param in params:
            if param in results:
                stats = results[param]
                label = param_labels.get(param, param)
                
                # Handle both dict and direct values
                if isinstance(stats, dict):
                    mean = stats.get('mean', stats.get('value', 0))
                    std = stats.get('std', stats.get('error', 0))
                    median = stats.get('median', mean)
                else:
                    mean = stats
                    std = 0.0
                    median = mean
                    
                print(f"{label:<12} | {mean:>12.4f} | {std:>12.4f} | {median:>12.4f}")
                
    elif format_type == 'comparison':
        # Multi-method comparison (for Exercise 3.3 or 4.2)
        methods = list(results.keys())
        n_methods = len(methods)
        
        # Dynamic column width
        col_width = 15
        header = f"{'Parameter':<12} |"
        for method in methods:
            header += f" {method[:col_width]:<{col_width}} |"
        
        print(header)
        print("-"*70)
        
        for param in params:
            label = param_labels.get(param, param)
            row = f"{label:<12} |"
            
            for method in methods:
                method_data = results[method]
                if 'values' in method_data:
                    val = method_data['values'].get(param, 0.0)
                    err = method_data.get('errors', {}).get(param, 0.0)
                    cell = f"{val:.3f}±{err:.3f}"
                else:
                    val = method_data.get(param, 0.0)
                    cell = f"{val:.4f}"
                    
                row += f" {cell:<{col_width}} |"
                
            print(row)
    
    print("="*70)


def compare_trend_parameters(sample_params, sample_errors, mle_results, grouped_data):
    """
    Compare trend parameters from sample estimates vs MLE-based fits.
    
    Parameters
    ----------
    sample_params : dict
        Parameters from sample estimates {lambda, Delta, a, b, c}
    sample_errors : dict
        Errors on sample parameters
    mle_results : dict
        Results from run_mle_fits()
    grouped_data : dict
        Grouped data for fitting trends
        
    Returns
    -------
    df : pandas.DataFrame
        Comparison table
    """
    import pandas as pd
    
    E0_list = sorted(mle_results.keys())
    mle_means = [mle_results[E0]['mu'] for E0 in E0_list]
    mle_mean_errs = [mle_results[E0]['mu_err'] for E0 in E0_list]
    mle_stds = [mle_results[E0]['sigma'] for E0 in E0_list]
    mle_std_errs = [mle_results[E0]['sigma_err'] for E0 in E0_list]
    
    _, mle_mean_params, mle_mean_errors = fit_mean_parameters(E0_list, mle_means, mle_mean_errs)
    _, mle_res_params, mle_res_errors = fit_resolution_parameters(E0_list, mle_stds, mle_std_errs)
    
    mle_params = {**mle_mean_params, **mle_res_params}
    mle_errors = {**mle_mean_errors, **mle_res_errors}
    

    params = ['lambda', 'Delta', 'a', 'b', 'c']
    data = [{
        'parameter': p,
        'sample': sample_params[p],
        'σ(sample)': sample_errors[p],
        'mle': mle_params[p],
        'σ(mle)': mle_errors[p],
        '∆': mle_params[p] - sample_params[p]
    } for p in params]
    
    df = pd.DataFrame(data)
    display(df)
    return df, mle_params, mle_errors


def _convert_keys_for_json(data):
    """
    Convert internal parameter names to JSON output format.

    Required by assignment:
    - 'lambda' -> 'lb'
    - 'Delta' -> 'dE'
    - 'sample_estimates' -> 'sample_ests'

    Parameters
    ----------
    data : dict
        Dictionary with internal key names

    Returns
    -------
    converted : dict
        Dictionary with JSON-compatible key names
    """
    if not isinstance(data, dict):
        return data

    # Key mapping for parameter names
    key_mapping = {
        'lambda': 'lb',
        'Delta': 'dE',
        'sample_estimates': 'sample_ests'
    }

    converted = {}
    for key, value in data.items():
        # Map the key if it needs conversion
        new_key = key_mapping.get(key, key)

        # Recursively convert nested dictionaries
        if isinstance(value, dict):
            converted[new_key] = _convert_keys_for_json(value)
        else:
            converted[new_key] = value

    return converted


def save_results_to_json(results, filename='results.json'):
    """
    Save results dictionary to a JSON file in the project root.
    Handles numpy types serialization and automatic key name conversion.

    Automatically converts internal key names to assignment-required format:
    - 'lambda' -> 'lb'
    - 'Delta' -> 'dE'
    - 'sample_estimates' -> 'sample_ests'

    Parameters
    ----------
    results : dict
        Dictionary to save (with internal key names)
    filename : str
        Output filename (will be saved in project root)
    """
    import json
    from pathlib import Path

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Get project root directory (parent of s1_sol module)
    module_dir = Path(__file__).parent
    project_root = module_dir.parent
    filepath = project_root / filename

    # Load existing if present to update
    data = {}
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception:
            pass

    # Convert keys to JSON format before saving
    converted_results = _convert_keys_for_json(results)

    data.update(converted_results)

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)

    print(f"Results saved to {filepath}")
