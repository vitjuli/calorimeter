import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_style_path = Path(__file__).parent.parent / 'mphil.mplstyle'
if _style_path.exists():
    plt.style.use(['default', str(_style_path)])
else:
    plt.style.use('default')

ENERGY_COLORS = {
    10: '#8e44ad',  # Purple
    20: '#9b59b6',  # Light purple
    30: '#3498db',  # Blue
    40: '#5dade2',  # Light blue
    50: '#48c9b0',  # Teal
    60: '#52be80',  # Green
    70: '#f4d03f',  # Yellow-green
    80: '#f7dc6f',  # Yellow
}

PARAM_LABELS = {
    'lambda': r'$\lambda$',
    'lambda_param': r'$\lambda$',
    'Delta': r'$\Delta$ [GeV]',
    'a': r'$a$ [GeV$^{1/2}$]',
    'b': r'$b$ [GeV]',
    'c': r'$c$'
}

DEFAULT_FIGSIZE = (6.4, 4.8)

def setup_figure(figsize=DEFAULT_FIGSIZE):
    """Create a standard figure with nice defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, filename, dpi=300):
    """
    Save figure to the figs directory in project root.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (e.g., 'Figure1.1.pdf')
    dpi : int
        Resolution
    """
    module_dir = Path(__file__).parent
    project_root = module_dir.parent
    filepath = project_root / 'figs' / filename
    
    filepath.parent.mkdir(exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")

def plot_residuals(E_true, E_rec):
    """
    Figure 1.1: Total distribution of residuals (E_rec - E_true).
    """
    fig, ax = setup_figure()
    
    residuals = E_rec - E_true
    mean_res = np.mean(residuals)
    std_res = np.std(residuals, ddof=1)
    
    ax.hist(residuals, bins=50, color='#48c9b0', alpha=0.7, edgecolor='black', label='Data')
    
    ax.axvline(mean_res, color='#8e44ad', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_res:.3f} GeV')
    
    ax.axvline(0, color='#f7dc6f', linestyle=':', linewidth=1.5, alpha=1, 
               label='Zero')

    ax.set_xlabel(r'$E_{\rm rec} - E_{\rm true}$ [GeV]')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Sample: Residual Distribution')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3) 
    
    plt.tight_layout()
    return fig, ax

def plot_grouped_residuals(grouped_data):
    """
    Figure 1.2: Overlaid histograms of residuals per E_true.
    """
    fig, ax = setup_figure()
    
    E0_list = sorted(grouped_data.keys())
    
    for E0 in E0_list:
        residuals = grouped_data[E0]['E_rec'] - E0
        color = ENERGY_COLORS.get(E0, 'gray')
        ax.hist(residuals, bins=30, histtype='step', 
                label=f'$E_{{\\rm true}}={E0}$ GeV', 
                color=color, linewidth=2)
    
    ax.set_xlabel(r'$E_{\rm rec} - E_{\rm true}$ [GeV]')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distributions by True Energy')
    ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_grouped_with_gaussian(grouped_data):
    """
    Figure 1.2b: E_rec distributions with Gaussian PDFs (Grid).
    """
    E0_list = sorted(grouped_data.keys())
    n_energies = len(E0_list)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    fig.suptitle('Reconstructed Energy Distributions with Gaussian Fit (Sample Estimates)', 
                 fontsize=16)
    
    for i, E0 in enumerate(E0_list):
        ax = axes[i]
        E_rec = grouped_data[E0]['E_rec']
        
        mu = np.mean(E_rec)
        sigma = np.std(E_rec, ddof=1)
        
        color = ENERGY_COLORS.get(E0, 'gray')
        ax.hist(E_rec, bins=30, density=True, alpha=0.6, 
                color=color, edgecolor='black', linewidth=0.5, label='Data')
        
        x = np.linspace(E_rec.min(), E_rec.max(), 200)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'$\\mathcal{{N}}({mu:.1f}, {sigma:.1f}^2)$')
        
        ax.axvline(E0, color='#52be80', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'$E_{{\\rm true}} = {E0:.1f}$')
        
        ax.set_xlabel(r'$E_{\rm rec}$ [GeV]')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'$E_{{\\rm true}} = {E0:.1f}$ GeV', fontsize=11)
        ax.legend(fontsize='small', frameon=True, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    for j in range(n_energies, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig, axes


def plot_sample_estimates(E0_list, means, mean_errors, stds, std_errors):
    """
    Figure 1.3: Sample estimates vs True Energy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    # Means
    axes[0].errorbar(E0_list, means, yerr=mean_errors, fmt='o', capsize=5,
                     markersize=8, color='#3498db', elinewidth=2)
    axes[0].set_xlabel(r'$E_{\rm true}$ [GeV]')
    axes[0].set_ylabel(r'$\hat{\mu}_{\rm samp}$ [GeV]')
    axes[0].set_title('Sample Mean vs True Energy')
    axes[0].grid(True, alpha=0.3)
    
    # Standard deviations
    axes[1].errorbar(E0_list, stds, yerr=std_errors, fmt='o', capsize=5,
                     markersize=8, color='#f4d03f', elinewidth=2)
    axes[1].set_xlabel(r'$E_{\rm true}$ [GeV]')
    axes[1].set_ylabel(r'$\hat{\sigma}_{\rm samp}$ [GeV]')
    axes[1].set_title('Sample Std Dev vs True Energy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_trend_fits(E0_values, means, mean_errors, stds, std_errors,
                    mean_params, resolution_params, bootstrap_results=None):
    """
    Figure 1.4: Fitted trends for mean and resolution with error bands.
    Plots residuals (mu - E0) and relative resolution (sigma/E0).
    """
    from s1_sol import fitting
    
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    E0_fine = np.linspace(min(E0_values), max(E0_values), 200)
    
    lambda_param = mean_params['lambda']
    Delta = mean_params['Delta']
    
    residuals_mean = np.array(means) - np.array(E0_values)
    predicted_residuals = fitting.mean_model(E0_fine, lambda_param, Delta) - E0_fine
    
    ax[0].errorbar(E0_values, residuals_mean, yerr=mean_errors,
                   fmt='o', capsize=2, label=r'$\hat{\mu}_{\rm samp} - E_{\rm true}$', markersize=8, color='#3498db')
    ax[0].plot(E0_fine, predicted_residuals, linestyle='-', linewidth=2, 
               label=f'Fit: $\\lambda={lambda_param:.4f}, \\Delta={Delta:.3f}$', color='#8e44ad')
    
    if bootstrap_results is not None:
        boot_curves = []
        for i in range(len(bootstrap_results['lambda'])):
            l_i = bootstrap_results['lambda'][i]
            d_i = bootstrap_results['Delta'][i]
            curve = fitting.mean_model(E0_fine, l_i, d_i) - E0_fine
            boot_curves.append(curve)
        
        boot_curves = np.array(boot_curves)
        curve_std = np.std(boot_curves, axis=0)
        
        upper = predicted_residuals + curve_std
        lower = predicted_residuals - curve_std
        
        ax[0].fill_between(E0_fine, lower, upper, alpha=0.4, color='#52be80', label=r'$\pm 1\sigma$')
    
    ax[0].set_xlabel(r'$E_{\rm true}$ [GeV]')
    ax[0].set_ylabel(r'$\hat{\mu} - E_{\rm true}$ [GeV]')
    ax[0].set_title('Mean Residuals vs True Energy')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='best')
    
    a = resolution_params['a']
    b = resolution_params['b']
    c = resolution_params['c']
    
    sigma_over_E = np.array(stds) / np.array(E0_values)
    sigma_over_E_err = np.array(std_errors) / np.array(E0_values)
    predicted_resolution = fitting.resolution_model(E0_fine, a, b, c)
    
    ax[1].errorbar(E0_values, sigma_over_E, yerr=sigma_over_E_err,
                   fmt='s', capsize=2, color='#3498db', label=r'$\hat{\sigma}_{\rm samp}/E_{\rm true}$', markersize=8)
    ax[1].plot(E0_fine, predicted_resolution, linestyle='-', linewidth=2,
               label=f'Fit: $a={a:.3f}, b={b:.3f}, c={c:.4f}$', color='#8e44ad')
    
    if bootstrap_results is not None:
        boot_curves = []
        for i in range(len(bootstrap_results['a'])):
            a_i = bootstrap_results['a'][i]
            b_i = bootstrap_results['b'][i]
            c_i = bootstrap_results['c'][i]
            curve = fitting.resolution_model(E0_fine, a_i, b_i, c_i)
            boot_curves.append(curve)
            
        boot_curves = np.array(boot_curves)
        curve_std = np.std(boot_curves, axis=0)
        
        upper = predicted_resolution + curve_std
        lower = predicted_resolution - curve_std
        
        ax[1].fill_between(E0_fine, lower, upper, alpha=0.4, color='#52be80', label=r'$\pm 1\sigma$')
    
    ax[1].set_xlabel(r'$E_{\rm true}$ [GeV]')
    ax[1].set_ylabel(r'$\hat{\sigma} / E_{\rm true}$')
    ax[1].set_title('Relative Resolution vs True Energy')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='best')
    
    fig.tight_layout()
    return fig, ax


def plot_simultaneous_fit_results(ex1_results, sim_params, boot_results_sim):
    """
    Figure 3.1: Simultaneous fit results with bootstrap error bands.
    Wrapper around plot_trend_fits for convenience.
    
    Parameters
    ----------
    ex1_results : dict
        Results from Exercise 1 (sample estimates) for data points
    sim_params : dict
        Simultaneous fit parameters {lambda, Delta, a, b, c}
    boot_results_sim : dict
        Bootstrap results for error bands
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    mean_params = {'lambda': sim_params['lambda'], 'Delta': sim_params['Delta']}
    res_params = {'a': sim_params['a'], 'b': sim_params['b'], 'c': sim_params['c']}
    
    fig, ax = plot_trend_fits(
        ex1_results['E0_list'],
        ex1_results['means'], ex1_results['mean_errors'],
        ex1_results['stds'], ex1_results['std_errors'],
        mean_params, res_params,
        bootstrap_results=boot_results_sim
    )
    
    ax[0].set_title('Mean vs Energy (Simultaneous Fit)')
    ax[1].set_title('Resolution vs Energy (Simultaneous Fit)')
    
    return fig, ax

# 2: MLE FITS

def plot_mle_histograms(grouped_data, results_mle):
    """
    Figure 2.1: Individual MLE fits and total distribution.
    """
    from s1_sol import mle_fits

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    E0_list = sorted(grouped_data.keys())
    
    for E0 in E0_list:
        E_rec = grouped_data[E0]['E_rec']
        residuals = E_rec - E0
        color = ENERGY_COLORS.get(E0, 'gray')
        
        ax[0].hist(residuals, bins=40, density=True, alpha=0.3, 
                   color=color, label=f'{E0} GeV')
        
        x = np.linspace(residuals.min(), residuals.max(), 100)
        mu_fit = results_mle[E0]['mu']
        sigma_fit = results_mle[E0]['sigma']
        
        y = mle_fits.gaussian_pdf(x + E0, mu_fit, sigma_fit)
        ax[0].plot(x, y, color=color, linewidth=2)

    ax[0].set_title("Individual Fits (Normalised)")
    ax[0].set_xlabel(r"$E_{\rm rec} - E_{\rm true}$ [GeV]")
    ax[0].legend(ncol=2, fontsize='small')
    ax[0].grid(True, alpha=0.3)

    all_residuals = []
    for E0 in E0_list:
        all_residuals.extend(grouped_data[E0]['E_rec'] - E0)

    ax[1].hist(all_residuals, bins=50, density=True, color='#5dade2', alpha=0.6, label='All Data')
    
    x_total = np.linspace(min(all_residuals), max(all_residuals), 200)
    y_total = np.zeros_like(x_total)
    total_events = len(all_residuals)
    
    for E0 in E0_list:
        mu = results_mle[E0]['mu']
        sigma = results_mle[E0]['sigma']
        n_events = len(grouped_data[E0]['E_rec'])
        weight = n_events / total_events
        y_total += weight * mle_fits.gaussian_pdf(x_total + E0, mu, sigma)
        
    ax[1].plot(x_total, y_total, 'r--', linewidth=2, label='Sum of Fits')
    
    ax[1].set_title("Total Residual Distribution")
    ax[1].set_xlabel(r"$E_{\rm rec} - E_{\rm true}$ [GeV]")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_profile_likelihood(profile_results, E0):
    """
    Figure 2.3: 1D profile likelihoods for mu and sigma (single energy).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    mu_vals, nll_mu = profile_results['mu_scan']
    sigma_vals, nll_sigma = profile_results['sigma_scan']
    best_fit = profile_results['best_fit']
    
    ax = axes[0]
    delta_nll_mu = nll_mu - best_fit['nll_min']
    ax.plot(mu_vals, delta_nll_mu, 'b-', linewidth=2, label='Profile')
    ax.axvline(best_fit['mu'], color='#8e44ad', linestyle='--', label=f"$\\hat{{\\mu}}$")
    ax.axhline(0.5, color='#52be80', linestyle='--', label='1$\\sigma$ CL')
    
    mu_hesse = best_fit['mu_error']
    mu_parabola = (mu_vals - best_fit['mu'])**2 / (2 * mu_hesse**2)
    ax.plot(mu_vals, mu_parabola, 'r--', alpha=0.6, label="Hesse")
    
    ax.set_xlabel(r'$\mu$ [GeV]')
    ax.set_ylabel(r'$\Delta$ NLL')
    ax.set_title(f'Profile Likelihood: $\\mu$ (E₀ = {E0} GeV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    ax = axes[1]
    delta_nll_sigma = nll_sigma - best_fit['nll_min']
    ax.plot(sigma_vals, delta_nll_sigma, 'b-', linewidth=2, label='Profile')
    ax.axvline(best_fit['sigma'], color='#8e44ad', linestyle='--', label=f"$\\hat{{\\sigma}}$")
    ax.axhline(0.5, color='#52be80', linestyle='--', label='1$\\sigma$ CL')
    
    sigma_hesse = best_fit['sigma_error']
    sigma_parabola = (sigma_vals - best_fit['sigma'])**2 / (2 * sigma_hesse**2)
    ax.plot(sigma_vals, sigma_parabola, 'r--', alpha=0.6, label="Hesse")
    
    ax.set_xlabel(r'$\sigma$ [GeV]')
    ax.set_ylabel(r'$\Delta$ NLL')
    ax.set_title(f'Profile Likelihood: $\\sigma$ (E₀ = {E0} GeV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    plt.tight_layout()
    return fig, axes


def plot_methods_comparison(sample_params, sample_errors, 
                            mle_params, mle_errors,
                            sim_params, sim_errors,
                            boot_results=None):
    """
    Figure 3.2/4.2: Wrapper for comparing all three methods.
    
    Parameters
    ----------
    sample_params, sample_errors : dict
        Sample estimates parameters and errors
    mle_params, mle_errors : dict
        Individual MLE parameters and errors
    sim_params, sim_errors : dict
        Simultaneous fit parameters and errors
    boot_results : dict, optional
        Bootstrap results to add as 4th method
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    results_dict = {
        'sample_ests': {'values': sample_params, 'errors': sample_errors},
        'individual_fits': {'values': mle_params, 'errors': mle_errors},
        'simultaneous_fit': {'values': sim_params, 'errors': sim_errors}
    }
    
    return plot_parameter_comparison(results_dict, boot_results)


# 3: SIMULTANEOUS FIT

def plot_parameter_comparison(results_dict, boot_results=None):
    """
    Figure 3.2 / 4.2: Comparison of parameters from different methods.
    Layout: 2 rows x 3 cols (λ, Δ, _  /  a, b, c)
    """
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = ['Sample Estimation', 'Individual Fits', 'Simultaneous Fit']
    colors = ['#8e44ad', '#3498db', '#52be80']  # Purple, Blue, Green
    
    if boot_results:
        methods.append('bootstrap')
        method_labels.append('Bootstrap')
        colors.append('#f4d03f')  # Yellow
        
        b_values = {}
        b_errors = {}
        for p in ['lambda', 'Delta', 'a', 'b', 'c']:
            b_values[p] = np.mean(boot_results[p])
            b_errors[p] = np.std(boot_results[p])
            
        results_dict['bootstrap'] = {'values': b_values, 'errors': b_errors}
    
    params = ['lambda', 'Delta', 'a', 'b', 'c']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    
    for idx, param in enumerate(params):
        row, col = positions[idx]
        ax = axes[row, col]
        
        vals = []
        errs = []
        
        for method in methods:
            res = results_dict[method]
            
            p_key = param
            if param == 'lambda' and 'lambda' not in res['values']: p_key = 'lb'
            if param == 'Delta' and 'Delta' not in res['values']: p_key = 'dE'
            
            vals.append(res['values'][p_key])
            errs.append(res['errors'][p_key])
            
        x_positions = np.arange(len(methods))
        for j, (val, err) in enumerate(zip(vals, errs)):
            ax.errorbar(x_positions[j], val, yerr=err, fmt='o', capsize=4, 
                        color=colors[j], label=method_labels[j], markersize=8, 
                        elinewidth=2, markeredgecolor='black', markeredgewidth=0.5)
            
        label = PARAM_LABELS.get(param, param)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Value' if col == 0 else '')
        ax.set_xticks([])  # Hide x ticks
        ax.grid(True, alpha=0.3, axis='y')
        
        y_min = min([v - e for v, e in zip(vals, errs)])
        y_max = max([v + e for v, e in zip(vals, errs)])
        y_range = y_max - y_min
        if y_range == 0: y_range = 1.0
        ax.set_ylim(y_min - 0.2*y_range, y_max + 0.2*y_range)
        
        if idx == 0:
            ax.legend(loc='best', fontsize='small', framealpha=0.9)
    
    fig.delaxes(axes[0, 2])
            
    plt.tight_layout()
    return fig, axes


def plot_simultaneous_profiles(profile_results):
    """
    Figure 3.3a: 1D profile likelihoods for all simultaneous fit parameters.
    Uses 2x3 grid layout: (λ, Δ) on row 0, (a, b, c) on row 1.
    """
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 9.6))
    
    best_nll = profile_results['best_nll']
    
    param_mapping = {
        'lambda_param': 'lambda',
        'Delta': 'Delta',
        'a': 'a',
        'b': 'b',
        'c': 'c'
    }
    
    params = ['lambda_param', 'Delta', 'a', 'b', 'c']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    
    for idx, p_name in enumerate(params):
        row, col = positions[idx]
        ax = axes[row, col]
        
        vals, nlls = profile_results['scans'][p_name]
        delta_nll = nlls - best_nll
        
        result_name = param_mapping[p_name]
        best_val = profile_results['params'][result_name]
        error = profile_results['errors'][result_name]
        
        ax.plot(vals, delta_nll, 'b-', linewidth=2, label='Profile')
        
        parabola = (vals - best_val)**2 / (2 * error**2)
        ax.plot(vals, parabola, 'r--', linewidth=1.5, alpha=0.6, label='Hesse')
        
        ax.axhline(0.5, color='#52be80', linestyle='--', alpha=0.7, label=r'1$\sigma$')
        ax.axvline(best_val, color='#8e44ad', linestyle='--', alpha=0.5)
        
        label = PARAM_LABELS.get(p_name, p_name)
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\Delta$ NLL')
        ax.set_title(f'Profile: {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 3)
        
 
    axes[0, 2].axis('off')
    
    fig.tight_layout()
    return fig, axes

def plot_likelihood_grid_scan(E_true, E_rec, param_x, param_y,
                              param_labels=None,
                              n_points=120,
                              max_delta_nll=10.0):
    """
    2D likelihood surface: -2 Δ ln L 
    """
    from s1_sol.simultaneous_fit import run_simultaneous_fit, SimultaneousNLL

    if param_labels is None:
        param_labels = {
            'lambda': r'$\lambda$',
            'Delta':  r'$\Delta\,[\mathrm{GeV}]$',
            'a':      r'$a\,[\mathrm{GeV}^{1/2}]$',
            'b':      r'$b\,[\mathrm{GeV}]$',
            'c':      r'$c$'
        }

    #  simultaneous fit
    m, params, errors = run_simultaneous_fit(E_true, E_rec)
    best_nll = m.fval

    param_keys = ['lambda', 'Delta', 'a', 'b', 'c']
    px = param_x
    py = param_y

    def auto_range(p_name, scale=2.0):
        val = params[p_name]
        err = errors.get(p_name, 0.1 * abs(val) if val != 0 else 0.1)

        if abs(err) > abs(val) and val != 0:
            err = 0.5 * abs(val)

        lo = val - scale * err
        hi = val + scale * err

        if p_name in ['a', 'b', 'c'] and lo < 0:
            lo = 0.0

        if hi - lo < 1e-3 * max(1.0, abs(val)):
            width = max(1e-3, 0.1 * abs(val))
            lo = val - width
            hi = val + width

        return np.linspace(lo, hi, n_points)

    x_range = auto_range(px, scale=2.0)
    y_range = auto_range(py, scale=2.0)

    if px == 'a' and py == 'b':
        a_val, a_err = params['a'], errors['a']
        b_val, b_err = params['b'], errors['b']
        x_range = np.linspace(a_val - 2*a_err, a_val + 2*a_err, n_points)
        y_range = np.linspace(b_val - 2*b_err, b_val + 2*b_err, n_points)

    if px == 'a' and py == 'c':
        a_val, a_err = params['a'], errors['a']
        c_val, c_err = params['c'], errors['c']
        x_range = np.linspace(a_val - 2*a_err, a_val + 2*a_err, n_points)
        y_range = np.linspace(c_val - 2*c_err, c_val + 2*c_err, n_points)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    nll_obj = SimultaneousNLL(E_true, E_rec)
    defaults = [params[k] for k in param_keys]
    idx_x = param_keys.index(px)
    idx_y = param_keys.index(py)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            args = list(defaults)
            args[idx_x] = X[i, j]
            args[idx_y] = Y[i, j]
            Z[i, j] = 2.0 * (nll_obj(*args) - best_nll)

    fig, ax = plt.subplots(figsize=(5, 4))

    Z_clipped = np.clip(Z, 0, max_delta_nll)
    cf = ax.contourf(
        X, Y, Z_clipped,
        levels=np.linspace(0, max_delta_nll, 20),
        cmap='viridis_r', alpha=0.9
    )
    cbar = fig.colorbar(cf, label=r'$-2 \Delta \ln L$')

    levels = [2.30, 6.18]
    ax.contour(
        X, Y, Z,
        levels=levels,
        colors=['white', 'red'],
        linewidths=2
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='white', linewidth=2,
               label=r'$1\sigma$: $-2\Delta\ln L = 2.30$ (39.3%)'),
        Line2D([0], [0], color='red', linewidth=2,
               label=r'$2\sigma$: $-2\Delta\ln L = 6.18$ (86.5%)'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='red', markersize=10,
               markeredgecolor='black', linestyle='None',
               label='Best Fit')
    ]

    ax.plot(params[px], params[py], 'r*',
            markersize=12, markeredgecolor='black')

    label_x = param_labels.get(px, px)
    label_y = param_labels.get(py, py)

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(f'2D Profiling: {label_x} vs {label_y}')
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize='small', framealpha=0.9)

    plt.tight_layout()
    return fig, ax



# 4. BOOTSTRAP

def plot_bootstrap_histograms(boot_results, methods_results=None):
    """
    Figure 4.1
    """
    params = ['lambda', 'Delta', 'a', 'b', 'c']
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 9.6))
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]

    method_order = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = {
        'sample_ests': 'Sample estimation',
        'individual_fits': 'Individual MLE',
        'simultaneous_fit': 'Simultaneous fit',
    }
    method_colors = {
        'sample_ests': '#8e44ad',      # purple
        'individual_fits': '#3498db',  # blue
        'simultaneous_fit': '#f7dc6f', # yellow
    }

    for idx, param in enumerate(params):
        row, col = positions[idx]
        ax = axes[row, col]

        for m in method_order:
            if m not in boot_results:
                continue
            vals = np.asarray(boot_results[m][param])
            if vals.size == 0:
                continue

            ax.hist(
                vals,
                bins=30,
                histtype='step',
                density=True,
                alpha=1,
                color=method_colors[m],
                label=method_labels[m],
                linewidth=2
            )

            if methods_results is not None:
                v0 = methods_results[m]['values'][param]
                ax.axvline(v0,
                           color=method_colors[m],
                           linestyle='--',
                           linewidth=2,
                           alpha=0.9)

        label = PARAM_LABELS.get(param, param)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Density' if col == 0 else '')
        ax.set_title(f'Bootstrap distributions: {label}', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        if idx == 0:
            ax.legend(loc='upper right', fontsize='small', framealpha=0.9)

    fig.delaxes(axes[0, 2])
    plt.tight_layout()
    return fig, axes


def plot_jackknife_comparison(sample_params, sample_errors, jk_results):
    """
    Plot comparison of sample estimates vs jackknife estimates.
    Shows values with error bars and differences.
    """
    params = ['lambda', 'Delta', 'a', 'b', 'c']
    labels = [r'$\lambda$', r'$\Delta$', r'$a$', r'$b$', r'$c$']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Values with error bars
    ax = axes[0]
    x = np.arange(len(params))
    width = 0.35
    
    sample_vals = [sample_params[p] for p in params]
    sample_errs = [sample_errors[p] for p in params]
    jk_vals = [jk_results['params'][p] for p in params]
    jk_errs = [jk_results['errors'][p] for p in params]
    
    ax.errorbar(x - width/2, sample_vals, yerr=sample_errs, fmt='o', 
                capsize=4, color='#8e44ad', label='Sample', markersize=8)
    ax.errorbar(x + width/2, jk_vals, yerr=jk_errs, fmt='s', 
                capsize=4, color='#3498db', label='Jackknife', markersize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.set_title('Sample vs Jackknife Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Differences (jk - sample)
    ax = axes[1]
    diffs = [jk_results['params'][p] - sample_params[p] for p in params]
    colors = ['#52be80' if d >= 0 else '#e74c3c' for d in diffs]
    
    ax.bar(x, diffs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Difference (JK - Sample)')
    ax.set_title('Jackknife Bias')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, axes


def plot_mle_trend_profiles(results):
    """
    Plot profile chi-squared for MLE trend parameters (λ, Δ, a, b, c).
    
    Parameters
    ----------
    results : dict
        Output from compute_mle_trend_profiles()
    """
    fig = plt.figure(figsize=(12, 7))
    
    ax1 = fig.add_subplot(2, 3, 1)  # lambda
    ax2 = fig.add_subplot(2, 3, 2)  # Delta
    ax3 = fig.add_subplot(2, 3, 4)  # a
    ax4 = fig.add_subplot(2, 3, 5)  # b
    ax5 = fig.add_subplot(2, 3, 6)  # c
    
    axes_list = [ax1, ax2, ax3, ax4, ax5]
    param_names = ['lambda', 'Delta', 'a', 'b', 'c']
    param_labels = [r'$\lambda$', r'$\Delta$', r'$a$', r'$b$', r'$c$']
    
    for idx, (ax, p_name, p_label) in enumerate(zip(axes_list, param_names, param_labels)):
        scan_range, chi2_vals = results['scans'][p_name]
        
        chi2_min = np.min(chi2_vals)
        delta_chi2 = chi2_vals - chi2_min
        
        ax.plot(scan_range, delta_chi2, color='#3498db', linewidth=2)
        
        best_val = results['params'][p_name]
        ax.axvline(best_val, color='#e74c3c', linestyle='--', label='Best fit')
        
        ax.axhline(1, color='gray', linestyle=':', alpha=0.7, label=r'$\Delta\chi^2=1$')
        
        ax.set_xlabel(p_label)
        ax.set_ylabel(r'$\Delta\chi^2$')
        ax.set_title(f'Profile: {p_label}')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize='small')
    
    fig.delaxes(fig.add_subplot(2, 3, 3))
    
    plt.tight_layout()
    return fig, axes_list