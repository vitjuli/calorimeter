"""
Simultaneous Maximum Likelihood Estimation fit.
Fits all data at once to the detector model.
"""
import numpy as np
from iminuit import Minuit

class SimultaneousNLL:
    """
    Negative Log-Likelihood function for simultaneous fit.
    """
    def __init__(self, E_true, E_rec):
        self.E_true = E_true
        self.E_rec = E_rec
        
    def __call__(self, lambda_param, Delta, a, b, c):
        """
        Calculate NLL for given parameters.
        
        Parameters
        ----------
        lambda_param, Delta : float
            Mean model parameters: mu = lambda * E + Delta
        a, b, c : float
            Resolution model parameters: sigma/E = sqrt(a^2/E + b^2/E^2 + c^2)
            
        Returns
        -------
        nll : float
            Negative Log-Likelihood value
        """
        mu = lambda_param * self.E_true + Delta
        
        term_a = (a / np.sqrt(self.E_true))**2
        term_b = (b / self.E_true)**2
        term_c = c**2
        
        sigma_over_E = np.sqrt(term_a + term_b + term_c)
        sigma = sigma_over_E * self.E_true
        
        z_score = (self.E_rec - mu) / sigma
        nll = np.sum(np.log(sigma) + 0.5 * z_score**2)
        
        return nll

def run_simultaneous_fit(E_true, E_rec):
    """
    Run the simultaneous fit using iminuit.
    
    Parameters
    ----------
    E_true : array-like
        True energy values for all events
    E_rec : array-like
        Reconstructed energy values for all events
        
    Returns
    -------
    minuit : Minuit
        Fitted object
    params : dict
        Best fit parameters
    errors : dict
        Parameter errors
    """
    nll = SimultaneousNLL(E_true, E_rec)
    
    m = Minuit(nll, 
               lambda_param=1.0, 
               Delta=0.0, 
               a=0.5, 
               b=0.5, 
               c=0.05)
    
    m.errordef = Minuit.LIKELIHOOD
    
    m.migrad()
    m.hesse()
    
    params = {
        'lambda': m.values['lambda_param'],
        'Delta': m.values['Delta'],
        'a': m.values['a'],
        'b': m.values['b'],
        'c': m.values['c']
    }
    
    errors = {
        'lambda': m.errors['lambda_param'],
        'Delta': m.errors['Delta'],
        'a': m.errors['a'],
        'b': m.errors['b'],
        'c': m.errors['c']
    }
    
    
    return m, params, errors


def bootstrap_simultaneous_fit(E_true, E_rec, n_bootstrap=100):
    """
    Perform bootstrap analysis for simultaneous fit.
    
    Parameters
    ----------
    E_true, E_rec : array-like
        Data arrays
    n_bootstrap : int
        Number of iterations
        
    Returns
    -------
    boot_results : dict
        Distributions of parameters
    """
    boot_results = {'lambda': [], 'Delta': [], 'a': [], 'b': [], 'c': []}
    n_events = len(E_true)
    
    for i in range(n_bootstrap):
        
        # Resample indices
        indices = np.random.randint(0, n_events, size=n_events)
        E_true_boot = E_true[indices]
        E_rec_boot = E_rec[indices]
        
        try:
            nll = SimultaneousNLL(E_true_boot, E_rec_boot)
            
            m = Minuit(nll, lambda_param=1.0, Delta=0.0, a=0.5, b=0.5, c=0.05)
            m.limits['a'] = (0, None)
            m.limits['b'] = (0, None)
            m.limits['c'] = (0, None)
            m.errordef = Minuit.LIKELIHOOD
            
            m.migrad()
            
            if m.valid:
                boot_results['lambda'].append(m.values['lambda_param'])
                boot_results['Delta'].append(m.values['Delta'])
                boot_results['a'].append(m.values['a'])
                boot_results['b'].append(m.values['b'])
                boot_results['c'].append(m.values['c'])
                
        except Exception:
            continue
                   
    return boot_results

