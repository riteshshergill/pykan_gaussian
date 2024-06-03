import torch

def G_batch(x, mu, sigma, device='cpu'):
    '''
    Evaluate x on Gaussian bases.
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of functions, number of samples)
        mu : 2D torch.tensor
            means of Gaussian functions, shape (number of functions, number of bases)
        sigma : 2D torch.tensor
            standard deviations of Gaussian functions, shape (number of functions, number of bases)
        device : str
            device
    
    Returns:
    --------
        gaussian values : 3D torch.tensor
            shape (number of functions, number of Gaussian bases, number of samples)
      
    Example
    -------
    >>> num_func = 5
    >>> num_sample = 100
    >>> num_basis = 10
    >>> x = torch.normal(0,1,size=(num_func, num_sample))
    >>> mu = torch.linspace(-1, 1, steps=num_basis).repeat(num_func, 1)
    >>> sigma = torch.ones(num_func, num_basis)
    >>> G_batch(x, mu, sigma).shape
    torch.Size([5, 10, 100])
    '''
    x = x.to(device)
    mu = mu.to(device)
    sigma = sigma.to(device)
    
    # Expand dimensions to match shapes for broadcasting
    x = x.unsqueeze(1)  # shape (num_func, 1, num_sample)
    mu = mu.unsqueeze(2)  # shape (num_func, num_basis, 1)
    sigma = sigma.unsqueeze(2)  # shape (num_func, num_basis, 1)
    
    # Evaluate Gaussian functions
    gaussian_values = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    return gaussian_values

def coef2gaussian(x_eval, coef, mu, sigma, device="cpu"):
    '''
    Converting coefficients to Gaussian functions. Evaluate x on Gaussian functions.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of functions, number of samples)
        coef : 2D torch.tensor
            shape (number of functions, number of coefficients)
        mu : 2D torch.tensor
            shape (number of functions, number of coefficients)
        sigma : 2D torch.tensor
            shape (number of functions, number of coefficients)
        device : str
            device
    
    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of functions, number of samples)
    '''
    x_eval = x_eval.to(device)
    coef = coef.to(device)
    mu = mu.to(device)
    sigma = sigma.to(device)
    
    # Evaluate Gaussian bases using G_batch
    gaussians = G_batch(x_eval, mu, sigma, device=device)
    
    # Sum over coefficients to get the evaluated values
    y_eval = (coef.unsqueeze(2) * gaussians).sum(dim=1)
    
    return y_eval

def gaussian2coef(x_eval, y_eval, mu, sigma, device="cpu"):
    '''
    Converting Gaussian functions to coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of functions, number of samples)
        y_eval : 2D torch.tensor
            shape (number of functions, number of samples)
        mu : 2D torch.tensor
            shape (number of functions, number of coefficients)
        sigma : 2D torch.tensor
            shape (number of functions, number of coefficients)
        device : str
            device
    
    Returns:
    --------
        coef : 2D torch.tensor
            shape (number of functions, number of coefficients)
    '''
    x_eval = x_eval.to(device)
    y_eval = y_eval.to(device)
    mu = mu.to(device)
    sigma = sigma.to(device)
    
    # Evaluate Gaussian basis functions using G_batch
    gaussians = G_batch(x_eval, mu, sigma, device=device).permute(0, 2, 1)
    
    y_eval_unsqueezed = y_eval.unsqueeze(dim=2).to(device)
    
    try:
        coef = torch.linalg.lstsq(gaussians, y_eval_unsqueezed, driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
        return coef.to(device)
    except Exception as e:
        print(f"Error in lstsq: {e}")
        print(f"gaussians shape: {gaussians.shape}, y_eval_unsqueezed shape: {y_eval_unsqueezed.shape}")
        raise e
