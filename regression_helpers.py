import pandas as pd
from sklearn.utils import resample
from basis_expansions import NaturalCubicSpline


def plot_univariate_smooth(ax, x, y,
    x_lim=None, mask=None, smooth=True, n_knots=6, bootstrap=False):
    """Draw a scatter plot of some (x, y) data, and optionally superimpose
    a cubic spline.

    Parameters
    ----------
    ax: A Matplotlib axis object to draw the plot on.

    x: A np.array or pd.Series object containing the x data.

    y: A np.array or pd.Series object containing the y data.

    x_lim: A tuple contining limits for the x-axis of the plot.  If not
    supplied, this is computed as the minimum and maximum of x.

    mask: A boolean np.array or pd.Series containing a mask for the x and y
    data, if supplied only the unmasked data contributes to the plot.

    smooth: A boolean, draw the cubic spline or not?
    n_knots: The number of knots to use in the cubic spline.

    bootstrap: False or an integer.  The number of times to boostrap the data
    when drawing the spline.  If not False, draw one spline per bootstrap
    sample of the data.

    Returns:
    --------
    None
    """
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    if mask is not None:
        if isinstance(mask, pd.Series):
            mask = mask.values
        x = x[mask]
        y = y[mask]
    if not x_lim:
        x_lim = (np.min(x), np.max(x))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    
    ax.scatter(x, y, color='grey', alpha=0.25)
    if smooth:
        if bootstrap:
            for _ in range(bootstrap):
                x_boot, y_boot = resample(x, y)
                plot_smoother(ax, x_boot, y_boot, 
                              x_lim, n_knots, 
                              alpha=0.5, color="lightblue")        
        plot_smoother(ax, x, y, x_lim, n_knots, 
                      linewidth=3, color="blue")

def make_natural_cubic_regression(n_knots):
    return Pipeline([
        ('standardizer', StandardScaler()),
        ('nat_cubic', NaturalCubicSpline(-2, 2, n_knots=n_knots)),
        ('regression', LinearRegression(fit_intercept=True))
    ])

def plot_smoother(ax, x, y, x_lim, n_knots, **kwargs):
    ncr = make_natural_cubic_regression(n_knots)
    ncr.fit(x, y)
    t = np.linspace(x_lim[0], x_lim[1], num=250)
    y_smoothed = ncr.predict(t.reshape(-1, 1))
    ax.plot(t, y_smoothed, **kwargs)


def bootstrap_train(model, X, y, bootstraps=1000, **kwargs):
    """Train a (linear) model on multiple bootstrap samples of some data and
    return all of the parameter estimates.

    Parameters
    ----------
    model: A sklearn class whose instances have a `fit` method, and a `coef_`
    attribute.

    X: A two dimensional numpy array of shape (n_observations, n_features).
    
    y: A one dimensional numpy array of shape (n_observations).

    bootstraps: An integer, the number of boostrapped models to train.

    Returns
    -------
    bootstrap_coefs: A (bootstraps, n_features) numpy array.  Each row contains
    the parameter estimates for one trained boostrapped model.
    """
    bootstrap_coefs = np.empty(shape=(bootstraps, X.shape[1]))
    for i in range(bootstraps):
        boot_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_boot = X[boot_idxs, :]
        y_boot = y[boot_idxs]
        M = model(**kwargs)
        M.fit(X_boot, y_boot)
        bootstrap_coefs[i, :] = M.coef_
    return bootstrap_coefs

def plot_bootstrap_coefs(bootstrap_coefs, coef_names, n_col=3):
    """Plot histograms of the bootstrapped parameter estimates from a model.
    """
    n_coeffs = bootstrap_coefs.shape[1]
    n_row = int(ceil(n_coeffs / n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col*3, n_row*2))
    for idx, ax in enumerate(axs.flatten()):
        ax.hist(bootstrap_coefs[:, idx], bins=25, color="grey", alpha=0.5)
        ax.set_title(coef_names[idx])
    return fig, axs


def plot_partial_depenence(ax, model, X, var_name,
                           y=None, pipeline=None, n_points=250):
    """Create a partial dependence plot of a feature in a model.

    Parameters
    ----------
    ax: A matplotlib axis object to draw the partial dependence plot on.

    model: A trained sklearn model.  Must implement a `predict` method.

    X: The raw data to use in making predictions when drawing the partial
    dependence plot. Must be a pandas DataFrame.

    var_name: A string, the name of the varaible to make the partial dependence
    plot of.

    y: The y values, only needed if a scatter plot of x vs. y is desired.

    pipeline: A sklearn Pipeline object containing the transformations of the
    raw features used in the model.

    n_points: The number of points to use in the grid when drawing the plot.
    """
    Xpd = make_partial_dependence_data(
        X, var_name, n_points, pipeline=pipeline)
    x_plot = Xpd[var_name]
    if pipeline is not None:
        Xpd = pipeline.transform(Xpd)
    if y is not None:
        ax.scatter(X[var_name], y, color="grey", alpha=0.5)
    y_hat = model.predict(Xpd)
    ax.plot(x_plot, y_hat, linewidth=3, color="blue")

def make_partial_dependence_data(X, var_name,
                                 n_points=250,
                                 pipeline=None):
    Xpd = np.empty((n_points, X.shape[1]))
    Xpd = pd.DataFrame(Xpd, columns=X.columns)
    all_other_var_names = set(X.columns) - {var_name}
    for name in all_other_var_names:
        Xpd[name] = X[name].mean()
    min, max = np.min(X[var_name]), np.max(X[var_name])
    Xpd[var_name] = np.linspace(min, max, num=n_points)
    return Xpd
