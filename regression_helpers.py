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
        
def plot_smoother(ax, x, y, x_lim, n_knots, **kwargs):
    ncr = make_natural_cubic_regression(n_knots)
    ncr.fit(x, y)
    t = np.linspace(x_lim[0], x_lim[1], num=250)
    y_smoothed = ncr.predict(t.reshape(-1, 1))
    ax.plot(t, y_smoothed, **kwargs)
