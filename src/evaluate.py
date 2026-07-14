"""
evaluate.py

Evaluation metrics and the Diebold-Mariano test used to answer RQ1-RQ4 in
the synopsis. Implemented directly from the original formulas (rather than
relying on a specific forecasting library) so this module only needs
numpy/scipy, which install cleanly anywhere - no extra dependency risk for
the actual accuracy comparisons.

References:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
"""

import numpy as np
from scipy import stats


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def regression_metrics(y_true, y_pred) -> dict:
    """Convenience wrapper returning all three metrics at once."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def diebold_mariano_test(y_true, pred1, pred2, h: int = 1, loss: str = "squared"):
    """
    Diebold-Mariano test for equal predictive accuracy between two
    forecasts of the same series.

    Parameters
    ----------
    y_true : array-like, actual values
    pred1, pred2 : array-like, forecasts from model 1 and model 2
    h : forecast horizon (used for the variance correction; h=1 for
        one-step-ahead forecasts)
    loss : "squared" or "absolute" loss differential

    Returns
    -------
    dict with the DM statistic and a two-sided p-value.
    A significantly negative DM statistic (p < 0.05) means model 1 is
    more accurate than model 2; significantly positive means model 2 is
    more accurate.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)

    e1 = y_true - pred1
    e2 = y_true - pred2

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    d_bar = np.mean(d)

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if n > lag else 0.0
        var_d += 2 * cov
    var_d = var_d / n

    if var_d <= 0:
        dm_stat = np.nan
        p_value = np.nan
    else:
        dm_stat = d_bar / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {"dm_statistic": float(dm_stat), "p_value": float(p_value), "n": n}


def cross_correlation(x, y, max_lag: int = 12):
    """
    Cross-correlation of x (e.g. search_index) against y (e.g. turnover)
    at lags from -max_lag to +max_lag. A peak at a positive lag k means
    x at time t-k is most correlated with y at time t - i.e. x LEADS y by
    k periods, which is the specific evidence RQ1 asks for ("does search
    volume act as a leading indicator").

    Returns a pandas Series indexed by lag (requires pandas at call site
    to interpret; here we return two plain lists to keep this module
    dependency-light).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = list(range(-max_lag, max_lag + 1))
    correlations = []
    for lag in lags:
        if lag < 0:
            xi, yi = x[:lag], y[-lag:]
        elif lag > 0:
            xi, yi = x[lag:], y[:-lag]
        else:
            xi, yi = x, y
        if len(xi) < 2:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(xi, yi)[0, 1]))
    return lags, correlations


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 50)
    good_pred = y + rng.normal(0, 1, 50)
    bad_pred = y + rng.normal(0, 5, 50)

    print("Metrics (good model):", regression_metrics(y, good_pred))
    print("Metrics (bad model):", regression_metrics(y, bad_pred))
    print("DM test (good vs bad):", diebold_mariano_test(y, good_pred, bad_pred))

    lags, corrs = cross_correlation(good_pred, y, max_lag=5)
    print("Cross-correlation lags:", lags)
    print("Cross-correlation values:", [round(c, 2) for c in corrs])
