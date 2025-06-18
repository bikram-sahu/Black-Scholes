import numpy as np
import pytest
from scipy.stats import norm

import BSM_app as bs


def test_black_scholes_call_put():
    S = 100
    X = 100
    T = 1
    r = 0.05
    sigma = 0.2

    call_price = bs.black_scholes(S, X, T, r, sigma, 'call')
    put_price = bs.black_scholes(S, X, T, r, sigma, 'put')

    assert call_price > 0
    assert put_price > 0


def test_call_put_parity():
    S = 100
    X = 100
    T = 1
    r = 0.05
    sigma = 0.2

    call_price = bs.black_scholes(S, X, T, r, sigma, 'call')
    put_price = bs.black_scholes(S, X, T, r, sigma, 'put')
    lhs = call_price - put_price
    rhs = S - X * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, rel=1e-8)


def test_bs_gamma():
    bs_gamma = getattr(bs, 'bs_gamma', None)
    if bs_gamma is None:
        pytest.skip('bs_gamma not available')

    S = 100
    X = 100
    T = 1
    r = 0.05
    sigma = 0.2

    result = bs_gamma(S, X, T, r, sigma, 'call')
    expected = norm.pdf(bs.d11(S, X, T, r, sigma)) / (S * sigma * np.sqrt(T))
    assert result == pytest.approx(expected, rel=1e-8)
