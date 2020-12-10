import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import os, urllib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from analytical_option_price import *
from bs_montecarlo import *
from option_greeks import *


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/bikram-sahu/Black-scholes/main/markdown_files/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("introduction.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Black-Scholes")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Select", "Option Price", "Monte Carlo", "Option Greeks", "Show the source code"])
    
    if app_mode == "Select":
        st.sidebar.success('To continue select from the dropdown.')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("BSM_app.py"))
    elif app_mode == "Option Price":
        readme_text.empty()
        run_option_price()
    elif app_mode == "Option Greeks":
        readme_text.empty()
        run_greeks()
    elif app_mode == "Monte Carlo":
        readme_text.empty()
        run_monte_carlo()

    return None



def d11(S, X, T, r, sigma):
    # Auxiliary function for d_one risk-adjusted probability
    return (np.log(S/X) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))


def d21(d1, T, sigma):
    # Auxiliary function for d_two risk-adjusted probability
    return d1 - sigma * np.sqrt(T)


def black_scholes(S, X, T, r, sigma, option_type):
    """Price a European option using the Black-Scholes option pricing formula.
    
    Arguments:
    S           -- the current spot price of the underlying stock
    X           -- the option strike price
    T           -- the time until maturity (in fractions of a year)
    r           -- the risk-free interest rate 
    sigma       -- the returns volatility of the underlying stock
    option_type -- the option type, either 'call' or 'put'
    
    Returns: a numpy.float_ representing the option value
    """
    d_one = d11(S, X, T, r, sigma)
    d_two = d21(d_one, T, sigma)
    if option_type == 'call':
        return S * norm.cdf(d_one) - np.exp(-r * T) * X * norm.cdf(d_two)
    elif option_type == 'put':
        return -(S * norm.cdf(-d_one) - np.exp(-r * T) * X * norm.cdf(-d_two))
    else:
        # Raise an error if the option_type is neither a call nor a put

        raise ValueError("Option type is either 'call' or 'put'.")


def run_monte_carlo():
    var = st.sidebar.selectbox("Select an option", [
                               "Algorithm", "Implementation"])

    if var == "Algorithm":
       st.markdown(get_file_content_as_string("monte_carlo_algo.md"))
       st.markdown("[Click here to access the Jupyter notebook of the code implemented.](https://github.com/bikram-sahu/Black-Scholes/blob/main/Monte-Carlo-BSM.ipynb)")
       st.markdown(
           "Select **Implementation** from dropdown box to see the results of Monte Carlo Simulation.")
    elif var == "Implementation":
        S0 = st.sidebar.slider('Stock price at t=0', 50, 200, 110)
        K = st.sidebar.slider('Strike price', 50, 200, 100)
        T = st.sidebar.slider('Time-to-Maturity', 0.0, 1.5, 0.5)
        r = st.sidebar.slider('Risk-free rate', 0.01, 0.10, 0.05)
        sigma = st.sidebar.slider('Sigma', 0.05, 0.60, 0.25)

        def simulate_stock_price(S0, r, sigma, dt, M, I):
            # Simulating I paths with M time step
            S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma *
                                      np.sqrt(dt) * np.random.standard_normal((M + 1, I)), axis=0))

            return S

        M = 50
        dt = T / M
        I = 250000
        S = simulate_stock_price(S0, r, sigma, dt, M, I)
        S[0] = S0

        # Calculating the Monte Carlo estimator
        C0 = np.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / I
        st.write('Option Price Calculated by Monte Carlo Value is:', C0)
        exact_C0 = black_scholes(S0, K, T, r, sigma, "call")
        st.write('Option Price Calculated from Analytical formula is:', exact_C0)

        fig, ax1 = plt.subplots()
        plt.plot(S[:, :10])
        ax1.set(xlabel='Steps', ylabel='Stock Price',
                title='Simulated Stock Price Using Euler Method.')
        st.pyplot(fig)


if __name__ == "__main__":
    main()
