from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import os, urllib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import yfinance as yf
sns.set()


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/bikram-sahu/Black-scholes/main/markdown_files/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def display_source_code(path):
    url = 'https://raw.githubusercontent.com/bikram-sahu/Black-scholes/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")



def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("introduction.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Black-Scholes")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Select", "Derivation of Black-Scholes", "Option Pricing", "Option Greeks", "Monte Carlo",
                                     "IV vs Strike Price", "Delta Hedging", "VaR", "Show the source code"])
    
    if app_mode == "Select":
        st.sidebar.success('To continue select from the dropdown.')
    elif app_mode == "Derivation of Black-Scholes":
        readme_text.empty()
        st.markdown(get_file_content_as_string("BSM_equation_derivation.md"))
        with st.beta_expander('Assumptions of Black-Scholes'):
            st.markdown(get_file_content_as_string("BSM_assupmtions.md"))

    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(display_source_code("BSM_app.py"))
    elif app_mode == "Option Pricing":
        readme_text.empty()
        run_option_price()
    elif app_mode == "Option Greeks":
        readme_text.empty()
        run_greeks()
    elif app_mode == "Monte Carlo":
        readme_text.empty()
        run_monte_carlo()
    elif app_mode == "VaR":
        readme_text.empty()
        #value_at_risk()
        st.markdown('Under Development')
    elif app_mode == "IV vs Strike Price":
        readme_text.empty()
        iv_vs_strike()
    elif app_mode == "Delta Hedging":
        readme_text.empty()
        delta_hedging()

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


def run_option_price():

    st.markdown(get_file_content_as_string("bs-analytical-formula.md"))
    st.markdown("**Calculate Option Price from Black-Scholes formula:**")
    col1, col2, col3 = st.beta_columns(3)
    opt_type = col1.selectbox("Option type", ['call', 'put'])
    S = col2.number_input('Spot')
    X = col3.number_input('Strike')
    T = col3.number_input('Time-to-Maturity')
    sigma = col1.number_input('Volatility')
    r = col2.number_input('Risk-free interest')
    if st.button('Calculate'):
        price = black_scholes(S, X, T, r, sigma, opt_type)
        st.write("Black-Scholes Option Price: ", price)

    with st.beta_expander('Wish to learn more on how option price varies with respect to each parameter?'):
        col1, col2 = st.beta_columns((3, 1))
        var = col1.selectbox("Select a plot",
                             ["Select", "Option Price Vs Sigma", "Option Price Vs Stock Price", "Option Price Vs Time-to-Maturity"])
        option_type = col2.selectbox("Select a Option type", ['call', 'put'])

    if var == "Option Price Vs Sigma":
        r = st.sidebar.slider('Risk-free rate', 0.01, 0.1, 0.05)
        X = st.sidebar.slider('Strike price', 50, 150, 100)
        S = st.sidebar.slider('Stock price', 50, 150, 110)
        T = st.sidebar.slider('Time-to-Maturity', 0.0, 2.0, 0.5)
        sigma = np.arange(0.05, 0.61, 0.01)
        V = black_scholes(S, X, T, r, sigma, option_type)

        if option_type == 'call':
            col1, col2 = st.beta_columns((3, 1))
            col1.info('*You can vary other parameters from sidebar*')
            if S > X:
                col2.success('In the Money!')
            elif S < X:
                col2.error('Out of the Money!')
            elif S == X:
                col2.warning('At the Money!')

            df = pd.DataFrame({'Sigma': sigma, 'Call prices': V})
            fig, ax = plt.subplots()
            ax.scatter(df['Sigma'], df['Call prices'])
            ax.set(xlabel='Sigma', ylabel='Call Prices',
                   title='Option Prices Vs Sigma')
            st.pyplot(fig)
            with st.beta_expander('Explain me!'):
                st.write("Add Explaination")

        elif option_type == 'put':
            col1, col2 = st.beta_columns((3, 1))
            col1.info('*You can vary other parameters from sidebar*')
            if S < X:
                col2.success('In the Money!')
            elif S > X:
                col2.error('Out of the Money!')
            elif S == X:
                col2.warning('At the Money!')

            df = pd.DataFrame({'Sigma': sigma, 'Put prices': V})
            fig, ax = plt.subplots()
            ax.scatter(df['Sigma'], df['Put prices'])
            ax.set(xlabel='Sigma', ylabel='Put Prices',
                   title='Option Prices Vs Sigma')
            st.pyplot(fig)

            with st.beta_expander('Explain me!'):
                st.write("Add Explaination")

    elif var == "Option Price Vs Stock Price":
        st.info('*You can vary other parameters from sidebar*')
        r = st.sidebar.slider('Risk-free rate', 0.01, 0.1, 0.05)
        X = st.sidebar.slider('Strike price', 50, 150, 100)
        T = st.sidebar.slider('Time-to-Maturity', 0.0, 2.0, 0.5)
        sigma = st.sidebar.slider('Sigma', 0.05, 0.61, 0.25)
        S = np.arange(50, 150, 3)
        V = black_scholes(S, X, T, r, sigma, option_type)
        df = pd.DataFrame({'Stock price': S, 'Option prices': V})

        
        fig, ax = plt.subplots()
        df.plot('Stock price', 'Option prices', kind='scatter', ax=ax)
        ax.set(title="Option Price Vs Stock Price")
        st.pyplot(fig)
        with st.beta_expander('Explain me!'):
            st.write("Add Explaination")

    elif var == "Option Price Vs Time-to-Maturity":
        r = st.sidebar.slider('Risk-free rate', 0.01, 0.1, 0.05)
        X = st.sidebar.slider('Strike price', 50, 150, 100)
        S = st.sidebar.slider('Stock price', 50, 150, 110)
        sigma = st.sidebar.slider('Sigma', 0.05, 0.60, 0.25)
        T = np.arange(0.1, 1.6, 0.1)
        V = black_scholes(S, X, T, r, sigma, option_type)

        if option_type == 'call':
            col1, col2 = st.beta_columns((3, 1))
            col1.info('*You can vary other parameters from sidebar*')
            if S > X:
                col2.success('In the Money!')
            elif S < X:
                col2.error('Out of the Money!')
            elif S == X:
                col2.warning('At the Money!')
            
            df = pd.DataFrame({'T': T, 'Option prices': V})
            fig, ax = plt.subplots()
            df.plot('T', 'Option prices', kind='scatter', ax=ax)
            ax.set(title="Option Price Vs Time-to-Maturity")
            st.pyplot(fig)
            with st.beta_expander('Explain me!'):
                st.write("Add Explaination")
        
        if option_type == 'put':
            col1, col2 = st.beta_columns((3, 1))
            col1.info('*You can vary other parameters from sidebar*')
            if S < X:
                col2.success('In the Money!')
            elif S > X:
                col2.error('Out of the Money!')
            elif S == X:
                col2.warning('At the Money!')

            df = pd.DataFrame({'T': T, 'Option prices': V})
            fig, ax = plt.subplots()
            df.plot('T', 'Option prices', kind='scatter', ax=ax)
            ax.set(title="Option Price Vs Time-to-Maturity")
            st.pyplot(fig)
            with st.beta_expander('Explain me!'):
                st.write("Add Explaination")


def run_greeks():
    Greek = st.sidebar.selectbox(
        'Which Option Greek you want to study?', ("Select", "Delta", "Gamma", "Vega", "Theta"))

    if Greek == "Delta":
        r'''
        ### Delta - Sensitivity of Option price w.r.t change in stock price (S)
    
        * Call :
        $$\frac{\partial V}{\partial S} = N\left(d_{1}\right)$$

        * Put :
        $$\frac{\partial V}{\partial S} = -N\left(-d_{1}\right)=N\left(d_{1}\right)-1$$
        '''
        run_delta()

    elif Greek == "Gamma":
        r'''
        ### Gamma - Sensitivity of Delta w.r.t change in Stock price (S)

        $$
        \frac{\partial^{2} V}{\partial S^{2}} =
        \frac{N^{\prime}\left(d_{1}\right)}{S \sigma \sqrt{T-t}}
        $$
        '''
        run_gamma()
    elif Greek == "Vega":
        r'''
        ### Vega - Sensitivity of Option price w.r.t change in sigma (Volatility)
         
        For Both Call and Put Option:

        $$
        S N ^ {\prime}\left(d_{1}\right) \sqrt{T-t}
        $$
        '''
        run_vega()
    elif Greek == "Theta":

        r'''
        ### Theta - Sensitivity of Option price w.r.t change in Time-to-Maturity (T)

        * Call : 
        $$
        -\frac{S N^{\prime}\left(d_{1}\right) \sigma}{2 \sqrt{T-t}}-r K e^{-r(T-t)} N\left(d_{2}\right)
        $$

        * Put : 
        $$
        \frac{S N^{\prime}\left(d_{1}\right) \sigma}{2 \sqrt{T-t}}+r K e^{-r(T-t)} N\left(-d_{2}\right)
        $$
        '''
        run_theta()


def run_delta():

    def bs_delta(S, X, T, r, sigma, option_type):

        if option_type == 'call':
            return norm.cdf(d11(S, X, T, r, sigma))
        elif option_type == 'put':
            return -norm.cdf(-d11(S, X, T, r, sigma))
        else:
            # Raise an error if the option_type is neither a call nor a put
            raise ValueError("Option type is either 'call' or 'put'.")

    S = np.arange(10, 200, 3)
    X = 100
    r = 0.05
    T = 0.5
    sigma = 0.25

    delta_call = bs_delta(S, X, T, r, sigma, 'call')
    delta_put = bs_delta(S, X, T, r, sigma, 'put')

    df = pd.DataFrame(
        {'S': S, 'delta_call': delta_call, 'delta_put': delta_put})
    #print(df)

    fig, ax = plt.subplots()
    ax = df.plot('S', 'delta_call', kind='scatter',
                 color='green', label='call delta', ax=ax)
    df.plot('S', 'delta_put', kind='scatter',
            color='red', label='put delta', ax=ax)
    ax.set(xlabel='stock price', ylabel='delta',
           title='Delta for European Call & Put Option')

    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")

    S = np.arange(10, 200, 1)
    X = 100
    r = 0.05
    T = np.array([0.05, 0.25, 0.5])
    sigma = 0.25

    delta_call_t1 = bs_delta(S, X, T[0], r, sigma, 'call')
    delta_call_t2 = bs_delta(S, X, T[1], r, sigma, 'call')
    delta_call_t3 = bs_delta(S, X, T[2], r, sigma, 'call')

    df = pd.DataFrame({'S': S, 'T1': delta_call_t1,
                       'T2': delta_call_t2, 'T3': delta_call_t3})

    fig, ax = plt.subplots()
    ax = df.plot('S', 'T1', kind='line', color='green',
                 label='T = 0.05', ax=ax)
    df.plot('S', 'T2', kind='line', color='red', label='T = 0.25', ax=ax)
    df.plot('S', 'T3', kind='line', color='blue', label='T = 0.5', ax=ax)
    ax.set(xlabel='stock price', ylabel='delta',
           title='Delta for European Call & Put Option')

    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")

    S = 110
    X = [S*0.9, S, S*1.1]
    r = 0.05
    T = np.arange(1.0, 0, -0.005)
    sigma = 0.25

    delta_call_X1 = bs_delta(S, X[0], T, r, sigma, 'call')
    delta_call_X2 = bs_delta(S, X[1], T, r, sigma, 'call')
    delta_call_X3 = bs_delta(S, X[2], T, r, sigma, 'call')

    df = pd.DataFrame({'T': T, 'X1': delta_call_X1,
                       'X2': delta_call_X2, 'X3': delta_call_X3})

    fig, ax = plt.subplots()
    ax = df.plot('T', 'X1', kind='line', color='green',
                 label='K = S*0.9', ax=ax)
    df.plot('T', 'X2', kind='line', color='red', label='K = S', ax=ax)
    df.plot('T', 'X3', kind='line', color='blue', label='K = S*1.1', ax=ax)
    ax.set_xlim(T[0], T[-1])
    ax.set(xlabel='Time-to-Maturity', ylabel='delta',
           title='Delta for European Call & Put Option as a function of Time-to-Maturity')

    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")


def run_gamma():

    def bs_gamma(S, X, T, r, sigma, option_type):

        if option_type == 'call':
            return norm.pdf(d11(S, X, T, r, sigma))/(S*sigma*T)
        elif option_type == 'put':
            return norm.pdf(d11(S, X, T, r, sigma))/(S*sigma*T)
        else:
            # Raise an error if the option_type is neither a call nor a put
            raise ValueError("Option type is either 'call' or 'put'.")

    S = np.arange(10, 200, 1)
    X = 100
    r = 0.05
    T = np.array([0.05, 0.25, 0.5])
    sigma = 0.25

    gamma_call_t1 = bs_gamma(S, X, T[0], r, sigma, 'call')
    gamma_call_t2 = bs_gamma(S, X, T[1], r, sigma, 'call')
    gamma_call_t3 = bs_gamma(S, X, T[2], r, sigma, 'call')

    df = pd.DataFrame({'S': S, 'T1': gamma_call_t1,
                       'T2': gamma_call_t2, 'T3': gamma_call_t3})

    fig, ax = plt.subplots()
    ax = df.plot('S', 'T1', kind='line', color='green',
                 label='T = 0.05', ax=ax)
    df.plot('S', 'T2', kind='line', color='red', label='T = 0.25', ax=ax)
    df.plot('S', 'T3', kind='line', color='blue', label='T = 0.5', ax=ax)
    ax.set(xlabel='stock price', ylabel='gamma',
           title='Gamma for European Call Options as Time-to-Maturity varies')

    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")

    S = 110
    X = [S*0.8, S*0.9, S]
    r = 0.05
    T = np.arange(0.3, 0, -0.005)
    sigma = 0.5

    gamma_call_X1 = bs_gamma(S, X[0], T, r, sigma, 'call')
    gamma_call_X2 = bs_gamma(S, X[1], T, r, sigma, 'call')
    gamma_call_X3 = bs_gamma(S, X[2], T, r, sigma, 'call')

    df = pd.DataFrame({'T': T, 'X1': gamma_call_X1,
                       'X2': gamma_call_X2, 'X3': gamma_call_X3})

    fig, ax = plt.subplots()
    ax = df.plot('T', 'X1', kind='line', color='green',
                 label='K = S*0.8', ax=ax)
    df.plot('T', 'X2', kind='line', color='red', label='K = S*0.9', ax=ax)
    df.plot('T', 'X3', kind='line', color='blue', label='K = S', ax=ax)
    ax.set_xlim(T[0], T[-1])
    ax.set(xlabel='Time-to-Maturity', ylabel='gamma',
           title='Gamma for European Call & Put Option as a function of Time-to-Maturity')
    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")


def run_vega():

    def bs_vega(S, X, T, r, sigma, option_type):

        if option_type == 'call':
            return norm.pdf(d11(S, X, T, r, sigma)) * S*T
        elif option_type == 'put':
            return norm.pdf(d11(S, X, T, r, sigma)) * S*T
        else:
            # Raise an error if the option_type is neither a call nor a put
            raise ValueError("Option type is either 'call' or 'put'.")

    S = np.arange(10, 200, 1)
    X = 100
    r = 0.05
    T = np.array([0.05, 0.25, 0.5])
    sigma = 0.25

    vega_call_t1 = bs_vega(S, X, T[0], r, sigma, 'call')
    vega_call_t2 = bs_vega(S, X, T[1], r, sigma, 'call')
    vega_call_t3 = bs_vega(S, X, T[2], r, sigma, 'call')

    df = pd.DataFrame({'S': S, 'T1': vega_call_t1,
                       'T2': vega_call_t2, 'T3': vega_call_t3})

    fig, ax = plt.subplots()
    ax = df.plot('S', 'T1', kind='line', color='green',
                 label='T = 0.05', ax=ax)
    df.plot('S', 'T2', kind='line', color='red', label='T = 0.25', ax=ax)
    df.plot('S', 'T3', kind='line', color='blue', label='T = 0.5', ax=ax)
    ax.set(xlabel='stock price', ylabel='vega',
           title='Vega for European Call Options as Time-to-Maturity varies')
    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")

    S = 110
    X = [S*0.8, S*0.9, S]
    r = 0.05
    T = np.arange(1.0, 0, -0.005)
    sigma = 0.25

    vega_call_X1 = bs_vega(S, X[0], T, r, sigma, 'call')
    vega_call_X2 = bs_vega(S, X[1], T, r, sigma, 'call')
    vega_call_X3 = bs_vega(S, X[2], T, r, sigma, 'call')

    df = pd.DataFrame({'T': T, 'X1': vega_call_X1,
                       'X2': vega_call_X2, 'X3': vega_call_X3})

    fig, ax = plt.subplots()
    ax = df.plot('T', 'X1', kind='line', color='green',
                 label='K = S*0.8', ax=ax)
    df.plot('T', 'X2', kind='line', color='red', label='K = S*0.9', ax=ax)
    df.plot('T', 'X3', kind='line', color='blue', label='K = S', ax=ax)
    ax.set_xlim(T[0], T[-1])
    ax.set(xlabel='Time-to-Maturity', ylabel='vega',
           title='Vega for European Call & Put Option as a function of Time-to-Maturity')
    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")


def run_theta():
    def bs_theta(S, X, T, r, sigma, option_type):

        if option_type == 'call':
            return -norm.pdf(d11(S, X, T, r, sigma)) * S*sigma/(2*np.sqrt(T)) - r*X*np.exp(-r*T*norm.cdf(d21(d11(S, X, T, r, sigma), T, sigma)))
        elif option_type == 'put':
            return norm.pdf(d11(S, X, T, r, sigma)) * S*sigma/(2*np.sqrt(T)) + r*X*np.exp(-r*T*norm.cdf(-d21(d11(S, X, T, r, sigma), T, sigma)))
        else:
            # Raise an error if the option_type is neither a call nor a put
            raise ValueError("Option type is either 'call' or 'put'.")

    S = np.arange(10, 200, 1)
    X = 100
    r = 0.05
    T = np.array([0.05, 0.25, 0.5])
    sigma = 0.25

    theta_call_t1 = bs_theta(S, X, T[0], r, sigma, 'call')
    theta_call_t2 = bs_theta(S, X, T[1], r, sigma, 'call')
    theta_call_t3 = bs_theta(S, X, T[2], r, sigma, 'call')

    df = pd.DataFrame({'S': S, 'T1': theta_call_t1,
                       'T2': theta_call_t2, 'T3': theta_call_t3})

    fig, ax = plt.subplots()
    ax = df.plot('S', 'T1', kind='line', color='green',
                 label='T = 0.05', ax=ax)
    df.plot('S', 'T2', kind='line', color='red', label='T = 0.25', ax=ax)
    df.plot('S', 'T3', kind='line', color='blue', label='T = 0.5', ax=ax)
    ax.set(xlabel='stock price', ylabel='theta',
           title='Theta for European Call Options as Time-to-Maturity varies')

    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")

    S = 110
    X = [S*0.8, S*0.9, S]
    r = 0.05
    T = np.arange(1.0, 0, -0.005)
    sigma = 0.25

    theta_call_X1 = bs_theta(S, X[0], T, r, sigma, 'call')
    theta_call_X2 = bs_theta(S, X[1], T, r, sigma, 'call')
    theta_call_X3 = bs_theta(S, X[2], T, r, sigma, 'call')

    df = pd.DataFrame({'T': T, 'X1': theta_call_X1,
                       'X2': theta_call_X2, 'X3': theta_call_X3})

    fig, ax = plt.subplots()
    ax = df.plot('T', 'X1', kind='line', color='green',
                 label='K = S*0.8', ax=ax)
    df.plot('T', 'X2', kind='line', color='red', label='K = S*0.9', ax=ax)
    df.plot('T', 'X3', kind='line', color='blue', label='K = S', ax=ax)
    ax.set_xlim(T[0], T[-1])
    ax.set(xlabel='Time-to-Maturity', ylabel='theta',
           title='Theta for European Call & Put Option as a function of Time-to-Maturity')

    st.pyplot(fig)
    with st.beta_expander('Explain me!'):
        st.write("Add Explaination")


def run_monte_carlo():
    var = st.sidebar.selectbox("Select an option", [
                               "Algorithm", "Implementation"])

    if var == "Algorithm":
       st.markdown(get_file_content_as_string("monte_carlo_algo.md"))
       st.markdown("[Click here to access the Jupyter notebook of the code implemented.](https://github.com/bikram-sahu/Black-Scholes/blob/main/jupyter_notebooks/Monte-Carlo-BSM.ipynb)")
       st.markdown(
           "Select **Implementation** from dropdown box to see the results of Monte Carlo Simulation.")
    elif var == "Implementation":
        st.write('**Enter the parameter values below!**')
        col1, col2, col3 = st.beta_columns(3)
        #opt_type = col1.selectbox("Option type", ['call', 'put'])
        S0 = col1.number_input('Spot', 110)
        K = col2.number_input('Strike', 100)
        T = col3.number_input('Time-to-Maturity', 0.5)
        sigma = col1.number_input('Volatility', 0.25)
        r = col2.number_input('Risk-free rate', 0.05)
        
        def simulate_stock_price(S0, r, sigma, dt, M, I):
            # Simulating I paths with M time step
            S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma *
                                      np.sqrt(dt) * np.random.standard_normal((M + 1, I)), axis=0))
            return S
        
        M = st.sidebar.slider('Number of Steps:', 50, 500, 100)
        dt = T / M
        I = st.sidebar.slider('Number of Simulations', 100000, 250000)
        S = simulate_stock_price(S0, r, sigma, dt, M, I)
        S[0] = S0
        #if col2.button('Calculate'):
        
        # Calculating the Monte Carlo estimator
        C0 = np.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / I
        exact_C0 = black_scholes(S0, K, T, r, sigma, "call")
        ind = ["Analytical", "Monte Carlo", "Difference"]
        df = pd.DataFrame(data = [exact_C0, C0, np.abs(exact_C0 - C0)], index = ind, columns = ["Option Price"] )
        st.table(df)

        fig, ax1 = plt.subplots()
        plt.plot(S[:, :10])
        ax1.set(xlabel='Steps', ylabel='Stock Price',
                title='Simulated Stock Price Using Euler Method.')
        st.pyplot(fig)


def value_at_risk():

    st.markdown(get_file_content_as_string("var.md"))
    col1, col2, col3 = st.beta_columns(3)
    ticker_sym = col1.text_input('Enter a ticker symbol')
    start_date = col2.date_input('Start date')
    end_date = col3.date_input('End date')
    if col2.button('Calculate'):
        df = yf.download(ticker_sym, start_date, end_date)
        df = df[['Close']]
        df['returns'] = df.Close.pct_change()

        mean = np.mean(df['returns'])
        std_dev = np.std(df['returns'])

        df['returns'].hist(bins=40, normed=True, histtype='stepfilled', alpha=0.5)
        x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
        fig, ax1 = plt.subplots()
        plt.plot(x, mlab.normpdf(x, mean, std_dev), "r")
        ax1.set()
        st.pyplot(fig)


def iv_vs_strike():
    st.markdown(
        r'''
        ## Volatility Surface
        The incorrectness of Black-Scholes is most obviously manifested through the volatility surface
        - a concept that is found throughout derivatives markets.
        The volatility surface is constructed using market prices of European call and put options
        - can also use American prices but it's a little trickier.

        ### Definition. 
        The volatility surface, $\sigma(K, T),$ is a function of the strike, $K,$ and expiration, $T$. 
        It is defined implicitly by
        $$
        \large
        C_{\mathrm{mkt}}(S, K, T)=C_{\mathrm{BS}}(S, T, r, c, K, \sigma(K, T))
        $$

        where $C_{\mathrm{mkt}}(S, K, T)$ denotes the market price of the call option with expiration, $T,$ 
        and strike, $K,$ and $C_{\mathrm{BS}}(\cdot)$ is the corresponding Black-Scholes formula for pricing 
        this call option.

        ## Volatility Smile

        A volatility smile is a common graph shape that results from plotting the strike price and implied 
        volatility of a group of options with the same underlying asset and expiration date. The volatility 
        smile is so named because it looks like a smiling mouth. Implied volatility rises when the underlying 
        asset of an option is further out of the money (OTM) or in the money (ITM), compared to at the money (ATM).

        ''')
    #data = pd.read_excel('strike-vs-iv.xlsx')
    file = st.file_uploader('Upload a file containg STRIKE PRICE & IV')
    if file:
        data = pd.read_excel(file)
        fig, ax = plt.subplots()
        data.plot('STRIKE PRICE', 'IV', kind='line',
                    label='Implied Volatility', ax=ax)
        ax.set(xlabel='Strike Price', ylabel='Implied Volatility',
                title='IV vs Strike Price for Nifty')
        st.pyplot(fig)

def delta_hedging():
    #st.markdown(get_file_content_as_string("delta-hedging.md"))
    st.markdown(
    r'''
    Let us now view the option price as a function of $S$ and $\sigma$ only.
    A simple application of Taylor's Theorem says
    $$
    \large
    \begin{aligned} 
    C(S+\Delta S, \sigma+\Delta \sigma) & \approx C(S, \sigma)+\Delta S \frac{\partial C}{\partial S}+\frac{1}{2}(\Delta S)^{2} \frac{\partial^{2} C}{\partial S^{2}}+\Delta \sigma \frac{\partial C}{\partial \sigma} \\
    &=C(S, \sigma)+\Delta S \delta+ \frac{1}{2}(\Delta S)^{2} \Gamma+\Delta \sigma \text { vega. }
    \end{aligned}
    $$
    We therefore obtain
    $$
    \large
    \begin{aligned}
    \mathrm{P} \& \mathrm{~L} & \approx \delta \Delta S+\frac{\Gamma}{2}(\Delta S)^{2}+\text { vega } \Delta \sigma \\
    &=\text { delta } \mathrm{P} \& \mathrm{~L}+\text { gamma } \mathrm{P} \& \mathrm{~L}+\text { vega } \mathrm{P} \& \mathrm{~L}
    \end{aligned}
    $$
    Can also write
    $$
    \large
    \begin{aligned}
    \mathrm{P} \& \mathrm{~L} & \approx \delta S\left(\frac{\Delta S}{S}\right)+\frac{\Gamma S^{2}}{2}\left(\frac{\Delta S}{S}\right)^{2}+\text { vega } \Delta \sigma \\
    &=\mathrm{ESP} \times \text { Return }+\$ \text { Gamma } \times \text { Return }^{2}+\text { vega } \Delta \sigma
    \end{aligned}
    $$
    where ESP denotes the equivalent stock position or "dollar" delta.
    ''')
    with st.beta_expander('Example'):
        notional = 100
        Delta = [0.5]
        Gamma = [0.13]
        Vega = [1.2]
        Spot = [100]
        Strike = 100
        Sigma = [16]
        col1, col2, col3, col4 = st.beta_columns(4)
        with col1:
            st.write('Notional =', notional)
            st.write('Delta =', Delta[0])
            st.write('Gamma = ', Gamma[0])
            st.write('Vega = ', Vega[0])
        with col2:
            st.write('Spot = ', Spot[0])
            st.write('Strike =', Strike)
            st.write('Sigma =', Sigma[0])
        with col3:
            st.write('Futures sold', -1*notional*Delta[0])
        
        st.markdown('## Case 1')
        Delta.append(0.6)
        Gamma.append(0.09)
        Vega.append(0.8)
        Spot.append(105)
        Sigma.append(17)
        deltaS = (Spot[1] - Spot[0])
        col1, col2, col3, col4 = st.beta_columns(4)
        with col1:
            st.write('Delta =', Delta[1])
            st.write('Gamma = ', Gamma[1])
            st.write('Vega = ', Vega[1])
        with col2:
            st.write('Spot = ', Spot[1])
            st.write('Strike =', Strike)
            st.write('Sigma = ', Sigma[1])
        with col3:
            st.write('Futures sold', -1*notional*Delta[1])
            st.write('Future PnL', -1*notional*Delta[0]*deltaS)
        with col4:
            #st.info('Option PnL')
            
            st.write('Delta PnL', Delta[0]*notional* deltaS)
            st.write('Gamma PnL', 0.5* Gamma[0]* notional * deltaS**2)
            st.write('Vega PnL', Vega[0]*notional* (Sigma[1] - Sigma[0]))

        st.markdown('## Case 2')
        Delta.append(0.5)
        Gamma.append(0.13)
        Vega.append(1.2)
        Spot.append(100)
        Sigma.append(16)
        deltaS = (Spot[2] - Spot[1])
        col1, col2, col3, col4 = st.beta_columns(4)
        with col1:
            st.write('Delta =', Delta[2])
            st.write('Gamma = ', Gamma[2])
            st.write('Vega = ', Vega[2])
        with col2:
            st.write('Spot = ', Spot[2])
            st.write('Strike =', Strike)
            st.write('Sigma = ', Sigma[2])
        with col3:
            st.write('Futures sold', -1*notional*Delta[2])
            st.write('Future PnL', -1*notional*Delta[1]*deltaS)
        with col4:
            #st.info('Option PnL')

            st.write('Delta PnL', Delta[1]*notional * deltaS)
            st.write('Gamma PnL', 0.5 * Gamma[1] * notional * deltaS**2)
            st.write('Vega PnL', Vega[1]*notional * (Sigma[2] - Sigma[1]))



        

        







if __name__ == "__main__":
    main()
