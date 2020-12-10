from BSM_app import *

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
