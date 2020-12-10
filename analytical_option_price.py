from BSM_app import *


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

    with st.beta_expander('Wish to know more on how option price varies with respect to each parameter?'):
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
        r = st.sidebar.slider('Risk-free rate', 0.01, 0.1, 0.05)

        X = st.sidebar.slider('Strike price', 50, 150, 100)
        T = st.sidebar.slider('Time-to-Maturity', 0.0, 2.0, 0.5)
        sigma = st.sidebar.slider('Sigma', 0.05, 0.61, 0.25)
        S = np.arange(50, 150, 3)

        V = black_scholes(S, X, T, r, sigma, 'call')
        df = pd.DataFrame({'Stock price': S, 'Call prices': V})


        fig, ax = plt.subplots()
        df.plot('Stock price', 'Call prices', kind='scatter', ax=ax)
        ax.set(title="Option Price Vs Stock Price")
        st.pyplot(fig)

    elif var == "Option Price Vs Time-to-Maturity":
        r = st.sidebar.slider('Risk-free rate', 0.01, 0.1, 0.05)

        X = st.sidebar.slider('Strike price', 50, 150, 100)
        S = st.sidebar.slider('Stock price', 50, 150, 110)
        sigma = st.sidebar.slider('Sigma', 0.05, 0.60, 0.25)
        T = np.arange(0.1, 1.6, 0.1)
        if S > X:
            st.sidebar.markdown('In the Money!')
        elif S < X:
            st.sidebar.markdown('Out of the Money!')
        elif S == X:
            st.sidebar.markdown('At the Money!')

        V = black_scholes(S, X, T, r, sigma, 'call')
        df = pd.DataFrame({'T': T, 'Call prices': V})
        if st.checkbox('Show data'):
            df

        fig, ax = plt.subplots()
        df.plot('T', 'Call prices', kind='scatter', ax=ax)
        ax.set(title="Option Price Vs Time-to-Maturity")
        st.pyplot(fig)
