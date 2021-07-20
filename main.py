import altair as alt
from bs4 import BeautifulSoup
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='Modern Portfolio Theory')

st.write("""
# Modern Portfolio Theory

_A web app by [Arkadiusz Hryc](https://arkadiuszhryc.gitlab.io)_

This script allows to find the minimum variance portfolio (MVP) based for stocks available via [Stooq](stooq.com) and to
plot the efficient frontier.

For detailed information about the calculation procedure of MVP and efficient frontier, please see the
[References](#references) section.

## Calculate MVP and efficient frontier
""")

tickers = st.text_input(label='Please enter tickers from Stooq (separated by comma and space):',
                        value='pko, dnp, pkn')
ticker_list = tickers.split(', ')


def get_close(ticker):
    data_url = 'https://stooq.com/q/d/l/?s=' + ticker + '&i=d'
    stooq_data = pd.read_csv(data_url)
    if len(stooq_data) == 0:
        return None
    else:
        stooq_data['Date'] = pd.to_datetime(stooq_data['Date'], format='%Y-%m-%d')
        stooq_data.rename(columns={'Close': ticker.upper()}, inplace=True)
        stooq_data = stooq_data.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
        return stooq_data


with st.spinner('Fetching data, please wait...'):
    for i, j in enumerate(ticker_list):
        if i == 0:
            data = get_close(j)
            if data is None:
                st.warning('Ticker "' + j +
                           '" not found. Ignoring this ticker for now. Please check spelling and try again.')
        elif data is None:
            data = get_close(j)
        else:
            new_data = get_close(j)
            if new_data is None:
                st.warning('Ticker "' + j +
                           '" not found. Ignoring this ticker for now. Please check spelling and try again.')
            else:
                data = data.merge(new_data, how='outer', on='Date')

    data = data.sort_values(by='Date')
    data.iloc[:, 1:] = data.iloc[:, 1:].astype('float')
    data = data.set_index('Date')
    data.fillna(method='ffill')

    identified_tickers = pd.DataFrame(columns=['ticker', 'match'])
    for k, l in enumerate(data.columns):
        profile_url = 'https://stooq.com/q/p/?s=' + l.lower()
        req = requests.get(profile_url)
        soup = BeautifulSoup(req.content, 'html.parser')
        full_name = soup.find(id='f14').get_text()
        identified_tickers = identified_tickers.append(pd.DataFrame(data=[[l.upper(), full_name]],
                                                                    columns=identified_tickers.columns,
                                                                    index=[k+1]))

st.write("""
Identified tickers:
""")
st.write(identified_tickers)

with st.spinner('Computing...'):
    log_ret = np.log(data) - np.log(data.shift(1))
    log_ret = log_ret.tail(n=252)

    stock_data = pd.DataFrame(data=None,
                              index=log_ret.columns,
                              columns=['mu', 'sigma'],
                              dtype='float64')

    for i in log_ret.columns:
        stock_data.loc[i, 'mu'] = np.mean(log_ret[i])
        stock_data.loc[i, 'sigma'] = np.std(log_ret[i], ddof=1)

    stock_data_cov = log_ret.cov()

    matrix_C = stock_data_cov.merge(pd.DataFrame(data=np.ones(
        len(log_ret.columns)),
        index=log_ret.columns,
        columns=['ones'],
        dtype='float64'),
        left_index=True,
        right_index=True)
    matrix_C = matrix_C.append(pd.DataFrame(data=np.append(
        np.ones(len(log_ret.columns)), 0).reshape(1, len(log_ret.columns) + 1),
                                            index=['ones'],
                                            columns=matrix_C.columns,
                                            dtype='float64'))

    matrix_I = np.append(np.zeros(len(log_ret.columns)),
                         1).reshape(len(log_ret.columns) + 1, 1)

    w = np.matmul(np.linalg.inv(matrix_C),
                  matrix_I.reshape(len(matrix_C.columns), 1))

    stock_data = stock_data.merge(pd.DataFrame(data=w[:-1],
                                               index=log_ret.columns,
                                               columns=['w'],
                                               dtype='float64'),
                                  left_index=True,
                                  right_index=True)

    stock_data['mu_annual'] = stock_data['mu'] * 252
    stock_data['sigma_annual'] = stock_data['sigma'] * math.sqrt(252)

    mu_min = np.dot(stock_data['w'].to_numpy(),
                    stock_data['mu'].to_numpy()) * 252

    sigma_min = math.sqrt(np.matmul(np.matmul(stock_data['w'].to_numpy().reshape(
        1, len(log_ret.columns)),
        stock_data_cov.to_numpy()),
        stock_data['w'].to_numpy().reshape(
            len(log_ret.columns), 1))) * math.sqrt(252)

    min_portfolio = pd.DataFrame(data=[[1, mu_min, sigma_min]],
                                 columns=['weight', 'return (annual)', 'standard deviation (annual)'],
                                 index=['portfolio'],
                                 dtype='float64')

    mvp_to_print = stock_data.iloc[:, 2:].copy()
    mvp_to_print.columns = min_portfolio.columns
    mvp_to_print = mvp_to_print.append(min_portfolio)

st.write("""
Minimal variance portfolio:
""")
st.write(mvp_to_print.style.format({'weight': "{:.2%}",
                                    'return (annual)': "{:.2%}",
                                    'standard deviation (annual)': "{:.4f}"}))

with st.spinner('Computing...'):
    matrix_D = matrix_C.merge(pd.DataFrame(data=np.append(
        stock_data['mu'].to_numpy(), 0),
                                           index=matrix_C.columns,
                                           columns=['r'],
                                           dtype='float64'),
                                    left_index=True,
                                    right_index=True)
    matrix_D = matrix_D.append(pd.DataFrame(data=np.append(
        stock_data['mu'].to_numpy(), np.zeros(2)).reshape(
        1, len(matrix_C.columns) + 1),
                                            index=['r'],
                                            columns=matrix_D.columns,
                                            dtype='float64'))

    r_sigma = pd.DataFrame(data=np.linspace(np.min(stock_data['mu']),
                                            np.max(stock_data['mu']),
                                            num=401),
                           columns=['r'])

    rows = np.empty((1, len(log_ret.columns) + 1))
    for i in r_sigma['r']:
        matrix_I0 = np.append(matrix_I, i).reshape(len(log_ret.columns) + 2, 1)

        wages = np.matmul(np.linalg.inv(matrix_D),
                          matrix_I0.reshape(len(matrix_D.columns), 1))[:-2]
        rows = np.append(rows,
                         np.append(wages, i).reshape(1, len(log_ret.columns) + 1),
                         axis=0)

    r_sigma = r_sigma.merge(pd.DataFrame(data=rows[1:],
                                         columns=np.append(log_ret.columns, 'r')),
                            on='r')

    rows2 = np.empty((1, 2))
    for i in r_sigma.index:
        sigma = math.sqrt(np.matmul(np.matmul(
            r_sigma.iloc[i, 1:].to_numpy().reshape(1, len(log_ret.columns)),
            stock_data_cov.to_numpy()),
            r_sigma.iloc[i, 1:].to_numpy().reshape(
                len(log_ret.columns), 1)))
        rows2 = np.append(rows2,
                          np.append(sigma, r_sigma.iloc[i, 0]).reshape(1, 2),
                          axis=0)

    r_sigma = r_sigma.merge(pd.DataFrame(data=rows2[1:],
                                         columns=['sigma', 'r']),
                            on='r')

    r_sigma['r'] = r_sigma['r'] * 252
    r_sigma['sigma'] = r_sigma['sigma'] * math.sqrt(252)

    ef_chart = alt.Chart(r_sigma).mark_circle().encode(
        alt.X('sigma', scale=alt.Scale(zero=False), axis=alt.Axis(format='%', title='standard deviation (annual)')),
        alt.Y('r', scale=alt.Scale(zero=False), axis=alt.Axis(format='%', title='expected return rate (annual)')),
        tooltip=[alt.Tooltip('r', format='p', title='return rate'),
                 alt.Tooltip('sigma', format='r', title='standard deviation')]
    ).interactive()

st.write("""
Efficient frontier:
""")
st.altair_chart(ef_chart, use_container_width=True)

st.write("""
## References

Procedure of calculating the MVP and efficient frontier for the portfolio of more than 2 stocks can be found in:
Jajuga, K. (2015). _Inwestycje. Instrumenty finansowe, aktywa niefinansowe, ryzyko niefinansowe, inżynieria finansowa._
p. 215-221. Wydawnictwo Naukowe PWN.

Weights (_w_) of stocks in MVP can be calculated using the following formula:
""")
st.latex(r'''
w = C^{-1} \cdot I
''')
st.write("""
where:
""")
st.latex(r'''
C =
\begin{bmatrix}
   2 \cdot \sigma_1^2 & 2 \cdot \sigma_1 \cdot \sigma_2 \cdot \rho_{1,2} & \dots &
   2 \cdot \sigma_1 \cdot \sigma_n \cdot \rho_{1,n} & 1 \\
   2 \cdot \sigma_2 \cdot \sigma_1 \cdot \rho_{2,1} & 2 \cdot \sigma_2^2 & \dots &
   2 \cdot \sigma_2 \cdot \sigma_n \cdot \rho_{1,n} & 1 \\
   \vdots & \vdots & \ddots & \vdots & \vdots \\
   2 \cdot \sigma_n \cdot \sigma_1 \cdot \rho_{n,1} & 2 \cdot \sigma_n \cdot \sigma_2 \cdot \rho_{n,2} &
   \dots & 2 \cdot \sigma_n^2 & 1 \\
   1 & 1 & \dots & 1 & 0
\end{bmatrix}
''')
st.latex(r'''
I =
\begin{bmatrix}
   0 \\ 0 \\ \vdots \\ 0 \\ 1
\end{bmatrix}
''')
st.write("""
To determinate the efficient frontier, we can compute weights for a portfolio with minimal variance at a given return
rate. Such weights can be obtained using another matrix multiplication:
""")
st.latex(r'''
w = D^{-1} \cdot I_0
''')
st.write("""
where:
""")
st.latex(r'''
D =
\begin{bmatrix}
   2 \cdot \sigma_1^2 & 2 \cdot \sigma_1 \cdot \sigma_2 \cdot \rho_{1,2} & \dots &
   2 \cdot \sigma_1 \cdot \sigma_n \cdot \rho_{1,n} & 1 & E(r_1) \\
   2 \cdot \sigma_2 \cdot \sigma_1 \cdot \rho_{2,1} & 2 \cdot \sigma_2^2 & \dots &
   2 \cdot \sigma_2 \cdot \sigma_n \cdot \rho_{1,n} & 1 & E(r_2) \\
   \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
   2 \cdot \sigma_n \cdot \sigma_1 \cdot \rho_{n,1} & 2 \cdot \sigma_n \cdot \sigma_2 \cdot \rho_{n,2} &
   \dots & 2 \cdot \sigma_n^2 & 1 & E(r_n) \\
   1 & 1 & \dots & 1 & 0 & 0 \\
   E(r_1) & E(r_2) & \dots & E(r_n) & 0 & 0
\end{bmatrix}
''')
st.latex(r'''
I_0 =
\begin{bmatrix}
   0 \\ 0 \\ \vdots \\ 0 \\ 1 \\ E(r_p)
\end{bmatrix}
''')
st.write("""
With weight of stocks in the portfolio and their expected return rates, we can calculate the expected return of the
whole portfolio. It is the weighted average of expected return rates of portfolio assets:
""")
st.latex(r'''
E(r_p) = \sum_{i=1}^{n}(w_i \cdot E(r_i))
''')
st.write("""
To measure risk of a portfolio, we can use the standard deviations of return rates. To compute it for the portfolio, we
can use the following matrix formula:
""")
st.latex(r'''
\sigma_p = \sqrt{w^T \cdot \Sigma \cdot w}
''')
st.write("""
where: Σ is a covariance matrix.
""")
