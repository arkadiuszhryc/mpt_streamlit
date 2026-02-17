import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Modern Portfolio Theory",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.write("""
# Modern Portfolio Theory

_A web app by Arkadiusz Hryc_

This script allows to find the minimum variance portfolio (MVP) based for stocks available via [Stooq](https://stooq.com/) and to
plot the efficient frontier.

For detailed information about the calculation procedure of MVP and efficient frontier, please see the
[References](/reference) page.

## Calculate MVP and efficient frontier
""")

tickers = st.text_input(
    label="Please enter tickers from Stooq (separated by comma and space):",
    value="pko, pzu, pkn",
)
ticker_list = tickers.split(", ")


def get_close(ticker):
    data_url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    stooq_data = pd.read_csv(data_url)
    if len(stooq_data) == 0:
        return None
    else:
        stooq_data["Date"] = pd.to_datetime(stooq_data["Date"], format="%Y-%m-%d")
        stooq_data.rename(columns={"Close": ticker.upper()}, inplace=True)
        stooq_data = stooq_data.drop(["Open", "High", "Low", "Volume"], axis=1)
        return stooq_data


with st.spinner("Fetching data, please wait..."):
    for i, j in enumerate(ticker_list):
        if i == 0:
            data = get_close(j)
            if data is None:
                st.warning(
                    f'Ticker "{j}" not found. Ignoring this ticker for now. Please check spelling and try again.'
                )
        elif data is None:
            data = get_close(j)
        else:
            new_data = get_close(j)
            if new_data is None:
                st.warning(
                    f'Ticker "{j}" not found. Ignoring this ticker for now. Please check spelling and try again.'
                )
            else:
                data = data.merge(new_data, how="outer", on="Date")

    data = data.sort_values(by="Date")
    data.iloc[:, 1:] = data.iloc[:, 1:].astype("float")
    data = data.set_index("Date")
    data = data.ffill()

with st.spinner("Computing..."):
    log_ret = np.log(data) - np.log(data.shift(1))
    log_ret = log_ret.tail(n=252)

    stock_data = pd.DataFrame(
        data=None, index=log_ret.columns, columns=["mu", "sigma"], dtype="float64"
    )

    for i in log_ret.columns:
        stock_data.loc[i, "mu"] = np.mean(log_ret[i])
        stock_data.loc[i, "sigma"] = np.std(log_ret[i], ddof=1)

    stock_data_cov = log_ret.cov()

    matrix_C = stock_data_cov.merge(
        pd.DataFrame(
            data=np.ones(len(log_ret.columns)),
            index=log_ret.columns,
            columns=["ones"],
            dtype="float64",
        ),
        left_index=True,
        right_index=True,
    )
    matrix_C = pd.concat(
        [
            matrix_C,
            pd.DataFrame(
                data=np.append(np.ones(len(log_ret.columns)), 0).reshape(
                    1, len(log_ret.columns) + 1
                ),
                index=["ones"],
                columns=matrix_C.columns,
                dtype="float64",
            ),
        ]
    )

    matrix_I = np.append(np.zeros(len(log_ret.columns)), 1).reshape(
        len(log_ret.columns) + 1, 1
    )

    w = np.matmul(np.linalg.inv(matrix_C), matrix_I.reshape(len(matrix_C.columns), 1))

    stock_data = stock_data.merge(
        pd.DataFrame(
            data=w[:-1], index=log_ret.columns, columns=["w"], dtype="float64"
        ),
        left_index=True,
        right_index=True,
    )

    stock_data["mu_annual"] = stock_data["mu"] * 252
    stock_data["sigma_annual"] = stock_data["sigma"] * math.sqrt(252)

    mu_min = np.dot(stock_data["w"].to_numpy(), stock_data["mu"].to_numpy()) * 252

    sigma_min = math.sqrt(
        np.matmul(
            np.matmul(
                stock_data["w"].to_numpy().reshape(1, len(log_ret.columns)),
                stock_data_cov.to_numpy(),
            ),
            stock_data["w"].to_numpy().reshape(len(log_ret.columns), 1),
        ).item()
    ) * math.sqrt(252)

    min_portfolio = pd.DataFrame(
        data=[[1, mu_min, sigma_min]],
        columns=["weight", "return (annual)", "standard deviation (annual)"],
        index=["portfolio"],
        dtype="float64",
    )

    mvp_to_print = stock_data.iloc[:, 2:].copy()
    mvp_to_print.columns = min_portfolio.columns
    mvp_to_print = pd.concat([mvp_to_print, min_portfolio])

st.write("""
Minimal variance portfolio:
""")
st.write(
    mvp_to_print.style.format(
        {
            "weight": "{:.2%}",
            "return (annual)": "{:.2%}",
            "standard deviation (annual)": "{:.4f}",
        }
    )
)

with st.spinner("Computing..."):
    matrix_D = matrix_C.merge(
        pd.DataFrame(
            data=np.append(stock_data["mu"].to_numpy(), 0),
            index=matrix_C.columns,
            columns=["r"],
            dtype="float64",
        ),
        left_index=True,
        right_index=True,
    )
    matrix_D = pd.concat(
        [
            matrix_D,
            pd.DataFrame(
                data=np.append(stock_data["mu"].to_numpy(), np.zeros(2)).reshape(
                    1, len(matrix_C.columns) + 1
                ),
                index=["r"],
                columns=matrix_D.columns,
                dtype="float64",
            ),
        ]
    )

    r_sigma = pd.DataFrame(
        data=np.linspace(np.min(stock_data["mu"]), np.max(stock_data["mu"]), num=401),
        columns=["r"],
    )

    rows = np.empty((1, len(log_ret.columns) + 1))
    for i in r_sigma["r"]:
        matrix_I0 = np.append(matrix_I, i).reshape(len(log_ret.columns) + 2, 1)

        wages = np.matmul(
            np.linalg.inv(matrix_D), matrix_I0.reshape(len(matrix_D.columns), 1)
        )[:-2]
        rows = np.append(
            rows, np.append(wages, i).reshape(1, len(log_ret.columns) + 1), axis=0
        )

    r_sigma = r_sigma.merge(
        pd.DataFrame(data=rows[1:], columns=np.append(log_ret.columns, "r")), on="r"
    )

    rows2 = np.empty((1, 2))
    for i in r_sigma.index:
        sigma = math.sqrt(
            np.matmul(
                np.matmul(
                    r_sigma.iloc[i, 1:].to_numpy().reshape(1, len(log_ret.columns)),
                    stock_data_cov.to_numpy(),
                ),
                r_sigma.iloc[i, 1:].to_numpy().reshape(len(log_ret.columns), 1),
            ).item()
        )
        rows2 = np.append(
            rows2, np.append(sigma, r_sigma.iloc[i, 0]).reshape(1, 2), axis=0
        )

    r_sigma = r_sigma.merge(
        pd.DataFrame(data=rows2[1:], columns=["sigma", "r"]), on="r"
    )

    r_sigma["r"] = r_sigma["r"] * 252
    r_sigma["sigma"] = r_sigma["sigma"] * math.sqrt(252)

    ef_chart = (
        alt.Chart(r_sigma)
        .mark_circle()
        .encode(
            alt.X(
                "sigma",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format="%", title="standard deviation (annual)"),
            ),
            alt.Y(
                "r",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format="%", title="expected return rate (annual)"),
            ),
            tooltip=[
                alt.Tooltip("r", format="p", title="return rate"),
                alt.Tooltip("sigma", format="r", title="standard deviation"),
            ],
        )
        .interactive()
    )


st.write("""
Efficient frontier:
""")

st.altair_chart(ef_chart, width="stretch")
