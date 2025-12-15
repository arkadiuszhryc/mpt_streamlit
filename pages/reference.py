import streamlit as st

st.set_page_config(
    page_title="Modern Portfolio Theory",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.write("""
## References

Procedure of calculating the MVP and efficient frontier for the portfolio of more than 2 stocks can be found in:
Jajuga, K. (2015). _Inwestycje. Instrumenty finansowe, aktywa niefinansowe, ryzyko niefinansowe, inżynieria finansowa._
p. 215-221. Wydawnictwo Naukowe PWN.

Weights (_w_) of stocks in MVP can be calculated using the following formula:
""")
st.latex(r"""
w = C^{-1} \cdot I
""")
st.write("""
where:
""")
st.latex(r"""
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
""")
st.latex(r"""
I =
\begin{bmatrix}
   0 \\ 0 \\ \vdots \\ 0 \\ 1
\end{bmatrix}
""")
st.write("""
Please note that for _n_ assets, _C_ is (_n+1_)x(_n+1_) matrix and _I_ is a vector of _n+1_ length. Resulting _w_ vector
has _n+1_ elements, from which only the first _n_ are the assets weights.

To determinate the efficient frontier, we can compute weights for a portfolio with minimal variance at a given return
rate. Such weights can be obtained using another matrix multiplication:
""")
st.latex(r"""
w = D^{-1} \cdot I_0
""")
st.write("""
where:
""")
st.latex(r"""
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
""")
st.latex(r"""
I_0 =
\begin{bmatrix}
   0 \\ 0 \\ \vdots \\ 0 \\ 1 \\ E(r_p)
\end{bmatrix}
""")
st.markdown("""
Similarly to the prior case, for _n_ assets matrix _D_ has size of (_n+2_)x(_n+2_) matrix and _I0_ is a vector with
_n+2_ elements. Vector _w_ in consequence has _n+2_ length, but only the first _n_ are the assets weights.

With weight of stocks in the portfolio and their expected return rates, we can calculate the expected return of the
whole portfolio. It is the weighted average of expected return rates of portfolio assets:
""")
st.latex(r"""
E(r_p) = \sum_{i=1}^{n}(w_i \cdot E(r_i))
""")
st.write("""
To measure risk of a portfolio, we can use the standard deviations of return rates. To compute it for the portfolio, we
can use the following matrix formula:
""")
st.latex(r"""
\sigma_p = \sqrt{w^T \cdot \Sigma \cdot w}
""")
st.write("""
where: Σ is a covariance matrix.
""")
