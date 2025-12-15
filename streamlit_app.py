import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/mpt_calculation.py", title="Calculation"),
        st.Page("pages/reference.py", title="Reference"),
    ]
)
with st.sidebar:
    st.markdown("""**Licence and source code**
Application is available under MIT license. The source code can be accessed on
[GitHub](https://github.com/arkadiuszhryc/mpt_streamlit).""")
pg.run()
