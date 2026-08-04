from __future__ import annotations

import streamlit as st
import wooldridge as woo


st.set_page_config(
    page_title="Ekonometriye Giriş",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ekonometriye Giriş")
st.subheader("Streamlit altyapı testi")

dataset_name = st.selectbox(
    "Veri seti",
    options=["wage1", "hprice1"],
)

df = woo.data(dataset_name)

st.success(
    f"{dataset_name.upper()} veri seti yüklendi: "
    f"{df.shape[0]} gözlem, {df.shape[1]} değişken."
)

st.dataframe(df.head(10), use_container_width=True)