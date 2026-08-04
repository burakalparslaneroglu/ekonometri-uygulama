"""Ekonometriye Giriş uygulamasının ortak arayüzü ve konu yönlendirmesi."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from topics.konu03_basit_regresyon import render as render_konu03


def load_styles() -> None:
    """Yerel stil dosyasını uygulamaya ekler."""
    style_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        css = style_path.read_text(encoding="utf-8")
    except OSError as error:
        st.warning(f"Görsel stil dosyası yüklenemedi: {error}")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main() -> None:
    """Ortak sayfa düzenini kurar ve seçili konuyu görüntüler."""
    st.set_page_config(
        page_title="Ekonometriye Giriş",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_styles()

    with st.sidebar:
        st.markdown("### Ekonometriye Giriş")
        st.caption("Etkileşimli ders uygulaması")
        topic = st.radio(
            "Konu seçimi",
            options=["Konu 03 — Basit Doğrusal Regresyon"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Yatay kesit verisi ile model kurma ve yorumlama.")

    st.markdown("<div class='app-kicker'>PAÜ • EKONOMETRİYE GİRİŞ</div>", unsafe_allow_html=True)
    st.title("Ekonometriye Giriş")

    if topic == "Konu 03 — Basit Doğrusal Regresyon":
        render_konu03()


if __name__ == "__main__":
    main()
