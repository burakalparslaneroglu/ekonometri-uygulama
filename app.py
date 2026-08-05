"""Ekonometriye Giriş uygulamasının ortak arayüzü ve konu yönlendirmesi."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_config import APP_CONFIG
from core.session_utils import synchronize_active_topic
from topics.konu01_ampirik_arastirma import render as render_konu01
from topics.konu02_veri_turleri_nedensellik import render as render_konu02
from topics.konu03_basit_regresyon import render as render_konu03
from topics.konu04_ols_cikti_fonksiyonel_bicimler import render as render_konu04


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
        page_title=f"{APP_CONFIG.course_name} | {APP_CONFIG.application_subtitle}",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_styles()

    with st.sidebar:
        st.markdown(f"### {APP_CONFIG.course_name}")
        st.caption(APP_CONFIG.application_subtitle)
        topic = st.radio(
            "Konu seçimi",
            options=[
                "Konu 01 — Ekonometri ve Ampirik Araştırma",
                "Konu 02 — Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus",
                "Konu 03 — Basit Doğrusal Regresyon",
                "Konu 04 — EKK Tahminini Değerlendirme: Uyum, Ölçü Birimleri ve Temel Fonksiyonel Biçimler",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(APP_CONFIG.institution_name)

    st.markdown(f"<div class='app-kicker'>{APP_CONFIG.institution_name.upper()}</div>", unsafe_allow_html=True)
    st.title(APP_CONFIG.course_name)
    st.caption(APP_CONFIG.application_subtitle)

    topic_id = (
        "konu01" if topic.startswith("Konu 01") else "konu02" if topic.startswith("Konu 02")
        else "konu03" if topic.startswith("Konu 03") else "konu04"
    )
    synchronize_active_topic(st.session_state, topic_id)

    if topic_id == "konu01":
        render_konu01()
    elif topic_id == "konu02":
        render_konu02()
    elif topic_id == "konu03":
        render_konu03()
    else:
        render_konu04()


if __name__ == "__main__":
    main()
