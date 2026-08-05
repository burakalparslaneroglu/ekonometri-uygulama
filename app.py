"""Ekonometriye Giriş uygulamasının ortak arayüzü ve konu yönlendirmesi."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_config import APP_CONFIG
from core.session_utils import synchronize_active_topic
from core.ui_preferences import DEFAULT_TEXT_SCALE_LABEL, TEXT_SCALE_OPTIONS, normalize_text_scale, text_scale_css
from topics.konu01_ampirik_arastirma import render as render_konu01
from topics.konu02_veri_turleri_nedensellik import render as render_konu02
from topics.konu03_basit_regresyon import render as render_konu03
from topics.konu04_ols_cikti_fonksiyonel_bicimler import render as render_konu04
from topics.konu05_coklu_regresyon import render as render_konu05
from topics.konu06_ols_varsayimlari_yanlilik import render as render_konu06
from topics.konu07_tekli_hipotez_testleri import render as render_konu07
from topics.konu08_coklu_testler_buyuk_orneklem import render as render_konu08
from topics.konu09_fonksiyonel_bicimler import render as render_konu09


TOPIC_RENDERERS = {
    "konu01": render_konu01,
    "konu02": render_konu02,
    "konu03": render_konu03,
    "konu04": render_konu04,
    "konu05": render_konu05,
    "konu06": render_konu06,
    "konu07": render_konu07,
    "konu08": render_konu08,
    "konu09": render_konu09,
}


def load_styles(scale: float) -> None:
    """Yerel stil dosyasını uygulamaya ekler."""
    style_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        css = style_path.read_text(encoding="utf-8")
    except OSError as error:
        st.warning(f"Görsel stil dosyası yüklenemedi: {error}")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(text_scale_css(scale), unsafe_allow_html=True)


def main() -> None:
    """Ortak sayfa düzenini kurar ve seçili konuyu görüntüler."""
    st.set_page_config(
        page_title=f"{APP_CONFIG.course_name} | {APP_CONFIG.application_subtitle}",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    scale_label = st.session_state.get("text_scale_label", DEFAULT_TEXT_SCALE_LABEL)
    if scale_label not in TEXT_SCALE_OPTIONS:
        scale_label = DEFAULT_TEXT_SCALE_LABEL
    st.session_state["text_scale_label"] = scale_label
    st.session_state["text_scale"] = normalize_text_scale(TEXT_SCALE_OPTIONS[scale_label])
    load_styles(st.session_state["text_scale"])

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
                "Konu 05 — Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu",
                "Konu 06 — EKK Varsayımları, Yansızlık ve Model Sorunları",
                "Konu 07 — Tek Katsayı İçin Hipotez Testleri",
                "Konu 08 — Birden Fazla Kısıtın Sınanması: F Testi ve Büyük Örneklem Mantığı",
                "Konu 09 — Ölçekleme, Logaritmik Modeller, Karesel Terimler ve Model Seçimi",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("#### Görünüm")
        selected_scale = st.selectbox(
            "Metin boyutu", tuple(TEXT_SCALE_OPTIONS), index=tuple(TEXT_SCALE_OPTIONS).index(scale_label), key="text_scale_label"
        )
        st.session_state["text_scale"] = normalize_text_scale(TEXT_SCALE_OPTIONS[selected_scale])
        st.divider()
        st.caption(APP_CONFIG.institution_name)

    st.markdown(f"<div class='app-kicker'>{APP_CONFIG.institution_name.upper()}</div>", unsafe_allow_html=True)
    st.title(APP_CONFIG.course_name)
    st.caption(APP_CONFIG.application_subtitle)

    topic_id = next(identifier for prefix, identifier in {
        "Konu 01": "konu01", "Konu 02": "konu02", "Konu 03": "konu03", "Konu 04": "konu04", "Konu 05": "konu05", "Konu 06": "konu06", "Konu 07": "konu07", "Konu 08": "konu08", "Konu 09": "konu09",
    }.items() if topic.startswith(prefix))
    synchronize_active_topic(st.session_state, topic_id)

    TOPIC_RENDERERS[topic_id]()


if __name__ == "__main__":
    main()
