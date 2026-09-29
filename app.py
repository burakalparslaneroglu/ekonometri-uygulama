"""İKT 305 Ekonometri I uygulamasının ortak arayüzü ve konu yönlendirmesi."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_config import APP_CONFIG
from core.codegen.base import LANGUAGES
from core.labs.registry import LABS
from core.session_utils import synchronize_active_topic
from core.topic_registry import list_topics
from core.ui_preferences import DEFAULT_TEXT_SCALE_LABEL, TEXT_SCALE_OPTIONS, normalize_text_scale, text_scale_css
from topics import (
    konu00_baslangic_arac_kutusu,
    konu01_ampirik_arastirma,
    konu02_veri_turleri_nedensellik,
    konu03_basit_regresyon,
    konu04_ols_cikti_fonksiyonel_bicimler,
    konu05_coklu_regresyon,
    konu06_ols_varsayimlari_yanlilik,
)
from topics.konu07_tekli_hipotez_testleri import render as render_konu07
from topics.konu08_coklu_testler_buyuk_orneklem import render as render_konu08
from topics.konu09_fonksiyonel_bicimler import render as render_konu09
from topics.konu10_kukla_degiskenler import render as render_konu10
from topics.konu11_etkilesimler_grup_farklari import render as render_konu11
from topics.konu12_heteroskedastisite import render as render_konu12
from topics.lab_ui import CODE_LANGUAGE_KEY
from topics.shared import keep_widget_state

# Uygulama, Sezgi ve Kendini sına sekmeli konular. (app.py Streamlit'in ana betiğidir: modül düzeyindeki çıplak
# metinleri "magic" ile sayfaya yazar; bu yüzden burada açıklamalar docstring değil yorum satırıdır.)
MIGRATED_PAGES = (
    konu00_baslangic_arac_kutusu,
    konu01_ampirik_arastirma,
    konu02_veri_turleri_nedensellik,
    konu03_basit_regresyon,
    konu04_ols_cikti_fonksiyonel_bicimler,
    konu05_coklu_regresyon,
    konu06_ols_varsayimlari_yanlilik,
)

TOPIC_RENDERERS = {
    "konu00": konu00_baslangic_arac_kutusu.render,
    "konu01": konu01_ampirik_arastirma.render,
    "konu02": konu02_veri_turleri_nedensellik.render,
    "konu03": konu03_basit_regresyon.render,
    "konu04": konu04_ols_cikti_fonksiyonel_bicimler.render,
    "konu05": konu05_coklu_regresyon.render,
    "konu06": konu06_ols_varsayimlari_yanlilik.render,
    "konu07": render_konu07,
    "konu08": render_konu08,
    "konu09": render_konu09,
    "konu10": render_konu10,
    "konu11": render_konu11,
    "konu12": render_konu12,
}

# Kenar çubuğundaki konu adı → konu anahtarı ("Konu 01 · Ekonometri ve Ampirik Araştırma" → "konu01").
TOPIC_LABELS = {topic.label: topic.key for topic in list_topics()}

# Çizilmediği çalıştırmalarda da korunan seçimler: kod dili, adım, spesifikasyon, deney ve kaydırıcılar.
WIDGET_KEYS = frozenset({CODE_LANGUAGE_KEY}).union(*(page.widget_keys() for page in MIGRATED_PAGES))


def load_styles(scale: float) -> None:
    """Yerel stil dosyasını ve metin ölçeğini uygulamaya ekler."""
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
    keep_widget_state(WIDGET_KEYS)
    scale_label = st.session_state.get("text_scale_label", DEFAULT_TEXT_SCALE_LABEL)
    if scale_label not in TEXT_SCALE_OPTIONS:
        scale_label = DEFAULT_TEXT_SCALE_LABEL
    st.session_state["text_scale_label"] = scale_label
    st.session_state["text_scale"] = normalize_text_scale(TEXT_SCALE_OPTIONS[scale_label])
    load_styles(st.session_state["text_scale"])

    with st.sidebar:
        st.markdown(f"### {APP_CONFIG.course_name}")
        st.caption(APP_CONFIG.application_subtitle)
        label = st.radio("Konu seçimi", options=list(TOPIC_LABELS), key="selected_topic", label_visibility="collapsed")
        topic_id = TOPIC_LABELS[label]
        if topic_id in LABS:
            st.divider()
            st.markdown("#### Kod dili")
            if st.session_state.get(CODE_LANGUAGE_KEY) not in LANGUAGES:
                st.session_state[CODE_LANGUAGE_KEY] = LANGUAGES[0]
            st.segmented_control(
                "Kod dili", options=LANGUAGES, key=CODE_LANGUAGE_KEY, label_visibility="collapsed", width="stretch",
                required=True,
            )
            st.caption("Uygulama ve Sezgi sekmelerindeki kodlar bu dilde gösterilir.")
        st.divider()
        st.markdown("#### Görünüm")
        selected_scale = st.selectbox("Metin boyutu", tuple(TEXT_SCALE_OPTIONS), key="text_scale_label")
        st.session_state["text_scale"] = normalize_text_scale(TEXT_SCALE_OPTIONS[selected_scale])
        st.divider()
        st.caption("Ders notları içerik, terminoloji ve konu sırası açısından bağlayıcı kaynaktır.")
        st.caption(APP_CONFIG.institution_name)

    st.markdown(f"<div class='app-kicker'>{APP_CONFIG.institution_name.upper()}</div>", unsafe_allow_html=True)
    st.title(APP_CONFIG.course_name)
    st.caption(APP_CONFIG.application_subtitle)

    synchronize_active_topic(st.session_state, topic_id)
    TOPIC_RENDERERS[topic_id]()


if __name__ == "__main__":
    main()
