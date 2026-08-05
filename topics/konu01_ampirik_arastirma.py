"""Konu 01: Ekonometri ve Ampirik Araştırmanın Mantığı."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.data_registry import get_dataset_metadata, load_dataset, variable_metadata
from core.model_utils import descriptive_statistics
from core.research_question_utils import ResearchQuestion, format_research_question, is_measurable_question
from core.scenario_registry import generate_konu01_question
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state


TOPIC_ID = "konu01"


@st.cache_data(show_spinner=False)
def _load_wage1() -> pd.DataFrame:
    """WAGE1 verisini önbellekli biçimde yükler."""
    return load_dataset("wage1")


def _render_research_question_builder() -> None:
    """Seçili bileşenlerden ölçülebilir araştırma sorusu kurar."""
    st.subheader("Genel ilgiden ölçülebilir araştırma sorusuna")
    st.write("Genel ilgi: Eğitim ile kazanç arasında nasıl bir bağlantı vardır?")
    columns = st.columns(2)
    observation_unit = columns[0].selectbox("İncelenen birim", ["Ücretli çalışanlar"])
    outcome = columns[1].selectbox("Sonuç değişkeni", ["Saatlik ücret"])
    explanatory = columns[0].selectbox("Temel açıklayıcı değişken", ["Tamamlanan eğitim yılı"])
    population_scope = columns[1].selectbox("Anakütle, yer ve dönem", ["İncelenen çalışan örneklemi"])
    purpose = st.selectbox("Araştırma amacı", ["İlişki", "Betimleme", "Tahmin", "Nedensel etki"])
    question = ResearchQuestion(observation_unit, outcome, explanatory, population_scope, purpose)
    if is_measurable_question(question):
        st.success(f"**Kurulan soru:** {format_research_question(question)}")
    if purpose == "Nedensel etki":
        st.info("Nedensel etki, ilişki sorusundan daha güçlü bir amaçtır; bu amaç için ek araştırma tasarımı gerekir.")


def _render_model_layers() -> None:
    """Ekonomik, matematiksel ve ekonometrik model ayrımını gösterir."""
    st.subheader("Ekonomik, matematiksel ve ekonometrik model")
    tabs = st.tabs(["Ekonomik model", "Matematiksel model", "Ekonometrik model"])
    with tabs[0]:
        st.write("Eğitim bilgi ve becerileri geliştirerek üretkenlikle, dolayısıyla ücretle ilişkili olabilir.")
    with tabs[1]:
        st.latex(r"Y = \beta_0 + \beta_1 X")
        st.caption("Bu doğrusal biçim, ilişkinin sadeleştirilmiş matematiksel gösterimidir.")
    with tabs[2]:
        st.latex(r"Y_i = \beta_0 + \beta_1 X_i + u_i")
        st.caption("Hata terimi, modelde ayrı değişken olarak yer almayan gözlenmeyen unsurları temsil eder.")


def _render_research_stages() -> None:
    """Ampirik araştırmanın ders notundaki sekiz aşamasını listeler."""
    st.subheader("Ampirik araştırmanın aşamaları")
    stages = (
        "Araştırma sorusunu ve amacı belirleme",
        "Ekonomik mekanizmayı kurma",
        "Anakütleyi, veriyi ve değişkenleri tanımlama",
        "Ekonometrik modeli yazma",
        "Modeli tahmin etme",
        "Belirsizliği değerlendirme",
        "Modeli ve sonucu sorgulama",
        "İktisadi yorum ve raporlama",
    )
    for position, stage in enumerate(stages, start=1):
        st.markdown(f"{position}. **{stage}**")
    st.caption("Bu aşamalar geri bildirimli bir süreçtir; veri ve ilk bulgular önceki kararların yeniden gözden geçirilmesini gerektirebilir.")


def _render_purpose_cards() -> None:
    """Dört ampirik amacı nottaki yorum sınırlarıyla gösterir."""
    st.subheader("Ampirik soruların dört temel amacı")
    purposes = (
        ("Betimleme", "Örneklemin veya anakütlenin düzeyini ve dağılımını özetler."),
        ("İlişki", "Değişkenlerin örneklemde birlikte nasıl hareket ettiğini inceler."),
        ("Tahmin", "Bilinen bilgilerle yeni veya gelecekteki sonucu öngörmeyi amaçlar."),
        ("Nedensel etki", "Bir müdahale veya değişikliğin sonuçta oluşturduğu farkı sorar."),
    )
    for column, (title, description) in zip(st.columns(4), purposes, strict=True):
        column.markdown(f"**{title}**\n\n{description}")


def _render_wage1_explorer() -> None:
    """WAGE1 için yalnızca ders kapsamındaki betimsel veri gezginini gösterir."""
    st.subheader("WAGE1 veri gezgini")
    metadata = get_dataset_metadata("wage1")
    try:
        frame = _load_wage1()
    except RuntimeError as error:
        st.error(str(error))
        return
    st.markdown(f"**Kaynak:** {metadata.source}")
    st.write(metadata.description)
    st.info(f"**Gözlem birimi:** {metadata.observation_unit}. Her satır bir çalışanı temsil eder.")
    variables = ("wage", "educ", "exper", "tenure")
    variable_table = pd.DataFrame(
        [
            {"Değişken": variable, "Açıklama": variable_metadata("wage1", variable).description, "Ölçü birimi": variable_metadata("wage1", variable).unit}
            for variable in variables
        ]
    )
    st.dataframe(variable_table, hide_index=True, width="stretch")
    st.metric("Gözlem sayısı", len(frame))
    summary = descriptive_statistics(frame, variables).round(3)
    st.dataframe(summary, width="stretch")
    st.caption("Tablo, seçili değişkenler için gözlem sayısı, ortalama, standart sapma, en küçük ve en büyük değeri verir.")
    figure = px.scatter(
        frame,
        x="educ",
        y="wage",
        labels={"educ": "Eğitim (yıl)", "wage": "Saatlik ücret (ABD doları/saat)"},
        color_discrete_sequence=["#107C89"],
        opacity=0.75,
    )
    figure.update_layout(template="plotly_white", height=410, margin={"l": 10, "r": 10, "t": 25, "b": 10})
    figure.update_traces(marker={"size": 8})
    st.subheader("Eğitim ve saatlik ücret saçılım grafiği")
    st.plotly_chart(figure, width="stretch")
    st.caption("Noktalar çalışanları gösterir. Grafik örneklemdeki ilişkiyi betimler; tek başına nedensellik göstermez.")


def _render_questions() -> None:
    """Konu 01'e ait ortak altyapılı deterministik soru panelini gösterir."""
    model_id = "konu01:wage1:ampirik_arastirma"
    synchronize_question_state(st.session_state, model_id, topic_id=TOPIC_ID)
    index_key, _, answer_key = question_state_keys(TOPIC_ID)
    index = int(st.session_state.get(index_key, 0))
    question = generate_konu01_question(model_id, index)
    st.subheader("Deterministik uygulama soruları")
    st.markdown(f"**Soru {index + 1}:** {question.prompt}")
    st.text_area("Yanıtınız", key=f"konu01_student_response_{index}", help="Yanıtınızı yazdıktan sonra çözümle karşılaştırabilirsiniz.")
    first, second = st.columns(2)
    if first.button("Cevabı göster", type="primary", width="stretch", key="konu01_show_answer"):
        reveal_answer(st.session_state, topic_id=TOPIC_ID)
    if second.button("Yeni soru", width="stretch", key="konu01_next_question"):
        next_question(st.session_state, topic_id=TOPIC_ID)
        st.rerun()
    if st.session_state.get(answer_key, False):
        st.success(f"**Çözüm:** {question.answer}")
    else:
        st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")


def render() -> None:
    """Konu 01'in ders notuna bağlı etkileşimli görünümünü oluşturur."""
    st.markdown("<span class='topic-badge'>KONU 01</span>", unsafe_allow_html=True)
    st.header("Ekonometri ve Ampirik Araştırmanın Mantığı")
    st.markdown(
        "<div class='lesson-note'><strong>Amaç:</strong> Ekonomik bir ilgiyi ölçülebilir araştırma sorusuna dönüştürmek, model düzeylerini ayırmak ve ampirik yorumun sınırlarını tanımak.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Ekonometri nedir?")
    st.write("Ekonometri; ekonomik teori, ekonomik veri ve istatistiksel yöntemleri bir araya getirerek ekonomik ilişkileri sayısal olarak inceler.")
    theory, data, method = st.columns(3)
    theory.markdown("**Ekonomik teori**\n\nDeğişkenlerin neden ilişkili olabileceğini açıklar.")
    data.markdown("**Ekonomik veri**\n\nTeorik ilişkiyi gözlemleyebileceğimiz ölçümleri sağlar.")
    method.markdown("**İstatistiksel yöntem**\n\nVerideki düzenli ilişkiyi incelemeye yardım eder.")
    _render_research_question_builder()
    st.subheader("Araştırma sorusunun bileşenleri")
    st.markdown("Gözlem birimi • Sonuç değişkeni • Temel açıklayıcı değişken • Anakütle, yer ve dönem • Araştırma amacı")
    _render_model_layers()
    st.subheader("Hata teriminin kavramsal rolü")
    st.write("Hata terimi, sonucu etkilediği hâlde modelde ayrı açıklayıcı değişken olarak gösterilmeyen unsurların net etkisini temsil eder; doğrudan gözlenmez ve yalnızca hesaplama hatası değildir.")
    _render_research_stages()
    _render_purpose_cards()
    _render_wage1_explorer()
    st.divider()
    _render_questions()
