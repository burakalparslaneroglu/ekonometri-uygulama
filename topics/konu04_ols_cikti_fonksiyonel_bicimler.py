"""Konu 04: EKK uyumu, ölçü birimleri ve temel fonksiyonel biçimler."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.ui_components import render_question_actions

from core.data_registry import get_dataset_metadata, konu04_model_pairs, load_dataset, variable_metadata
from core.konu04_questions import generate_konu04_question
from core.model_utils import (
    FunctionalForm, fit_simple_ols, format_number, functional_form_interpretation, observation_decomposition,
    observation_result, r_squared_from_error_sum, r_squared_from_model_sum, rescaled_coefficients,
    sum_of_squares, transform_functional_form, validate_sum_of_squares, validate_unit_scale,
)
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state


FORMS: tuple[FunctionalForm, ...] = ("düzey-düzey", "log-düzey", "düzey-log", "log-log")
SCALE_OPTIONS = ("0.001", "0.01", "0.1", "1", "10", "100", "1000", "Özel çarpan")


@st.cache_data(show_spinner=False)
def _load_dataset_cached(dataset_key: str) -> pd.DataFrame:
    """Wooldridge verisini oturumlar arasında güvenle önbelleğe alır."""
    return load_dataset(dataset_key)


def _model_figure(result, selected_position: int, x_label: str, y_label: str) -> go.Figure:
    """Saçılımı, tahmin doğrusunu ve seçilen artık uzaklığını üretir."""
    data = pd.DataFrame({"x": result.explanatory_values, "y": result.observed_values}).sort_values("x")
    figure = px.scatter(data, x="x", y="y", labels={"x": x_label, "y": y_label}, color_discrete_sequence=["#107C89"])
    figure.add_scatter(x=data["x"], y=result.intercept + result.slope * data["x"], mode="lines", name="EKK tahmin doğrusu", line={"color": "#B3392F", "width": 3})
    selected = observation_result(result, selected_position)
    figure.add_scatter(x=[selected["x"], selected["x"]], y=[selected["predicted"], selected["observed"]], mode="lines", name="Seçili gözlemin artığı", line={"color": "#2F9E6B", "width": 3, "dash": "dot"})
    figure.update_layout(template="plotly_white", height=410, margin={"l": 10, "r": 10, "t": 35, "b": 10}, legend_title_text="")
    return figure


def _scale_selector(variable_name: str, key_prefix: str) -> float:
    """Hazır ya da kullanıcı tanımlı pozitif ölçü çarpanını döndürür."""
    choice = st.selectbox(f"{variable_name} için çarpan", SCALE_OPTIONS, index=3, key=f"{key_prefix}_choice")
    if choice == "Özel çarpan":
        value = st.number_input(
            f"{variable_name} için özel çarpan", min_value=1e-6, max_value=1e6, value=1.0,
            format="%.6g", key=f"{key_prefix}_custom",
        )
    else:
        value = float(choice)
    return validate_unit_scale(float(value), label=f"{variable_name} çarpanı")


def _render_questions(model_id: str, result, sums, decomposition, form: FunctionalForm, y_label: str, x_label: str) -> None:
    """Konu 04'e ait cevap-gizli deterministik soru panelini gösterir."""
    topic_id = "konu04"
    synchronize_question_state(st.session_state, model_id, topic_id=topic_id)
    index_key, _, answer_key = question_state_keys(topic_id)
    index = int(st.session_state.get(index_key, 0))
    question = generate_konu04_question(model_id, index, result, sums, decomposition, form, y_label, x_label)
    st.subheader("Kendini dene")
    st.markdown(f"**Soru {index + 1}:** {question.prompt}")
    render_question_actions(topic_id=topic_id)
    if st.session_state.get(answer_key, False):
        st.success(f"**Çözüm:** {question.answer}")
    else:
        st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")


def render() -> None:
    """Konu 04 öğretim bloklarını ders notundaki sırayla görüntüler."""
    st.markdown("<span class='topic-badge'>KONU 04</span>", unsafe_allow_html=True)
    st.header("EKK Tahminini Değerlendirme: Uyum, Ölçü Birimleri ve Temel Fonksiyonel Biçimler")
    st.markdown("<div class='lesson-note'><strong>Geçiş:</strong> Konu 03'te tek bir gözlemde gerçekleşen değer, tahmin edilen değer ve artığı gördünüz. Burada bu sapmaları örneklem genelinde kareler toplamlarıyla değerlendiriyoruz.</div>", unsafe_allow_html=True)

    dataset_key = st.selectbox("Veri seti", ("wage1", "hprice1"), key="konu04_dataset", format_func=lambda key: get_dataset_metadata(key).title)
    metadata = get_dataset_metadata(dataset_key)
    dependent, explanatory = konu04_model_pairs(dataset_key)[0]
    y_info, x_info = variable_metadata(dataset_key, dependent), variable_metadata(dataset_key, explanatory)
    form = st.selectbox("Fonksiyonel biçim", FORMS, key="konu04_form")
    try:
        frame = _load_dataset_cached(dataset_key)
        transformed = transform_functional_form(frame, dependent, explanatory, form)
        result = fit_simple_ols(transformed.data, transformed.dependent_name, transformed.explanatory_name)
        sums = sum_of_squares(result)
    except (RuntimeError, ValueError) as error:
        st.error(f"Model kurulamadı: {error}")
        return

    with st.expander("Veri seti kaynağı, değişkenler ve örneklem", expanded=True):
        st.markdown(f"**Kaynak:** {metadata.source}")
        st.write(metadata.description)
        st.dataframe(pd.DataFrame([
            {"Değişken": y_info.name, "Açıklama": y_info.description, "Ölçü birimi": y_info.unit},
            {"Değişken": x_info.name, "Açıklama": x_info.description, "Ölçü birimi": x_info.unit},
        ]), hide_index=True, width="stretch")
        st.caption(f"Kullanılan örneklem: {result.nobs} geçerli gözlem. Log dönüşümünde hiçbir gözlem sessizce çıkarılmaz.")

    cards = st.columns(3)
    cards[0].metric("Gözlem sayısı", result.nobs)
    cards[1].metric("Eğim katsayısı", format_number(result.slope))
    cards[2].metric("R-kare", format_number(result.r_squared))
    st.subheader("Regresyon çıktısını okuma")
    st.latex(rf"\widehat{{{transformed.dependent_name}}}={format_number(result.intercept)} {('+' if result.slope >= 0 else '−')} {format_number(abs(result.slope))}\,{transformed.explanatory_name}")
    st.dataframe(pd.DataFrame({"Alan": ["Bağımlı değişken", "Açıklayıcı değişken", "Tahmin edilen sabit", "Tahmin edilen eğim", "R-kare"], "Değer": [transformed.dependent_name, transformed.explanatory_name, format_number(result.intercept), format_number(result.slope), format_number(result.r_squared)]}), hide_index=True, width="stretch")
    st.caption("Bu çıktı yalnızca örneklemdeki doğrusal ilişkiyi ve uyumu özetler; nedensellik hakkında karar vermez.")

    st.subheader("Bir gözlemdeki sapmanın ayrıştırılması")
    position = st.slider("Seçili gözlem", 1, result.nobs, 1, key="konu04_observation") - 1
    decomposition = observation_decomposition(result, position)
    st.latex(rf"Y_i-\bar Y=(\widehat Y_i-\bar Y)+\widehat u_i\\ {format_number(decomposition.total_deviation)}={format_number(decomposition.model_deviation)}+{format_number(decomposition.residual_deviation)}")
    st.dataframe(pd.DataFrame({"Bileşen": ["Yᵢ", "Ȳ", "Ŷᵢ", "ûᵢ", "Toplam sapma", "Model sapması", "Artık sapması"], "Değer": [format_number(value) for value in [decomposition.observed, decomposition.mean_observed, decomposition.fitted, decomposition.residual, decomposition.total_deviation, decomposition.model_deviation, decomposition.residual_deviation]]}), hide_index=True, width="stretch")
    st.plotly_chart(_model_figure(result, position, f"{x_info.label} ({x_info.unit})", f"{y_info.label} ({y_info.unit})"), width="stretch")

    st.subheader("Kareler toplamlarının ayrıştırılması")
    square_cards = st.columns(3)
    square_cards[0].metric("TKT — Toplam Kareler Toplamı", format_number(sums.total))
    square_cards[1].metric("MKT — Model Kareler Toplamı", format_number(sums.model))
    square_cards[2].metric("HKT — Hata Kareler Toplamı", format_number(sums.error))
    st.latex(r"\mathrm{TKT}=\sum(Y_i-\bar Y)^2,\quad \mathrm{MKT}=\sum(\widehat Y_i-\bar Y)^2,\quad \mathrm{HKT}=\sum\widehat u_i^2")
    st.info(f"Sayısal kontrol: TKT = MKT + HKT → {format_number(sums.total)} = {format_number(sums.model + sums.error)}. {'Eşitlik tolerans içinde sağlanır.' if validate_sum_of_squares(sums) else 'Eşitlik beklenen tolerans içinde sağlanmıyor.'}")
    st.plotly_chart(px.bar(x=["TKT", "MKT", "HKT"], y=[sums.total, sums.model, sums.error], labels={"x": "Kareler toplamı", "y": "Kareli büyüklük"}, color_discrete_sequence=["#107C89"]).update_layout(template="plotly_white", height=300, margin={"l": 10, "r": 10, "t": 20, "b": 10}), width="stretch")

    r2_model, r2_error = r_squared_from_model_sum(sums), r_squared_from_error_sum(sums)
    st.subheader("Belirleme katsayısı (R-kare) ve uyum")
    st.latex(rf"R^2=\frac{{MKT}}{{TKT}}={format_number(r2_model)}=1-\frac{{HKT}}{{TKT}}={format_number(r2_error)}")
    st.info(f"Örneklemde {y_info.label} değişkenindeki değişimin yaklaşık %{format_number(100*r2_model)}'i, {x_info.label} ile kurulan {form} model tarafından izlenmektedir.")
    st.warning("R-kare nedensellik kanıtı değildir; yüksek olması modelin mutlaka doğru olduğunu, düşük olması modelin yararsız olduğunu, tahminlerin birebir doğru olduğunu veya iktisadi önemi tek başına göstermez.")

    st.subheader("Ölçü birimi dönüşümü")
    y_scale = _scale_selector("Y", "konu04_y_scale")
    x_scale = _scale_selector("X", "konu04_x_scale")
    new_intercept, new_slope = rescaled_coefficients(result, dependent_scale=y_scale, explanatory_scale=x_scale)
    st.dataframe(pd.DataFrame({"Büyüklük": ["Sabit terim", "Eğim katsayısı", "R-kare"], "Önce": [format_number(result.intercept), format_number(result.slope), format_number(result.r_squared)], "Sonra": [format_number(new_intercept), format_number(new_slope), format_number(result.r_squared)]}), hide_index=True, width="stretch")
    st.latex(r"Y^*=aY,\quad X^*=bX")
    st.latex(r"\widehat{\beta}_0^*=a\widehat{\beta}_0,\quad \widehat{\beta}_1^*=\frac{a}{b}\widehat{\beta}_1,\quad R^{2*}=R^2")
    st.caption("Pozitif çarpanlar yalnızca ölçü birimini değiştirir; negatif çarpan ise işaret dönüşümüdür ve kabul edilmez. Bu doğrusal ölçü dönüşümünde R-kare değişmez.")

    st.subheader("Fonksiyonel biçim ve katsayı yorumu")
    st.markdown(functional_form_interpretation(form, result.slope))
    st.caption("Yaklaşık yüzde yorumları küçük değişimler içindir. Bağımlı değişken dönüşmüş modellerin R-kareleri düzeydeki bağımlı değişkenli modellerle mekanik biçimde karşılaştırılmaz; model seçimi en yüksek R-kareye indirgenmez.")
    model_id = f"konu04:{dataset_key}:{form}:{position}:{y_scale}:{x_scale}"
    _render_questions(model_id, result, sums, decomposition, form, y_info.label, x_info.label)
