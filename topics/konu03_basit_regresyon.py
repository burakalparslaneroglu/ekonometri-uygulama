"""Konu 03: Basit Doğrusal Regresyon Streamlit pilotu."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data_registry import (
    allowed_explanatory_variables,
    get_dataset_metadata,
    list_datasets,
    load_dataset,
    variable_metadata,
)
from core.model_utils import descriptive_statistics, fit_simple_ols, observation_result, predict_value
from core.question_engine import generate_question
from core.session_utils import (
    ANSWER_VISIBLE_KEY,
    QUESTION_INDEX_KEY,
    next_question,
    reveal_answer,
    synchronize_question_state,
)


@st.cache_data(show_spinner=False)
def _load_dataset_cached(dataset_key: str) -> pd.DataFrame:
    """Wooldridge verisini Streamlit önbelleğiyle yükler."""
    return load_dataset(dataset_key)


def _format_variable(dataset_key: str, variable: str) -> str:
    """Seçim kutuları için değişken etiketini döndürür."""
    item = variable_metadata(dataset_key, variable)
    return f"{item.label} ({item.name})"


def _scatter_figure(result, x_label: str, y_label: str, selected_position: int) -> go.Figure:
    """Saçılım grafiğini, tahmin doğrusunu ve seçili gözlemin artığını üretir."""
    plot_data = pd.DataFrame({"x": result.explanatory_values, "y": result.observed_values}).sort_values("x")
    figure = px.scatter(
        plot_data,
        x="x",
        y="y",
        labels={"x": x_label, "y": y_label},
        color_discrete_sequence=["#107C89"],
        opacity=0.75,
    )
    figure.add_scatter(
        x=plot_data["x"],
        y=result.intercept + result.slope * plot_data["x"],
        mode="lines",
        name="EKK tahmin doğrusu",
        line={"color": "#B3392F", "width": 3},
    )
    selected = observation_result(result, selected_position)
    figure.add_scatter(
        x=[selected["x"], selected["x"]],
        y=[selected["predicted"], selected["observed"]],
        mode="lines",
        name="Seçili gözlemin dikey artık mesafesi",
        line={"color": "#2F9E6B", "width": 3, "dash": "dot"},
        hovertemplate="Seçili gözlemin artık mesafesi<br>X=%{x:.4f}<br>Y=%{y:.4f}<extra></extra>",
    )
    figure.add_scatter(
        x=[selected["x"]],
        y=[selected["observed"]],
        mode="markers",
        name=f"Seçili gözlem ({selected_position + 1})",
        marker={"color": "#07373D", "size": 13, "symbol": "diamond"},
        hovertemplate=(
            f"Seçili gözlem {selected_position + 1}<br>X=%{{x:.4f}}<br>"
            "Gerçekleşen Y=%{y:.4f}<extra></extra>"
        ),
    )
    figure.update_layout(
        legend_title_text="",
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        height=420,
        template="plotly_white",
    )
    figure.update_traces(marker={"size": 8}, selector={"mode": "markers"})
    return figure


def _render_prediction_panel(result, dataset_key: str) -> None:
    """Kullanıcının belirli bir X değeri için tahmin üretmesini sağlar."""
    x_info = variable_metadata(dataset_key, result.explanatory)
    y_info = variable_metadata(dataset_key, result.dependent)
    st.subheader("Belirli bir X değeri için tahmin")
    x_min, x_max = float(result.explanatory_values.min()), float(result.explanatory_values.max())
    x_value = st.number_input(
        f"{x_info.label} ({x_info.unit})",
        min_value=x_min,
        max_value=x_max,
        value=float(result.explanatory_values.median()),
        help="Yorumlanabilirlik için yalnızca örneklemde gözlenen X aralığındaki değerler kullanılabilir.",
    )
    predicted = predict_value(result, float(x_value))
    st.latex(
        rf"\widehat{{{result.dependent}}} = {result.intercept:.4f} {('+' if result.slope >= 0 else '-')} "
        rf"{abs(result.slope):.4f}\,({float(x_value):.4f}) = {predicted:.4f}"
    )
    st.info(f"**Tahmin edilen değer:** {y_info.label} = {predicted:.4f} {y_info.unit}.")
    st.caption(
        "Bu serbest X değeri için gerçekleşen Y bilinmediğinden artık hesaplanmaz. "
        "Artık yalnızca veri setindeki belirli bir gözlem için hesaplanabilir."
    )


def _render_observation_panel(result, dataset_key: str) -> tuple[int, dict[str, float | int]]:
    """Seçilen gözlem için tahmin, artık ve bağlamsal yorumu gösterir."""
    x_info = variable_metadata(dataset_key, result.explanatory)
    y_info = variable_metadata(dataset_key, result.dependent)
    st.subheader("Bir gözlem için tahmin edilen değer ve artık")
    observation_options = tuple(f"{position + 1}. gözlem" for position in range(result.nobs))
    selected_label = st.selectbox(
        "Gözlem numarası",
        options=observation_options,
        help="Gözlem numarası, modelde kullanılan geçerli gözlemlerin sırasını gösterir.",
    )
    selected_position = int(selected_label.split(".", maxsplit=1)[0]) - 1
    selected = observation_result(result, selected_position)
    values = pd.DataFrame(
        {
            "Büyüklük": [x_info.label, f"Gerçekleşen {y_info.label}", f"Tahmin edilen {y_info.label}", "Artık"],
            "Değer": [selected["x"], selected["observed"], selected["predicted"], selected["residual"]],
            "Birim": [x_info.unit, y_info.unit, y_info.unit, y_info.unit],
        }
    )
    st.dataframe(values.round(4), hide_index=True, width="stretch")
    st.latex(
        rf"\widehat{{{result.dependent}}}_{{{selected_position + 1}}} = {result.intercept:.4f} "
        rf"{('+' if result.slope >= 0 else '-')} {abs(result.slope):.4f}\,({selected['x']:.4f}) "
        rf"= {selected['predicted']:.4f}"
    )
    st.latex(
        rf"\widehat{{u}}_{{{selected_position + 1}}} = {selected['observed']:.4f} - "
        rf"{selected['predicted']:.4f} = {selected['residual']:.4f}"
    )
    if abs(selected["residual"]) < 1e-10:
        interpretation = "Artık sıfıra yakındır: gerçekleşen Y, tahmin edilen değerle yaklaşık aynıdır."
    elif selected["residual"] > 0:
        interpretation = "Pozitif artık: gerçekleşen Y, modelin tahmin ettiği değerin üzerindedir; model Y'yi eksik tahmin etmiştir."
    else:
        interpretation = "Negatif artık: gerçekleşen Y, modelin tahmin ettiği değerin altındadır; model Y'yi fazla tahmin etmiştir."
    st.info(interpretation)
    st.caption(
        f"Mutlak artık |û| = {abs(selected['residual']):.4f} {y_info.unit}; bu, bu gözleme özgü tahmin hatasının büyüklüğünü gösterir. "
        "Büyük bir artık tek başına veri giriş hatasının kanıtı değildir."
    )
    st.markdown(
        "EKK, gözlemlerin tahmin doğrusuna dikey uzaklıklarının **kareleri toplamını** en küçük yapan doğruyu seçer."
    )
    return selected_position, selected


def _render_questions(result, model_id: str, dataset_key: str, observation_position: int) -> None:
    """Deterministik soru panelini ve oturum durumunu görüntüler."""
    synchronize_question_state(st.session_state, model_id)
    index = int(st.session_state.get(QUESTION_INDEX_KEY, 0))
    y_info = variable_metadata(dataset_key, result.dependent)
    x_info = variable_metadata(dataset_key, result.explanatory)
    question = generate_question(
        result,
        model_id,
        index,
        dependent_label=y_info.label,
        explanatory_label=x_info.label,
        dependent_unit=y_info.unit,
        explanatory_unit=x_info.unit,
        observation_position=observation_position,
    )
    st.subheader("Kendini dene")
    st.markdown(f"**Soru {index + 1}:** {question.prompt}")
    first, second = st.columns(2)
    if first.button("Cevabı göster", type="primary", width="stretch"):
        reveal_answer(st.session_state)
    if second.button("Yeni soru", type="secondary", width="stretch"):
        next_question(st.session_state)
        st.rerun()
    if st.session_state.get(ANSWER_VISIBLE_KEY, False):
        st.success(f"**Çözüm:** {question.answer}")
    else:
        st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")


def render() -> None:
    """Konu 03'ün tam etkileşimli arayüzünü oluşturur."""
    st.markdown("<span class='topic-badge'>KONU 03</span>", unsafe_allow_html=True)
    st.header("Basit Doğrusal Regresyon")
    st.markdown(
        "<div class='lesson-note'><strong>Amaç:</strong> İki değişken arasındaki doğrusal ilişkiyi yatay kesit verisiyle betimlemek, basit EKK doğrusunu kurmak, eğim ve sabit terimi bağlama uygun biçimde yorumlamak ve belirli bir X değeri için tahmin üretmek.</div>",
        unsafe_allow_html=True,
    )

    datasets = list_datasets()
    dataset_key = st.selectbox(
        "Veri seti",
        options=[item.key for item in datasets],
        format_func=lambda key: get_dataset_metadata(key).title,
    )
    metadata = get_dataset_metadata(dataset_key)
    try:
        frame = _load_dataset_cached(dataset_key)
    except RuntimeError as error:
        st.error(str(error))
        return

    with st.expander("Veri seti kaynağı ve değişkenler", expanded=True):
        st.markdown(f"**Kaynak:** {metadata.source}")
        st.write(metadata.description)
        variable_table = pd.DataFrame(
            [
                {"Değişken": item.name, "Açıklama": item.description, "Birim": item.unit}
                for item in metadata.variables.values()
            ]
        )
        st.dataframe(variable_table, hide_index=True, width="stretch")

    dependent_options = tuple(metadata.allowed_pairs)
    dependent = st.selectbox(
        "Bağımlı değişken (Y)",
        options=dependent_options,
        format_func=lambda variable: _format_variable(dataset_key, variable),
    )
    explanatory_options = allowed_explanatory_variables(dataset_key, dependent)
    explanatory = st.selectbox(
        "Açıklayıcı değişken (X)",
        options=explanatory_options,
        format_func=lambda variable: _format_variable(dataset_key, variable),
    )

    try:
        result = fit_simple_ols(frame, dependent, explanatory)
        summary = descriptive_statistics(frame, (dependent, explanatory))
    except ValueError as error:
        st.error(f"Model kurulamadı: {error}")
        return

    y_info = variable_metadata(dataset_key, dependent)
    x_info = variable_metadata(dataset_key, explanatory)
    model_id = f"{dataset_key}:{dependent}:{explanatory}"

    cards = st.columns(2)
    cards[0].metric("Geçerli gözlem sayısı", result.nobs)
    cards[1].metric("Eğim katsayısı", f"{result.slope:.4f}")

    st.subheader("Seçili değişkenlerin tanımlayıcı bilgileri")
    st.dataframe(summary.round(3), width="stretch")

    st.subheader("Tahmin edilen basit regresyon modeli")
    st.latex(rf"\widehat{{{dependent}}} = {result.intercept:.4f} {('+' if result.slope >= 0 else '-')} {abs(result.slope):.4f}\,{explanatory}")
    st.caption("Bu eşitlik örneklemdeki doğrusal ilişkiyi betimler; tek başına nedensellik iddiası taşımaz.")
    coefficient_table = pd.DataFrame(
        {"Terim": ["Sabit", x_info.label], "Katsayı": [result.intercept, result.slope]}
    )
    st.dataframe(coefficient_table.round(4), hide_index=True, width="stretch")
    st.info(
        "Bu konuda standart hata, t istatistiği, p-değeri ve güven aralığı yorumlanmaz. Bu çıkarımsal araçlar sonraki konularda ele alınacaktır."
    )

    _render_prediction_panel(result, dataset_key)
    selected_position, _ = _render_observation_panel(result, dataset_key)
    st.subheader("Saçılım grafiği, EKK tahmin doğrusu ve seçili gözlem")
    st.plotly_chart(
        _scatter_figure(result, f"{x_info.label} ({x_info.unit})", f"{y_info.label} ({y_info.unit})", selected_position),
        width="stretch",
    )
    st.divider()
    _render_questions(result, f"{model_id}:observation:{selected_position}", dataset_key, selected_position)
