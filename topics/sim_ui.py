"""Sezgi sekmesi: veri üretim süreci bilinen kontrollü deneyler.

Her deney aynı sırayı izler: soru → DGP ve parametreler → neye bakıyoruz → sonuç → ne gördük → kod.
Kod, uygulamanın hesabıyla aynı tanımdan üretilir; Python sürümü bu sayfadaki sayıların aynısını verir.
"""

from __future__ import annotations

import streamlit as st

from core.charts import CHART_TYPES, figure_for, show_figure
from core.codegen.base import LANGUAGE_INFO, LANGUAGES, generator
from core.labs.runner import LabState, execute, shown_frame
from core.labs.sezgi import SimExperiment
from core.labs.spec import REPRO_DESCRIPTIONS, ReproClass, ShowFrame
from topics.lab_ui import CODE_LANGUAGE_KEY, display_table, frame_display, show_table


@st.cache_resource(show_spinner=False, max_entries=48)
def _run(key: str, parameters: tuple[tuple[str, float], ...], _experiment: SimExperiment) -> LabState:
    state = LabState()
    for op in _experiment.build(dict(parameters)):
        execute(op, state)
    return state


def _parameter_key(experiment: SimExperiment, name: str) -> str:
    return f"{experiment.key}_{name}"


def _current_parameters(experiment: SimExperiment) -> dict[str, float]:
    values = {}
    for item in experiment.parameters:
        key = _parameter_key(experiment, item.key)
        if key not in st.session_state:
            st.session_state[key] = int(item.default) if item.integer else float(item.default)
        values[item.key] = st.session_state[key]
    return values


def _render_sliders(experiment: SimExperiment) -> None:
    for column, item in zip(st.columns(len(experiment.parameters)), experiment.parameters):
        if item.integer:
            column.slider(item.label, min_value=int(item.minimum), max_value=int(item.maximum), step=int(item.step),
                          key=_parameter_key(experiment, item.key), help=item.help)
        else:
            column.slider(item.label, min_value=float(item.minimum), max_value=float(item.maximum),
                          step=float(item.step), key=_parameter_key(experiment, item.key), help=item.help,
                          format=f"%.{item.decimals}f")


def _render_code(experiment: SimExperiment, parameters: dict[str, float]) -> None:
    language = st.session_state.get(CODE_LANGUAGE_KEY) or LANGUAGES[0]
    info = LANGUAGE_INFO[language]
    code = generator(experiment.spec(parameters), language).script()
    title, description = REPRO_DESCRIPTIONS[ReproClass.DISTRIBUTIONAL]
    st.markdown(f"**{language} kodu** (şu anki kaydırıcı değerleriyle)")
    st.caption(f"İki dilde sonuç: **{title}**. {description} Python sürümü bu sayfadaki sayıların aynısını verir.")
    with st.expander("Kodu göster", icon=":material/code:"):
        st.code(code, language=info.highlight, line_numbers=True)
    st.download_button(
        f"{language} kodunu indir (.{info.extension})",
        data=code,
        file_name=f"ikt305_{experiment.key}.{info.extension}",
        mime=info.mime,
        key=f"{experiment.key}_download_{language}",
        icon=":material/download:",
    )


def _selector_key(topic_key: str) -> str:
    return f"{topic_key}_sezgi_deney"


def widget_keys(experiments: tuple[SimExperiment, ...]) -> set[str]:
    """Deney seçimi ve kaydırıcıların oturum anahtarları (``topics.shared.keep_widget_state``)."""

    keys = {_selector_key(experiments[0].topic_key)}
    return keys | {_parameter_key(experiment, item.key) for experiment in experiments for item in experiment.parameters}


def render_experiments(experiments: tuple[SimExperiment, ...]) -> None:
    topic_key = experiments[0].topic_key
    st.markdown(
        "Bu sekmedeki veriler **simülasyondur**: veri üretim süreci (DGP) bilinir. Böylece gerçek veride "
        "göremediğimiz nesneleri (ör. anakütle ortalaması, gerçek etki) veriden hesaplananla yan yana "
        "görebiliriz. Notlardaki örnekler **Uygulama** sekmesindedir."
    )
    selector = _selector_key(topic_key)
    numbers = [item.number for item in experiments]
    if st.session_state.get(selector) not in numbers:
        st.session_state[selector] = numbers[0]
    titles = {item.number: item.title for item in experiments}
    st.segmented_control("Deney", options=numbers, format_func=lambda n: f"Deney {n}: {titles[n]}", key=selector,
                         label_visibility="collapsed")
    experiment = next(item for item in experiments if item.number == st.session_state[selector])

    st.subheader(f"Deney {experiment.number}: {experiment.title}")
    st.caption(experiment.note.label())
    st.markdown(f"**Soru.** {experiment.question}")

    parameters = _current_parameters(experiment)
    with st.container(border=True):
        st.markdown("**Veri üretim süreci (DGP)**")
        for line in experiment.dgp(parameters):
            st.latex(line)
        st.caption(experiment.dgp_note)
        _render_sliders(experiment)

    state = _run(experiment.key, tuple(sorted(parameters.items())), experiment)
    st.markdown("**Neye bakıyoruz?**")
    st.markdown("\n".join(f"- {line}" for line in experiment.look_at))
    operations = experiment.build(parameters)
    for index, op in enumerate(operations):
        if isinstance(op, CHART_TYPES):
            show_figure(figure_for(op, state, experiment.label), key=f"{experiment.key}_grafik_{index}")
        elif isinstance(op, ShowFrame):
            st.markdown(f"**{op.comment}**")
            show_table(frame_display(shown_frame(op, state), experiment.label, op.decimals))

    metrics = experiment.metrics(state, parameters)
    for column, metric in zip(st.columns(len(metrics)), metrics):
        column.metric(metric.label, metric.value, help=metric.help)

    producers = {op.result: op for op in operations if hasattr(op, "result")}
    for name, title in experiment.tables:
        st.markdown(f"**{title}**")
        show_table(display_table(producers[name], state.tables[name], experiment.label))

    st.info(experiment.takeaway(state, parameters), icon=":material/lightbulb:")
    _render_code(experiment, parameters)
