"""Konu 05: Çoklu regresyon ve ceteris paribus öğretim modülü."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data_registry import RegressionModelSpec, get_dataset_metadata, konu05_model_specs, load_dataset, variable_metadata
from core.konu05_questions import generate_konu05_question
from core.model_utils import fit_simple_ols, format_number
from core.multiple_regression_utils import add_konu05_derived_columns, fit_multiple_ols, multiple_observation_result, partial_regression_data, profile_prediction_difference
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state


@st.cache_data(show_spinner=False)
def _dataset(key: str) -> pd.DataFrame:
    """Wooldridge verisini ve gerekli açık dönüşümleri önbelleğe alır."""
    return add_konu05_derived_columns(load_dataset(key), key)


def _fit(spec: RegressionModelSpec, frame: pd.DataFrame):
    """Sabit spesifikasyona uygun basit veya çoklu modeli kurar."""
    explanatory = (spec.focal_explanatory, *spec.controls)
    if len(explanatory) == 1:
        return fit_simple_ols(frame, spec.dependent, explanatory[0]), explanatory
    return fit_multiple_ols(frame, spec.dependent, explanatory), explanatory


def _coefficient_table(result, spec: RegressionModelSpec, dataset: str) -> pd.DataFrame:
    """Çıkarımsal alanlar içermeyen öğrenci katsayı tablosunu üretir."""
    rows = []
    explanatory = result.explanatory if isinstance(result.explanatory, tuple) else (result.explanatory,)
    for item in ("const", *explanatory):
        if item == "const":
            rows.append({"Değişken": "Sabit terim", "Modeldeki rolü": "Sabit terim", "Tahmin edilen katsayı": result.coefficients[item] if hasattr(result, "coefficients") else result.intercept, "Ölçü birimi": variable_metadata(dataset, spec.dependent).unit, "Sabit tutulan değişkenler": "—"})
        else:
            info = variable_metadata(dataset, item)
            others = [variable_metadata(dataset, name).label for name in explanatory if name != item]
            coefficient = result.coefficients[item] if hasattr(result, "coefficients") else result.slope
            rows.append({"Değişken": info.label, "Modeldeki rolü": "Temel açıklayıcı" if item == spec.focal_explanatory else "Kontrol", "Tahmin edilen katsayı": coefficient, "Ölçü birimi": f"{variable_metadata(dataset, spec.dependent).unit}/{info.unit}", "Sabit tutulan değişkenler": ", ".join(others) or "—"})
    return pd.DataFrame(rows)


def _actual_fitted_figure(observed: pd.Series, fitted: pd.Series, unit: str, position: int, residual: float) -> go.Figure:
    """Gözlenen–tahmin edilen grafiği ve 45 derece çizgisini oluşturur."""
    low, high = float(min(observed.min(), fitted.min())), float(max(observed.max(), fitted.max()))
    fig = px.scatter(x=fitted, y=observed, labels={"x": f"Tahmin edilen değer ({unit})", "y": f"Gözlenen değer ({unit})"}, color_discrete_sequence=["#107C89"])
    fig.add_scatter(x=[low, high], y=[low, high], mode="lines", name="45 derece çizgisi", line={"color": "#B3392F", "dash": "dash"})
    fig.add_scatter(x=[float(fitted.iloc[position])], y=[float(observed.iloc[position])], mode="markers", name="Seçili gözlem", marker={"color": "#2F9E6B", "size": 14, "symbol": "diamond", "line": {"color": "#07373D", "width": 2}}, hovertemplate=f"Seçili gözlem: {position + 1}<br>Gözlenen=%{{y:.4f}}<br>Tahmin edilen={float(fitted.iloc[position]):.4f}<br>Artık={residual:.4f}<extra></extra>")
    return fig.update_layout(template="plotly_white", height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10})


def _result_value(result, name: str) -> str:
    """Basit ya da çoklu model sonucundaki katsayıyı tutarlı biçimde yazar."""
    if name == "const": return format_number(float(result.coefficients["const"] if hasattr(result, "coefficients") else result.intercept))
    if hasattr(result, "coefficients") and name in result.coefficients: return format_number(float(result.coefficients[name]))
    if not hasattr(result, "coefficients") and name == result.explanatory: return format_number(float(result.slope))
    return "—"


def _adjusted(result) -> float:
    """Basit ve çoklu model için düzeltilmiş R-kareyi döndürür."""
    return float(getattr(result, "adjusted_r_squared", 1 - (1 - result.r_squared) * (result.nobs - 1) / (result.nobs - 2)))


def _article_table(wage_results: list[object]) -> pd.DataFrame:
    """Ders notu yapısındaki üç sütunlu ücret tablosunu üretir."""
    labels = ["(1) Ücret", "(2) Ücret", "(3) ln(Ücret)"]
    rows = []
    for name, label in (("educ", "Eğitim"), ("exper", "Deneyim"), ("tenure", "Kıdem"), ("const", "Sabit")):
        rows.append({"Satır": label, **{column: _result_value(result, name) for column, result in zip(labels, wage_results)}})
    rows.extend([
        {"Satır": "Kontroller", "(1) Ücret": "Hayır", "(2) Ücret": "Evet", "(3) ln(Ücret)": "Evet"},
        {"Satır": "Gözlem sayısı", **{column: str(result.nobs) for column, result in zip(labels, wage_results)}},
        {"Satır": "R-kare", **{column: format_number(result.r_squared) for column, result in zip(labels, wage_results)}},
        {"Satır": "Düzeltilmiş R-kare", **{column: format_number(_adjusted(result)) for column, result in zip(labels, wage_results)}},
    ])
    return pd.DataFrame(rows)


def render() -> None:
    """Konu 05 öğretim bloklarını ders notu sırasıyla gösterir."""
    st.markdown("<span class='topic-badge'>KONU 05</span>", unsafe_allow_html=True)
    st.header("Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu")
    st.markdown("<div class='lesson-note'><strong>Geçiş:</strong> Konu 04'te tek açıklayıcılı modelin uyumunu, ölçü birimlerini ve fonksiyonel biçimlerini incelediniz. Şimdi ekonomik sonucu aynı anda birden fazla gözlenen faktörle ilişkilendiriyor ve her katsayıyı modeldeki diğer açıklayıcı değişkenler sabitken yorumluyoruz.</div>", unsafe_allow_html=True)
    st.info("Bu konuda katsayıların örnekleme belirsizliği ve hipotez testleri yorumlanmaz; bunlar sonraki konulara bırakılır.")
    st.subheader("Tek değişken neden yetmeyebilir?")
    st.write("Saatlik ücret eğitim yanında deneyim ve kıdemle de ilişkili olabilir. Basit ilişki ≠ kısmi ilişki. Çoklu regresyonun amacı mümkün olan bütün değişkenleri eklemek değil, araştırma sorusuna uygun gözlenen faktörlerle koşullu bir karşılaştırma kurmaktır.")
    st.warning("Pozitif bir basit katsayı nedensel sonuç vermez; modeldeki kontroller gözlenmeyen bütün faktörleri otomatik olarak çözmez.")

    specs = konu05_model_specs()
    selected_model_id = st.selectbox("Model kurma laboratuvarı", [item.model_id for item in specs], format_func=lambda ident: next(item.title for item in specs if item.model_id == ident), key="konu05_model")
    spec = next(item for item in specs if item.model_id == selected_model_id)
    metadata = get_dataset_metadata(spec.dataset_key)
    frame = _dataset(spec.dataset_key)
    try:
        result, explanatory = _fit(spec, frame)
    except ValueError as error:
        st.error(f"Model kurulamadı: {error}")
        return
    infos = {name: variable_metadata(spec.dataset_key, name) for name in (spec.dependent, *explanatory)}
    with st.expander("Veri seti, kaynak, değişkenler ve örneklem", expanded=True):
        st.markdown(f"**Kaynak:** {metadata.source}")
        st.write(f"**Gözlem birimi:** {metadata.observation_unit}. {metadata.description}")
        if spec.dataset_key == "hprice1": st.caption("Açık dönüşümler: lotsize1000 = lotsize / 1000; sqrft100 = sqrft / 100.")
        st.dataframe(pd.DataFrame([{"Değişken": item.name, "Açıklama": item.description, "Ölçü birimi": item.unit} for item in infos.values()]), hide_index=True, width="stretch")
    st.latex(rf"Y_i=\beta_0+\sum_{{j=1}}^k\beta_jX_{{ji}}+u_i\qquad \widehat{{Y}}_i=\widehat{{\beta}}_0+\sum_{{j=1}}^k\widehat{{\beta}}_jX_{{ji}}")
    cards = st.columns(3); cards[0].metric("Complete-case gözlem sayısı", result.nobs); cards[1].metric("R-kare", format_number(result.r_squared)); cards[2].metric("Düzeltilmiş R-kare", format_number(getattr(result, "adjusted_r_squared", 1-(1-result.r_squared)*(result.nobs-1)/(result.nobs-2))) )
    st.subheader("Katsayılar ve ceteris paribus yorumu")
    scenario_key = ""
    position = 0
    if hasattr(result, "coefficients"):
        st.dataframe(_coefficient_table(result, spec, spec.dataset_key).round(4), hide_index=True, width="stretch")
        focal_beta = float(result.coefficients[spec.focal_explanatory])
    else:
        focal_beta = result.slope
        st.dataframe(pd.DataFrame([{"Değişken": "Sabit terim", "Modeldeki rolü": "Sabit terim", "Tahmin edilen katsayı": result.intercept, "Ölçü birimi": infos[spec.dependent].unit, "Sabit tutulan değişkenler": "—"}, {"Değişken": infos[spec.focal_explanatory].label, "Modeldeki rolü": "Temel açıklayıcı", "Tahmin edilen katsayı": result.slope, "Ölçü birimi": f"{infos[spec.dependent].unit}/{infos[spec.focal_explanatory].unit}", "Sabit tutulan değişkenler": "—"}]).round(4), hide_index=True, width="stretch")
    controls = ", ".join(infos[item].label for item in spec.controls) or "modelde başka açıklayıcı yoktur"
    difference = st.number_input(f"{infos[spec.focal_explanatory].label} farkı", value=1.0, key="konu05_focal_difference")
    st.markdown(f"1. Değişen değişken: **{infos[spec.focal_explanatory].label}**, fark: {difference:g} {infos[spec.focal_explanatory].unit}.  \\n+2. Tahmin edilen {infos[spec.dependent].label} farkı: **{format_number(float(difference) * focal_beta)} {infos[spec.dependent].unit}**.  \\n+3. Sabit tutulanlar: **{controls}**.")
    st.caption("Ceteris paribus, gerçek dünyada değişkenleri fiziksel olarak eşitlemek değil, modeldeki diğer gözlenen doğrusal katkılar sabitken yapılan koşullu karşılaştırmadır.")

    if hasattr(result, "coefficients"):
        st.subheader("Profil karşılaştırması ve katkılar")
        columns = st.columns(len(explanatory)); a = {}; b = {}
        for col, item in zip(columns, explanatory):
            default = float(result.design_data[item].median()); a[item] = col.number_input(f"A — {infos[item].label}", value=default, key=f"konu05_a_{item}"); b[item] = col.number_input(f"B — {infos[item].label}", value=default + (1.0 if item == spec.focal_explanatory else 0.0), key=f"konu05_b_{item}")
        comparison = profile_prediction_difference(result, a, b)
        scenario_key = f"{tuple(sorted(a.items()))}:{tuple(sorted(b.items()))}"
        st.write(f"A tahmini: {format_number(float(comparison['prediction_a']))}; B tahmini: {format_number(float(comparison['prediction_b']))}; B−A: {format_number(float(comparison['difference']))}.")
        contribution_rows = [{"Değişken": infos[key].label, "Profil A değeri": a[key], "Profil B değeri": b[key], "Değişim (B − A)": b[key] - a[key], "Tahmin edilen katsayı": float(result.coefficients[key]), "Tahmin farkına katkı": value} for key, value in comparison["contributions"].items()]
        st.dataframe(pd.DataFrame(contribution_rows).round(4), hide_index=True, width="stretch")
        st.caption(r"Her satır: $\widehat\beta_j(X_{Bj}-X_{Aj})$. Katkı toplamı = " + format_number(sum(comparison["contributions"].values())) + "; B−A tahmin farkı = " + format_number(float(comparison["difference"])) + ".")
        st.info("Yalnız eğitim bir birim değiştiğinde eğitim katkısı = eğitim katsayısı × 1, deneyim ve kıdem katkıları 0 olur. Birden fazla değer değiştiğinde toplam tahmin farkı bütün katkıların toplamıdır.")
        st.subheader("Bir gözlem için tahmin edilen değer ve artık")
        position = st.slider("Seçili gözlem", 1, result.nobs, 1, key="konu05_observation") - 1
        selected = multiple_observation_result(result, position)
        st.dataframe(pd.DataFrame([*[{"Büyüklük": infos[key].label, "Değer": format_number(value)} for key, value in selected["explanatory"].items()], {"Büyüklük": "Gözlenen Y", "Değer": format_number(selected["observed"])}, {"Büyüklük": "Tahmin edilen Y", "Değer": format_number(selected["predicted"])}, {"Büyüklük": "Artık = Y − Ŷ", "Değer": format_number(selected["residual"])}, {"Büyüklük": "Artığın işareti", "Değer": "Pozitif" if selected["residual"] > 0 else "Negatif"}]), hide_index=True, width="stretch")
        st.plotly_chart(_actual_fitted_figure(result.observed_values, result.fitted_values, infos[spec.dependent].unit, position, float(selected["residual"])), width="stretch")
        st.subheader("Kısmi ilişki: kontrollerin doğrusal katkısını ayırmak")
        partial = partial_regression_data(result, spec.focal_explanatory)
        st.write("1. Y'yi kontrollere göre tahmin et. 2. Temel X'i aynı kontrollere göre tahmin et. 3. İki artığı ilişkilendir.")
        st.info(f"Kısmi regresyon eğimi = {format_number(partial.partial_slope)}; tam modeldeki {infos[spec.focal_explanatory].label} katsayısı = {format_number(partial.full_model_slope)}. Eşitlik tolerans içinde sağlanır: {'Evet' if partial.slopes_match else 'Hayır'}. Bu nedensellik kanıtı değildir.")
    st.subheader("Basit ve çoklu model karşılaştırması")
    simple_spec, multiple_spec = specs[0], specs[1]
    simple_result, _ = _fit(simple_spec, _dataset("wage1")); multiple_result, _ = _fit(multiple_spec, _dataset("wage1"))
    st.dataframe(pd.DataFrame({"Satır": ["Bağımlı değişken", "Açıklayıcı değişkenler", "Temel açıklayıcı değişken", "Kontrol değişkenleri", "Eğitim katsayısı", "Sabit terim", "Gözlem sayısı", "R-kare", "Düzeltilmiş R-kare"], "Basit model: wage ~ educ": ["Ücret", "Eğitim", "Eğitim", "—", _result_value(simple_result, "educ"), _result_value(simple_result, "const"), str(simple_result.nobs), format_number(simple_result.r_squared), format_number(_adjusted(simple_result))], "Çoklu model: wage ~ educ + exper + tenure": ["Ücret", "Eğitim, deneyim, kıdem", "Eğitim", "Deneyim, kıdem", _result_value(multiple_result, "educ"), _result_value(multiple_result, "const"), str(multiple_result.nobs), format_number(multiple_result.r_squared), format_number(_adjusted(multiple_result))]}), hide_index=True, width="stretch")
    st.caption("Basit modelde eğitim katsayısı eğitim ile ücret arasındaki örneklem ilişkisinin basit doğrusal özetidir. Çoklu modelde eğitim katsayısı deneyim ve kıdemin doğrusal katkısı sabit tutulduğunda elde edilen kısmi ilişkidir.")
    st.warning("Katsayının değişmesi tek başına yazılım hatası, nedensel düzeltme veya eksik değişken yanlılığı testi değildir. Çoklu modelin daha yüksek R-kare vermesi nedensel olarak doğru olduğu anlamına gelmez.")
    st.subheader("R-kare ve düzeltilmiş R-kare")
    r2_col, adjusted_col = st.columns(2)
    r2_col.markdown("**R-kare**"); r2_col.latex(r"R^2=\frac{\mathrm{MKT}}{\mathrm{TKT}}=1-\frac{\mathrm{HKT}}{\mathrm{TKT}}"); r2_col.caption("Bağımlı değişkendeki örneklem değişiminin modelle bağlantılı bölümünü özetler. Yeni açıklayıcı değişken eklenince düşmez; nedensellik veya doğru model garantisi değildir.")
    adjusted_col.markdown("**Düzeltilmiş R-kare**"); adjusted_col.latex(r"\bar R^2=1-(1-R^2)\frac{n-1}{n-k-1}"); adjusted_col.caption("Açıklayıcı değişken sayısını dikkate alan örneklem içi uyum özetidir. Yeni değişken eklenince düşebilir; nedensellik veya doğru model garantisi değildir. Burada n gözlem sayısı, k sabit dışındaki açıklayıcı sayısıdır.")
    st.warning("Uyum ölçüleri yalnız aynı bağımlı değişken ve aynı örneklemde karşılaştırılır. wage ve ln(wage) modellerinin R-kareleri mekanik olarak sıralanmaz.")
    st.subheader("Makale tipi tablo")
    wage_specs = [item for item in specs if item.dataset_key == "wage1"]
    wage_results = [_fit(item, _dataset("wage1"))[0] for item in wage_specs]
    st.dataframe(_article_table(wage_results), hide_index=True, width="stretch"); st.caption("Sütun (2) ve (3)'te kontrol değişkenleri deneyim ve kıdemdir. “Kontroller: Evet” ifadesi tek başına hangi değişkenlerin kontrol edildiğini göstermediği için yeterli değildir. Sütun (1) ve (2) aynı bağımlı değişkeni ve aynı örneklemi kullandığı için uyum ölçüleri doğrudan karşılaştırılabilir; sütun (3)'te ln(Ücret) kullanıldığı için mekanik sıralama yapılmaz.")
    st.subheader("Kontrol değişkeni seçimi")
    st.write("Temel açıklayıcı değişken, katsayısı özellikle yorumlanmak istenen değişkendir. Kontrol değişkeni, hedeflenen kısmi karşılaştırmayı daha anlamlı kurmak amacıyla modelde tutulan gözlenen değişkendir.")
    st.markdown("Bir değişkeni kontrol olarak düşünürken:\n\n1. Bağımlı değişkenle neden ilişkili olabilir?\n2. Temel açıklayıcı değişkenle ilişkili olabilir mi?\n3. Temel açıklayıcı değişkenden önce mi, sonra mı belirlenir?\n4. Ölçümü yeterince güvenilir midir?\n5. Sabit tutmak araştırma sorusuna uygun mudur?\n6. Başka bir değişkenin gereksiz tekrarı mıdır?")
    st.warning("Amaç mümkün olan bütün değişkenleri modele eklemek değildir. Kontrol seçimi araştırma sorusuna, teoriye ve zamansal sıraya dayanmalıdır. Eğitimden sonra belirlenen meslek gibi bir değişkeni sabit tutmak, soruya bağlı olarak ilişkinin bir bölümünü dışarıda bırakabilir.")

    topic_id = "konu05"; model_id = f"{spec.model_id}:{position}:{difference}:{scenario_key}"
    synchronize_question_state(st.session_state, model_id, topic_id); index_key, _, answer_key = question_state_keys(topic_id); index = int(st.session_state.get(index_key, 0))
    multi_result = result if hasattr(result, "coefficients") else fit_multiple_ols(frame, spec.dependent, (spec.focal_explanatory, "exper" if spec.dataset_key == "wage1" else "lotsize1000"))
    question = generate_konu05_question(model_id, index, multi_result, spec, infos)
    st.subheader("Kendini dene"); st.markdown(f"**Soru {index + 1}:** {question.prompt}"); left, right = st.columns(2)
    if left.button("Cevabı göster", key="konu05_show_answer", type="primary", width="stretch"): reveal_answer(st.session_state, topic_id)
    if right.button("Yeni soru", key="konu05_next_question", type="secondary", width="stretch"): next_question(st.session_state, topic_id); st.rerun()
    if st.session_state.get(answer_key, False): st.success(f"**Çözüm:** {question.answer}")
    else: st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")
