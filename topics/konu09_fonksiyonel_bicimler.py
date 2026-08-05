"""Konu 09: ölçekleme, log modeller, karesel terimler ve model seçimi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.ui_components import render_question_actions

from core.data_registry import get_dataset_metadata, load_dataset
from core.functional_form_utils import (add_wage1_quadratic_columns, center_quadratic_model, log_level_percent_change,
    quadratic_discrete_change, quadratic_marginal_effect, quadratic_turning_point, rescale_and_refit, standardized_regression,
    wage1_model_comparison, wrong_functional_form_data)
from core.konu09_questions import Konu09QuestionContext, generate_konu09_question
from core.regression_inference_utils import format_p_value, significance_stars
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state
from core.ui_preferences import plot_font_sizes


BRAND = {"ink": "#07373D", "teal": "#107C89", "green": "#2F9E6B", "red": "#B3392F"}


@st.cache_data(show_spinner=False)
def _comparison():
    """WAGE1 M1–M4 sonuçlarını ve ortak F'yi önbelleğe alır."""
    return wage1_model_comparison(load_dataset("wage1"))


def render() -> None:
    """Konu 09 arayüzünü ders notu sırasıyla görüntüler."""
    comparison = _comparison(); data = add_wage1_quadratic_columns(load_dataset("wage1")); m4 = comparison.models["M4"]
    fonts = plot_font_sizes(st.session_state.get("text_scale", 1.10))
    st.markdown("<span class='topic-badge'>KONU 09</span>", unsafe_allow_html=True)
    st.header("Ölçekleme, Logaritmik Modeller, Karesel Terimler ve Model Seçimi")
    st.markdown("<div class='lesson-note'><strong>Geçiş:</strong> Aynı doğrusal EKK altyapısı, X² gibi dönüştürülmüş değişkenlerle eğri ilişkileri de tahmin edebilir. Burada parametrelerde doğrusal olma, birimler ve model seçimi birlikte ele alınır.</div>", unsafe_allow_html=True)
    st.warning("Standartlaştırma veya merkezleme nedensellik, içsellik ya da doğru model garantisi sağlamaz. Kukla ve etkileşim terimleri Konu 10–11, dayanıklı çıkarım Konu 12 kapsamındadır.")

    st.subheader("Parametrelerde doğrusallık ve ölçekleme")
    st.latex(r"E(Y\mid X)=\beta_0+\beta_1X+\beta_2X^2")
    st.write("Bu model parametrelerde doğrusaldır: X² veri sütunudur. β₁² içeren model parametrelerde doğrusal değildir.")
    scale_cols = st.columns(2)
    y_factor = scale_cols[0].selectbox("Y ölçek çarpanı", (1.0, 10.0, 100.0), index=0, key="konu09_y_scale")
    x_factor = scale_cols[1].selectbox("Eğitim X ölçek çarpanı", (1.0, 0.1, 10.0), index=0, key="konu09_x_scale")
    scaling = rescale_and_refit(data, "lwage", ("educ", "exper", "tenure"), dependent_factor=y_factor, explanatory_factor=x_factor, focal_explanatory="educ")
    st.dataframe(pd.DataFrame([{"Model": "Ham", "Eğitim katsayısı": scaling.original.coefficients["educ"], "SH": scaling.original.standard_errors["educ"], "t": scaling.original.t_values_zero["educ"], "p": format_p_value(scaling.original.p_values_two_sided_zero["educ"]), "R²": scaling.original.r_squared}, {"Model": "Ölçeklenmiş", "Eğitim katsayısı": scaling.rescaled.coefficients["educ"], "SH": scaling.rescaled.standard_errors["educ"], "t": scaling.rescaled.t_values_zero["educ"], "p": format_p_value(scaling.rescaled.p_values_two_sided_zero["educ"]), "R²": scaling.rescaled.r_squared}]), hide_index=True, width="stretch")
    st.caption("Katsayı ve standart hata birimle değişir; eşdeğer modelde t, p, F ve R² değişmez.")

    st.subheader("Standartlaştırılmış katsayılar")
    standardized = standardized_regression(data, "lwage", ("educ", "exper", "tenure"))
    st.dataframe(pd.DataFrame({"Değişken": standardized.standardized_coefficients.index, "Standartlaştırılmış katsayı": standardized.standardized_coefficients.values}), hide_index=True, width="stretch")
    st.caption("Standartlaştırılmış katsayı bir standart sapmalık X farkıyla ilişkili Y standart sapması farkını özetler; nedensel önem sırası değildir.")

    st.subheader("Log-düzey: yaklaşık ve tam yüzde değişim")
    controls = st.columns(2)
    beta = controls[0].number_input("Log-düzey katsayısı β", value=0.06, step=0.01, key="konu09_log_beta")
    delta = controls[1].slider("X değişimi", 1, 6, 1, key="konu09_log_delta")
    st.metric("Yaklaşık yüzde", f"%{log_level_percent_change(beta, delta):.2f}")
    st.metric("Tam yüzde", f"%{log_level_percent_change(beta, delta, exact=True):.2f}")
    st.caption("Log-log modelde katsayı esneklik olarak okunur; log-düzeyde yaklaşık ve tam yüzde hesapları açıkça ayrılır.")

    st.subheader("Karesel model: marjinal etki ve dönüm noktası")
    with st.expander("Genel karesel model laboratuvarı"):
        generic_b1 = st.number_input("Genel model β1", value=4.0, step=0.1, key="konu09_generic_b1")
        generic_b2 = st.number_input("Genel model β2", value=-0.10, step=0.01, key="konu09_generic_b2")
        generic_x = st.slider("Genel model X", 0, 30, 10, key="konu09_generic_x")
        st.write(f"X={generic_x} için marjinal etki: **{quadratic_marginal_effect(generic_b1, generic_b2, generic_x):.3f}**; X→X+1 tam değişim: **{quadratic_discrete_change(generic_b1, generic_b2, generic_x):.3f}**.")
        generic_turn = quadratic_turning_point(generic_b1, generic_b2, np.arange(0, 31))
        st.caption(f"Dönüm noktası: {generic_turn.value:.3f}" if generic_turn.value is not None else "β2=0 olduğunda dönüm noktası yoktur.")
    b1, b2 = float(m4.coefficients["exper"]), float(m4.coefficients["expersq"])
    x = st.slider("İş deneyimi (yıl)", int(data.exper.min()), int(data.exper.max()), 10, key="konu09_exper")
    marginal, discrete = quadratic_marginal_effect(b1, b2, x), quadratic_discrete_change(b1, b2, x)
    point = quadratic_turning_point(b1, b2, data.exper)
    cols = st.columns(4)
    cols[0].metric("Marjinal log-eğim", f"{marginal:.4f}"); cols[1].metric("Yaklaşık yüzde", f"%{100*marginal:.2f}"); cols[2].metric("X→X+1 tam yüzde", f"%{log_level_percent_change(discrete, exact=True):.2f}"); cols[3].metric("Dönüm noktası", f"{point.value:.2f} yıl" if point.value is not None else "Yok")
    st.caption(f"Dönüm noktası {point.kind}; gözlenen deneyim aralığı [{point.data_min:.0f}; {point.data_max:.0f}] ve aralık içinde: {'evet' if point.inside_data_range else 'hayır'}. Düzey ve kare katsayıları birlikte yorumlanmalıdır.")
    grid = np.linspace(float(data.exper.min()), float(data.exper.max()), 120)
    figure = go.Figure(go.Scatter(x=grid, y=[quadratic_marginal_effect(b1, b2, value) * 100 for value in grid], line={"color": BRAND["teal"]}, name="Yaklaşık yüzde marjinal etki"))
    figure.add_hline(y=0, line_dash="dash", line_color=BRAND["red"])
    st.plotly_chart(figure.update_layout(template="plotly_white", height=300, font={"size": fonts["base"]}, hoverlabel={"font_size": fonts["hover"]}, legend={"font": {"size": fonts["legend"]}}, xaxis={"title": {"text": "İş deneyimi (yıl)", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}}, yaxis={"title": {"text": "Yaklaşık yüzde değişim", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}}), width="stretch")

    st.subheader("WAGE1 M1–M4 ve karesel terimlerin ortak F testi")
    meta = get_dataset_metadata("wage1"); st.caption(f"Kaynak: {meta.source} | Bağımlı değişken: ln(saatlik ücret)")
    st.dataframe(comparison.metrics, hide_index=True, width="stretch")
    comparison_chart = go.Figure(go.Bar(x=comparison.metrics["Model"], y=comparison.metrics["Düzeltilmiş R²"], marker_color=BRAND["teal"]))
    st.plotly_chart(comparison_chart.update_layout(template="plotly_white", height=270, font={"size": fonts["base"]}, hoverlabel={"font_size": fonts["hover"]}, xaxis={"title": {"text": "Model", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}}, yaxis={"title": {"text": "Düzeltilmiş R²", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}}), width="stretch")
    joint = comparison.quadratic_joint_test
    cols = st.columns(4); cols[0].metric("q", joint.q); cols[1].metric("F(2,520)", f"{joint.f_from_ssr:.3f}"); cols[2].metric("Ortak p", format_p_value(joint.p_value)); cols[3].metric("Formüller", "Uyumlu" if joint.formulas_match else "Denetlenmeli")
    st.caption("H0: expersq=0 ve tenursq=0. Konu 08'in aynı nested SSR/R² ve genel ortak F yardımcıları yeniden kullanılır; ortak ret karesel modelin bütün olası biçimler arasında kesin doğru olduğunu kanıtlamaz.")

    st.subheader("Merkezleme laboratuvarı")
    center_option = st.selectbox("Deneyim merkez noktası", (0.0, 5.0, 10.0, 20.0), index=2, key="konu09_center")
    centered = center_quadratic_model(data, "lwage", "exper", "expersq", ("educ", "tenure", "tenursq"), center=center_option)
    st.dataframe(pd.DataFrame([{"Merkez": centered.center, "Merkezlenmiş deneyim katsayısı": centered.centered_result.coefficients[centered.centered_column], "Ham modelde aynı noktada marjinal etki": centered.marginal_effect_at_center, "Fitted maksimum fark": centered.fitted_max_difference, "Artık maksimum fark": centered.residual_max_difference, "SSR farkı": centered.centered_result.ssr - centered.raw_result.ssr, "R² farkı": centered.centered_result.r_squared - centered.raw_result.r_squared, "Korelasyon önce": centered.correlation_before, "Korelasyon sonra": centered.correlation_after}]), hide_index=True, width="stretch")
    st.caption("Ekonomik ilişki değişmez; referans noktasındaki doğrusal katsayı değişir. Merkezleme içselliği çözmez.")

    st.subheader("Makale tipi tablo ve model seçimi")
    rows = []
    for variable, label in (("educ", "Eğitim"), ("exper", "Deneyim"), ("expersq", "Deneyim²"), ("tenure", "Kıdem"), ("tenursq", "Kıdem²"), ("const", "Sabit")):
        row = {"Satır": label}
        for name in ("M1", "M4"):
            result = comparison.models[name]
            row["(1) Doğrusal" if name == "M1" else "(2) Karesel"] = f"{result.coefficients[variable]:.3f}{significance_stars(result.p_values_two_sided_zero[variable])}\n({result.standard_errors[variable]:.3f})" if variable in result.coefficients else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Parantez içinde geleneksel EKK standart hataları; *** p<0.01, ** p<0.05, * p<0.10. Kanıt panelinde teori, işaretler, veri aralığı, basitlik, SSR, R², düzeltilmiş R² ve ortak F birlikte değerlendirilmelidir.")

    st.subheader("Python çıktısı, makale tablosu ve sık hatalar")
    st.code('model = smf.ols("lwage ~ educ + exper + expersq + tenure + tenursq", data=wage).fit()\nmodel.f_test("expersq = 0, tenursq = 0")', language="python")
    st.dataframe(pd.DataFrame([
        {"Sık hata": "Yalnız düzey katsayısını yorumlamak", "Neden sorun?": "Karesel modelde etki X düzeyine bağlıdır."},
        {"Sık hata": "X² ekleyip X'i çıkarmak", "Neden sorun?": "Hiyerarşi ilkesi ve referans eğimi bozulur."},
        {"Sık hata": "Dönüm noktasını örneklem dışına taşırmak", "Neden sorun?": "Veri dışındaki güçlü yorumlar desteklenmez."},
        {"Sık hata": "En yüksek R²'yi otomatik seçmek", "Neden sorun?": "Uyum, nedensellik veya örneklem dışı başarı garantisi değildir."},
        {"Sık hata": "Merkezleme içselliği çözer demek", "Neden sorun?": "Yalnız referans noktası değişir."},
    ]), hide_index=True, width="stretch")

    st.subheader("Yanlış fonksiyonel biçim")
    wrong = wrong_functional_form_data(); linear = np.polyfit(wrong.x, wrong.y, 1); quadratic = np.polyfit(wrong.x, wrong.y, 2)
    figure = go.Figure(); figure.add_trace(go.Scatter(x=wrong.x, y=wrong.y, mode="markers", marker={"size": 4, "color": "#AAAAAA"}, name="Gözlem")); figure.add_trace(go.Scatter(x=wrong.x, y=wrong.gercek_egri, line={"color": BRAND["green"]}, name="Gerçek eğri")); figure.add_trace(go.Scatter(x=wrong.x, y=np.polyval(linear, wrong.x), line={"color": BRAND["red"]}, name="Doğrusal fit")); figure.add_trace(go.Scatter(x=wrong.x, y=np.polyval(quadratic, wrong.x), line={"color": BRAND["teal"]}, name="Karesel fit")); st.plotly_chart(figure.update_layout(template="plotly_white", height=330, font={"size": fonts["base"]}, hoverlabel={"font_size": fonts["hover"]}, legend={"font": {"size": fonts["legend"]}}, xaxis={"title": {"text": "X", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}}, yaxis={"title": {"text": "Y", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}}), width="stretch")
    st.info("Bu yalnız illüstrasyondur; ayrı bir resmi tanı testi geliştirmez.")

    st.subheader("Kendini dene")
    context = Konu09QuestionContext("wage1-m4", comparison); synchronize_question_state(st.session_state, context.context_id, "konu09")
    index_key, _, answer_key = question_state_keys("konu09"); question = generate_konu09_question(context, int(st.session_state.get(index_key, 0))); st.write(question.prompt)
    render_question_actions(topic_id="konu09")
    if st.session_state.get(answer_key, False): st.success("Çözüm: " + question.answer)
