"""Konu 07: tek katsayı için geleneksel EKK çıkarımı öğretim arayüzü."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data_registry import get_dataset_metadata, konu07_model_specs, load_dataset, variable_metadata
from core.konu07_questions import Konu07QuestionContext, generate_konu07_question
from core.regression_inference_utils import (
    ci_test_equivalence, coefficient_confidence_interval, coefficient_test,
    fit_ols_inference, format_p_value, scale_coefficient_inference, selected_coverage_plot_data,
    significance_stars, simulate_confidence_coverage, simulate_standard_errors,
    t_distribution_plot_data,
)
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state


BRAND = {"ink": "#07373D", "teal": "#107C89", "bright": "#15A4B5", "green": "#2F9E6B", "red": "#B3392F"}


@st.cache_data(show_spinner=False)
def _model(model_id: str):
    """Sabit model seçiminin saf ve önbellekli çıkarım sonucunu döndürür."""
    spec = next(item for item in konu07_model_specs() if item.model_id == model_id)
    return fit_ols_inference(load_dataset(spec.dataset_key), spec.dependent, (spec.focal_explanatory, *spec.controls))


@st.cache_data(show_spinner=False)
def _standard_error_simulation(nobs: int, repetitions: int):
    """Pahalı olmayan fakat tekrarlı SH benzetimini önbellekler."""
    return simulate_standard_errors(nobs=nobs, repetitions=repetitions, seed=202507)


@st.cache_data(show_spinner=False)
def _coverage_simulation(nobs: int, repetitions: int, level: float):
    """Kapsama benzetimini seçilen anahtarla önbellekler."""
    return simulate_confidence_coverage(nobs=nobs, repetitions=repetitions, confidence_level=level, seed=202508)


def _display_number(value: float, decimals: int = 3) -> str:
    """Sonlu sayıyı öğrenci için okunur biçimde yazar."""
    return f"{float(value):.{decimals}f}"


def _label(spec, name: str) -> str:
    """Sabit terim dahil değişkenin Türkçe etiketini döndürür."""
    return "Sabit" if name == "const" else variable_metadata(spec.dataset_key, name).label


def _inference_table(spec, result) -> pd.DataFrame:
    """Modeldeki tüm katsayıları çıkarım alanlarıyla görünür tabloya dönüştürür."""
    rows = []
    for name in result.coefficients.index:
        lower, upper = result.confidence_intervals_95.loc[name]
        rows.append({
            "Değişken": _label(spec, name), "Modeldeki rolü": "Sabit terim" if name == "const" else "Açıklayıcı değişken",
            "Katsayı": _display_number(result.coefficients[name], 4), "Geleneksel standart hata": _display_number(result.standard_errors[name], 4),
            "t (H0: β=0)": _display_number(result.t_values_zero[name], 3), "İki taraflı p": format_p_value(result.p_values_two_sided_zero[name]),
            "%95 GA": f"[{_display_number(lower, 4)}; {_display_number(upper, 4)}]",
        })
    return pd.DataFrame(rows)


def _t_figure(inference) -> go.Figure:
    """Kritik bölgeleri ve p-kuyruğunu etiketli Student-t grafiğinde gösterir."""
    plot_limit = max(5.0, inference.critical_value + 1.2)
    data = t_distribution_plot_data(inference.df_resid, inference.t_statistic, plot_limit=plot_limit)
    figure = go.Figure(go.Scatter(x=data["t"], y=data["density"], name="t yoğunluğu", line={"color": BRAND["ink"]}))
    x, density = data["t"].to_numpy(), data["density"].to_numpy()
    observed_in_window = abs(inference.t_statistic) <= plot_limit
    if inference.alternative == "two-sided":
        mask = np.abs(x) >= abs(inference.t_statistic) if observed_in_window else np.zeros_like(x, dtype=bool)
        criticals = (-inference.critical_value, inference.critical_value)
    elif inference.alternative == "greater":
        mask, criticals = (x >= inference.t_statistic if observed_in_window else np.zeros_like(x, dtype=bool)), (inference.critical_value,)
    else:
        mask, criticals = (x <= inference.t_statistic if observed_in_window else np.zeros_like(x, dtype=bool)), (-inference.critical_value,)
    if mask.any():
        figure.add_trace(go.Scatter(x=x[mask], y=density[mask], fill="tozeroy", name="p-değeri alanı", line={"color": BRAND["bright"]}, fillcolor="rgba(21,164,181,0.25)"))
    for critical in criticals:
        figure.add_vline(x=critical, line_dash="dash", line_color=BRAND["red"], annotation_text="kritik sınır")
    if observed_in_window:
        figure.add_vline(x=inference.t_statistic, line_color=BRAND["green"], annotation_text="gözlenen t")
    else:
        edge = np.sign(inference.t_statistic) * plot_limit * 0.94
        direction = "→" if inference.t_statistic > 0 else "←"
        figure.add_annotation(x=edge, y=float(density.max()) * 0.82, text=f"Gözlenen t = {inference.t_statistic:.3f}, çizim alanının dışında {direction}", showarrow=True, arrowhead=2, arrowcolor=BRAND["green"], font={"color": BRAND["green"]}, xanchor="right" if inference.t_statistic > 0 else "left")
        figure.add_trace(go.Scatter(x=[edge], y=[0], mode="markers", marker={"symbol": "triangle-right" if inference.t_statistic > 0 else "triangle-left", "color": BRAND["green"], "size": 12}, name="gözlenen t çizim alanı dışında"))
    return figure.update_layout(template="plotly_white", height=330, margin={"l": 10, "r": 10, "t": 35, "b": 10}, xaxis_title="t istatistiği", yaxis_title="Yoğunluk")


def _confidence_figure(lower: float, upper: float, estimate: float, null: float) -> go.Figure:
    """Tek satırlı güven aralığı grafiğini kurar."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[lower, upper], y=["Güven aralığı", "Güven aralığı"], mode="lines+markers", name="Güven aralığı", line={"color": BRAND["teal"], "width": 6}))
    figure.add_trace(go.Scatter(x=[estimate], y=["Güven aralığı"], mode="markers", name="Katsayı", marker={"color": BRAND["ink"], "size": 12}))
    figure.add_vline(x=null, line_dash="dash", line_color=BRAND["red"], annotation_text="null")
    return figure.update_layout(template="plotly_white", height=230, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="Katsayı değeri", yaxis_title="")


def render() -> None:
    """Ders notu sırasını izleyen Konu 07 öğrenci arayüzünü görüntüler."""
    st.markdown("<span class='topic-badge'>KONU 07</span>", unsafe_allow_html=True)
    st.header("Tek Katsayı İçin Hipotez Testleri")
    st.markdown("<div class='lesson-note'><strong>Geçiş:</strong> Konu 06 yansızlıkta tahmin dağılımının merkezini açıkladı. Burada tek örneklemdeki katsayının belirsizliğini standart hata ile ölçecek ve tek katsayı hakkında sınırlı istatistiksel çıkarım yapacağız.</div>", unsafe_allow_html=True)
    st.warning("Bu konu yalnız tek katsayıya ilişkin çıkarımı ele alır. Birden fazla katsayının birlikte sınanması ve F testi Konu 08'in içeriğidir. Standart hatalar geleneksel, homoskedastisiteye dayanan EKK standart hatalarıdır.")

    st.subheader("Nokta tahmini neden yetmez?")
    st.dataframe(pd.DataFrame([{"Araştırma": "A", "Katsayı": 0.60, "Standart hata": 0.05, "İlk okuma": "Görece hassas"}, {"Araştırma": "B", "Katsayı": 0.60, "Standart hata": 0.40, "İlk okuma": "Çok daha belirsiz"}]), hide_index=True, width="stretch")
    st.write("“Katsayı kaç?” ile “Bu katsayı anakütlede null değerden ayrışıyor mu?” aynı soru değildir.")
    st.subheader("Üç farklı değişkenlik ve standart hata sezgisi")
    st.dataframe(pd.DataFrame([{"Büyüklük": "Y'nin standart sapması", "Neyi özetler?": "Bağımlı değişkenin toplam yayılımı"}, {"Büyüklük": "Artıkların standart sapması", "Neyi özetler?": "Modelin gözlem düzeyindeki tipik hatası"}, {"Büyüklük": "Katsayının standart hatası", "Neyi özetler?": "Katsayı tahmininin örnekleme belirsizliği"}]), hide_index=True, width="stretch")
    st.latex(r"\operatorname{se}(\widehat\beta_j)=\frac{\widehat\sigma}{\sqrt{\sum_i(X_{ji}-\bar X_j)^2}\sqrt{1-R_j^2}}")
    st.caption("Formülün ezberlenmesi beklenmez: artık değişkenliği ve Rj² yükseldikçe SH genellikle yükselir; Xj değişkenliği ve n arttıkça genellikle düşer.")

    st.subheader("Standart hata benzetimi")
    n_sim, rep_sim = st.columns(2)
    nobs_sim = n_sim.selectbox("Benzetim örneklem büyüklüğü", (30, 50, 100, 250), index=1, key="konu07_se_n")
    repetitions_sim = rep_sim.selectbox("Benzetim tekrar sayısı", (500, 1000, 5000), index=2, key="konu07_se_repetitions")
    simulation = _standard_error_simulation(nobs_sim, repetitions_sim)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Gerçek eğim", _display_number(simulation.true_slope))
    metric_columns[1].metric("Tahminlerin tekrarlı örneklem standart sapması", _display_number(simulation.empirical_slope_std, 4))
    metric_columns[2].metric("Ortalama raporlanan standart hata", _display_number(simulation.mean_reported_standard_error, 4))
    fig_sim = px.histogram(x=simulation.slope_estimates, nbins=40, labels={"x": "Eğim tahminleri", "y": "Sıklık"}, color_discrete_sequence=[BRAND["teal"]])
    fig_sim.add_vline(x=simulation.true_slope, line_color=BRAND["green"], annotation_text="gerçek eğim")
    st.plotly_chart(fig_sim.update_layout(template="plotly_white", height=300), width="stretch")
    st.caption("Raporlanan standart hata, tekrar tekrar örneklem alsaydık katsayı tahminlerinin ne kadar değişeceğini tek örneklemden tahmin etmeye çalışır; iki ölçünün birebir eşit olması beklenmez.")

    specs = konu07_model_specs()
    selected_id = st.selectbox("Çıkarım modeli", tuple(spec.model_id for spec in specs), format_func=lambda item: next(spec.title for spec in specs if spec.model_id == item), key="konu07_model")
    spec = next(item for item in specs if item.model_id == selected_id)
    result = _model(spec.model_id)
    metadata = get_dataset_metadata(spec.dataset_key)
    names = tuple(result.coefficients.index)
    default_name = spec.focal_explanatory if spec.focal_explanatory in names else names[0]
    coefficient = st.selectbox("Katsayı satırı", names, index=names.index(default_name), format_func=lambda item: _label(spec, item), key="konu07_coefficient")
    st.subheader("Model, veri kaynağı ve çıkarım tablosu")
    st.caption(f"Kaynak: {metadata.source} | Gözlem birimi: {metadata.observation_unit} | {metadata.description}")
    st.write(f"Model: **{spec.dependent} ~ {' + '.join((spec.focal_explanatory, *spec.controls))}**. n={result.nobs}, k={result.n_explanatory}, serbestlik derecesi={result.df_resid}, R²={result.r_squared:.3f}, düzeltilmiş R²={result.adjusted_r_squared:.3f}.")
    st.dataframe(_inference_table(spec, result), hide_index=True, width="stretch")
    st.caption("Standart hata türü: geleneksel EKK (nonrobust). Bu standart hatalar homoskedastisite varsayımına dayanır. Dayanıklı standart hatalar Konu 12'de ele alınacaktır.")

    st.subheader("Hipotez kurma laboratuvarı")
    controls = st.columns(3)
    null_value = controls[0].number_input("Null değeri a", value=0.0, step=0.05, key="konu07_null")
    alternative = controls[1].selectbox("Alternatif", ("two-sided", "greater", "less"), format_func=lambda item: {"two-sided": "İki taraflı", "greater": "Daha büyük", "less": "Daha küçük"}[item], key="konu07_alternative")
    alpha = controls[2].selectbox("Anlamlılık düzeyi α", (0.10, 0.05, 0.01), index=1, key="konu07_alpha")
    inference = coefficient_test(result, coefficient, null_value=null_value, alternative=alternative, alpha=alpha)
    symbol = {"two-sided": "≠", "greater": ">", "less": "<"}[alternative]
    null_sign = "=" if alternative == "two-sided" else ("≤" if alternative == "greater" else "≥")
    st.latex(rf"H_0: \beta_{{{coefficient}}} {null_sign} {null_value:.3f}\qquad H_1: \beta_{{{coefficient}}} {symbol} {null_value:.3f}")
    values = st.columns(5)
    values[0].metric("Tahmin", _display_number(inference.estimate, 4)); values[1].metric("SH", _display_number(inference.standard_error, 4)); values[2].metric("t", _display_number(inference.t_statistic, 3)); values[3].metric("p", format_p_value(inference.p_value)); values[4].metric("Karar", "H0 reddedilir" if inference.reject_null else "H0 reddedilemez")
    if inference.t_statistic >= 0:
        st.write(f"Yorum: Tahmin, sıfır hipotezindeki değerin yaklaşık {inference.t_statistic:.3f} standart hata üzerindedir. İki taraflı testte karar |t| üzerinden verilir.")
    else:
        st.write(f"Yorum: Tahmin, sıfır hipotezindeki değerin yaklaşık {abs(inference.t_statistic):.3f} standart hata altındadır. İki taraflı testte karar |t| üzerinden verilir.")
    if alternative != "two-sided":
        st.info("Test yönü katsayının işareti ve p-değeri görüldükten sonra değil, araştırma sorusu veya teori temelinde önceden belirlenmelidir.")
    st.plotly_chart(_t_figure(inference), width="stretch")
    st.caption("p-değeri, H0 ve model varsayımları doğru kabul edildiğinde gözlenen kadar veya daha uç bir test istatistiği elde etme olasılığıdır.")
    st.markdown("**Seçilmiş %5 iki taraflı kritik değerler**")
    st.dataframe(pd.DataFrame({"sd": ["10", "20", "30", "60", "120", "Çok büyük"], "kritik değer": [2.228, 2.086, 2.042, 2.000, 1.980, 1.960]}), hide_index=True, width="stretch")
    st.caption("|t|>2 yalnız büyük örneklemler için kaba bir kontroldür; aktif testte exact Student-t kritik değeri kullanılır.")

    st.subheader("İki taraflı ve tek taraflı test neden farklı sonuç verebilir?")
    wage_for_tail = _model("W7-W")
    experience_two = coefficient_test(wage_for_tail, "exper", alternative="two-sided", alpha=0.05)
    experience_positive = coefficient_test(wage_for_tail, "exper", alternative="greater", alpha=0.05)
    st.dataframe(pd.DataFrame([
        {"Test": "İki taraflı", "Hipotez": "H1: β_exper ≠ 0", "p-değeri": format_p_value(experience_two.p_value), "%5 kararı": "H0 reddedilemez" if not experience_two.reject_null else "H0 reddedilir"},
        {"Test": "Pozitif tek taraflı", "Hipotez": "H1: β_exper > 0", "p-değeri": format_p_value(experience_positive.p_value), "%5 kararı": "H0 reddedilir" if experience_positive.reject_null else "H0 reddedilemez"},
    ]), hide_index=True, width="stretch")
    st.warning("Yön araştırma sorusu veya teoriyle önceden belirlenmelidir. Sonuç görüldükten sonra daha küçük p-değeri için yön değiştirilemez. Tek taraflı p-değeri her durumda iki taraflının yarısı değildir.")

    st.subheader("Güven aralığı ve test–GA eşdeğerliği")
    confidence_level = st.selectbox("Güven düzeyi", (0.90, 0.95, 0.99), index=1, format_func=lambda value: f"%{int(value * 100)}", key="konu07_confidence")
    lower, upper = coefficient_confidence_interval(result, coefficient, confidence_level=confidence_level)
    margin = (upper - lower) / 2
    st.write(f"{_label(spec, coefficient)} için %{int(confidence_level * 100)} GA: **[{_display_number(lower, 4)}; {_display_number(upper, 4)}]**; hata payı: {_display_number(margin, 4)}.")
    st.plotly_chart(_confidence_figure(lower, upper, inference.estimate, null_value), width="stretch")
    equivalence = ci_test_equivalence(inference)
    if equivalence.applicable:
        st.success(f"p-değeri kararı ile güven aralığı kararı uyumlu: {'Evet' if equivalence.consistent else 'Hayır — hesaplama denetlenmelidir.'}")
    else:
        st.caption("İki taraflı güven aralığı eşdeğerliği tek taraflı test için gösterilmez.")
    st.write("Aynı örnekleme ve aralık kurma yöntemi çok kez tekrarlansaydı, %95 aralıkların yaklaşık %95'i gerçek parametreyi kapsardı. Parametre sabittir; aralık rassaldır.")

    st.subheader("Güven aralığı kapsama benzetimi")
    coverage_controls = st.columns(3)
    nobs_coverage = coverage_controls[0].selectbox("Kapsama için n", (30, 50, 100, 250), index=1, key="konu07_coverage_n")
    reps_coverage = coverage_controls[1].selectbox("Kapsama tekrar sayısı", (500, 1000, 5000), index=1, key="konu07_coverage_reps")
    coverage_level = coverage_controls[2].selectbox("Kapsama güven düzeyi", (0.90, 0.95, 0.99), index=1, format_func=lambda value: f"%{int(value * 100)}", key="konu07_coverage_level")
    coverage = _coverage_simulation(nobs_coverage, reps_coverage, coverage_level)
    st.metric("Gözlenen kapsama oranı", f"{coverage.coverage_rate:.3f}")
    st.markdown("**Seçilmiş 25 güven aralığı**")
    plot_data = selected_coverage_plot_data(coverage)
    figure = go.Figure()
    for status, color, dash, symbol in (("Kapsıyor", BRAND["green"], "solid", "circle"), ("Kaçırıyor", BRAND["red"], "dash", "x")):
        subset = plot_data.loc[plot_data["status"] == status]
        for position, row in subset.iterrows():
            figure.add_trace(go.Scatter(x=[row["lower"], row["upper"]], y=[row["interval"], row["interval"]], mode="lines+markers", line={"color": color, "dash": dash, "width": 3}, marker={"color": color, "symbol": symbol, "size": 7}, name=status, legendgroup=status, showlegend=position == subset.index[0], hovertemplate=f"{status}<br>Alt=%{{x:.3f}}<extra></extra>"))
    figure.add_vline(x=coverage.true_slope, line_dash="dot", line_color=BRAND["ink"], annotation_text="gerçek eğim")
    st.plotly_chart(figure.update_layout(template="plotly_white", height=540, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="Güven aralığı sınırları", yaxis_title="Seçilmiş aralık sıra numarası"), width="stretch")
    misses = int((~plot_data["contains_truth"]).sum())
    st.caption("Aşağıdaki aralıklar, kapsayan ve kaçıran örnekleri birlikte göstermek amacıyla simülasyon tekrarları içinden seçilmiştir; ilk 25 tekrar değildir." if misses else "Bu küçük benzetimde kaçıran aralık oluşmadığı için seçili örnek yalnız kapsayan aralıklardan oluşur; toplam kapsama oranı değişmez.")

    st.subheader("Python çıktısı, makale tablosu ve iktisadi önem")
    st.code('import statsmodels.formula.api as smf\nmodel = smf.ols("wage ~ educ + exper + tenure", data=data).fit()', language="python")
    st.write("`coef`, `std err`, `t`, `P>|t|`, `[0.025, 0.975]`, `Df Residuals` ve `Covariance Type: nonrobust` alanlarını modelin birimi ve araştırma sorusuyla birlikte okuyun.")
    wage_specs = [item for item in specs if item.dataset_key == "wage1"]
    wage_results = [_model(item.model_id) for item in wage_specs]
    article_rows = []
    for variable in ("educ", "exper", "tenure", "const"):
        coefficient_row = {"Satır": _label(wage_specs[0], variable)}
        standard_error_row = {"Satır": ""}
        for item, model_result in zip(wage_specs, wage_results):
            if variable in model_result.coefficients:
                coefficient_row[item.title] = f"{model_result.coefficients[variable]:.3f}{significance_stars(model_result.p_values_two_sided_zero[variable])}"
                standard_error_row[item.title] = f"({model_result.standard_errors[variable]:.3f})"
            else:
                coefficient_row[item.title] = "—"
                standard_error_row[item.title] = ""
        article_rows.extend((coefficient_row, standard_error_row))
    st.dataframe(pd.DataFrame(article_rows), hide_index=True, width="stretch")
    st.caption("Parantez içinde geleneksel EKK standart hataları: *** p<0.01, ** p<0.05, * p<0.10. Yıldızlar etki büyüklüğü değildir; parantezlerin anlamı her makalede tablo notundan okunur.")
    education_result = wage_results[0]
    education_low, education_high = coefficient_confidence_interval(education_result, "educ")
    years = st.slider("WAGE1 eğitim farkı (yıl)", 1, 6, 4, key="konu07_education_change")
    scaled_education = scale_coefficient_inference(float(education_result.coefficients["educ"]), education_low, education_high, years)
    st.write(f"{years} yıllık eğitim farkı için tahmin edilen ücret farkı: **{_display_number(scaled_education.scaled_estimate, 3)} ABD doları/saat**; %95 aralık: [{_display_number(scaled_education.scaled_lower, 3)}; {_display_number(scaled_education.scaled_upper, 3)}]. Bu gözlemsel model ilişkiyi özetler, nedensellik kanıtlamaz.")
    hprice = _model("H7-P")
    hlow, hupp = coefficient_confidence_interval(hprice, "sqrft")
    home_change = st.selectbox("HPRICE1 konut alanı farkı (kare fit)", (50, 100, 250, 500), index=1, key="konu07_hprice_change")
    scaled_home = scale_coefficient_inference(float(hprice.coefficients["sqrft"]), hlow, hupp, home_change)
    st.write(f"HPRICE1'de {home_change} kare fit farkı için tahmin edilen fiyat farkı: **{_display_number(scaled_home.scaled_estimate, 2)} bin ABD doları**; %95 aralık: [{_display_number(scaled_home.scaled_lower, 2)}; {_display_number(scaled_home.scaled_upper, 2)}]. Δprice = Δsqrft × β̂_sqrft. Bu gözlemsel sonuç tahmin edilen farkı özetler; nedensellik kanıtlamaz.")
    st.caption("Yatak odası katsayısının aralığı sıfırı kapsıyorsa ayrı katkı dar biçimde belirlenememiştir; bu ‘etki yoktur’ demek değildir.")

    st.subheader("Sık yapılan yanlış yorumları düzeltelim")
    misconceptions = (
        ("p = 0.03 olduğuna göre H0 yüzde 3 olasılıkla doğrudur.", "p-değeri H0'ın olasılığı değildir.", "H0 ve model varsayımları doğru kabul edilirse en az bu kadar uç bir istatistik görme olasılığı 0.03'tür."),
        ("Katsayı anlamlı değilse etki yoktur.", "Reddedememek katsayının sıfır olduğunu göstermez.", "Veri, ayrı katkıyı seçilen düzeyde dar biçimde belirleyememiş olabilir."),
        ("Üç yıldızlı katsayı iktisadi olarak çok büyüktür.", "Yıldızlar tablo notundaki p-eşiklerine dayanır.", "İktisadi büyüklük katsayı, birim ve anlamlı değişim miktarıyla ayrıca değerlendirilir."),
        ("Güven aralığı sıfırı kapsamıyorsa X kesin olarak Y'ye neden olur.", "İstatistiksel çıkarım nedensel tasarımın yerini tutmaz.", "Gözlemsel model koşullu ilişkiyi özetler; nedensellik ek tasarım ve varsayım gerektirir."),
        ("p = 0.000 gerçek olasılığın sıfır olduğunu gösterir.", "Bu yalnız yuvarlama kaynaklı yanıltıcı bir görünüm olur.", "Çok küçük p-değeri '< 0.001' olarak gösterilmelidir."),
        ("H0 reddedilemediğine göre H0 kanıtlandı.", "Test, H0'ı kanıtlamaz.", "Doğru dil: H0 reddedilemez."),
        ("Tek taraflı test daha küçük p verdiği için sonuç görüldükten sonra seçilebilir.", "Yön veriye bakıldıktan sonra seçilirse hata riski değişir.", "Test yönü araştırma sorusu veya teoriyle önceden belirlenir."),
        ("Küçük standart hata modelin doğru kurulduğunu kanıtlar.", "Kesinlik, model spesifikasyonunun doğruluğunu kanıtlamaz.", "Küçük SH tahminin örnekleme belirsizliğini özetler; model ve nedensellik varsayımları ayrıdır."),
    )
    for number, (wrong, why, better) in enumerate(misconceptions, start=1):
        with st.expander(f"{number}. Yanlış ifade: {wrong}"):
            st.markdown(f"**Neden yanlış?** {why}\n\n**Daha doğru ifade:** {better}")

    st.subheader("Sonucu bütünleşik biçimde raporla")
    relation = "üzerindedir" if inference.t_statistic >= 0 else "altındadır"
    distance = abs(inference.t_statistic)
    report_interval = coefficient_confidence_interval(result, coefficient, confidence_level=0.95)
    st.write(
        f"{spec.title} modelinde {_label(spec, coefficient)} için ceteris paribus katsayı tahmini "
        f"{_display_number(inference.estimate, 4)}'tür. H0: β={_display_number(null_value, 3)} hipotezinde "
        f"t={_display_number(inference.t_statistic, 3)}; tahmin null değerin yaklaşık {_display_number(distance, 3)} standart hata {relation}. "
        f"p={format_p_value(inference.p_value)} olduğundan α={alpha:.2f} için {'H0 reddedilir' if inference.reject_null else 'H0 reddedilemez'}. "
        f"%95 güven aralığı [{_display_number(report_interval[0], 4)}; {_display_number(report_interval[1], 4)}]'tür. "
        f"Seçilen {home_change} kare fit HPRICE1 farkı için tahmin edilen değişim {_display_number(scaled_home.scaled_estimate, 2)} bin ABD dolarıdır. "
        "Bu gözlemsel EKK modeli ilişkiyi özetler; tek başına nedensellik veya modelin eksiksizliği kanıtlanmış olmaz."
    )

    st.subheader("Kendini dene")
    context_id = f"{spec.model_id}:{coefficient}:{null_value}:{alternative}:{alpha}:{confidence_level}"
    context = Konu07QuestionContext(context_id, spec.model_id, result, inference, _label(spec, coefficient), variable_metadata(spec.dataset_key, spec.dependent).label, "birim")
    synchronize_question_state(st.session_state, context_id, "konu07")
    index_key, _, answer_key = question_state_keys("konu07")
    index = int(st.session_state.get(index_key, 0))
    question = generate_konu07_question(context, index)
    st.markdown(f"**Soru {index + 1}:** {question.prompt}")
    left, right = st.columns(2)
    if left.button("Cevabı göster", key="konu07_show_answer", type="primary", width="stretch"):
        reveal_answer(st.session_state, "konu07")
    if right.button("Yeni soru", key="konu07_next_question", width="stretch"):
        next_question(st.session_state, "konu07")
        st.rerun()
    if st.session_state.get(answer_key, False):
        st.success(f"Çözüm: {question.answer}")
    else:
        st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")
    st.info("Özet: katsayı ile belirsizliği birlikte okuyun; küçük p nedensellik ya da iktisadi büyüklük değildir. Birden fazla katsayının ortak testi Konu 08'e, dayanıklı standart hatalar Konu 12'ye bırakılmıştır.")
