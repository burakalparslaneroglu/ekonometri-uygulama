"""Konu 08: ortak F testleri ve büyük örneklem mantığı arayüzü."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_registry import get_dataset_metadata, load_dataset, variable_metadata
from core.joint_inference_utils import (LinearRestriction, classify_large_sample_scenario, f_distribution_plot_data,
    classify_restriction_system, format_linear_restriction, joint_f_test, nested_exclusion_f_test, overall_f_test,
    simulate_large_sample_joint_test, single_restriction_equivalence, validate_restriction_system)
from core.konu08_questions import Konu08QuestionContext, generate_konu08_question
from core.regression_inference_utils import coefficient_test, fit_ols_inference, format_p_value
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state
from core.ui_preferences import plot_font_sizes


BRAND = {"ink": "#07373D", "teal": "#107C89", "bright": "#15A4B5", "green": "#2F9E6B", "red": "#B3392F"}


@st.cache_data(show_spinner=False)
def _tests() -> dict[str, object]:
    """Sabit WAGE1 ve HPRICE1 ortak testlerini önbelleğe alır."""
    wage = load_dataset("wage1"); hprice = load_dataset("hprice1")
    return {
        "wage": nested_exclusion_f_test(wage, "wage", ("educ", "exper", "tenure"), ("educ",)),
        "hprice": nested_exclusion_f_test(hprice, "price", ("lotsize", "sqrft", "bdrms"), ("sqrft",)),
        "wage_result": fit_ols_inference(wage, "wage", ("educ", "exper", "tenure")),
    }


@st.cache_data(show_spinner=False)
def _simulation(nobs: int, repetitions: int):
    """Ağır batch benzetimini UI seçimiyle önbelleğe alır."""
    return simulate_large_sample_joint_test(sample_sizes=(nobs,), repetitions=repetitions, seed=202508)


def _f_figure(statistic: float, df_num: int, df_denom: int, alpha: float, scale: float) -> go.Figure:
    """Üst kuyruk F grafiğini kurar."""
    data = f_distribution_plot_data(statistic, df_num, df_denom, alpha)
    fig = go.Figure(go.Scatter(x=data["x"], y=data["density"], line={"color": BRAND["ink"]}, name="F yoğunluğu"))
    x, density = data["x"], data["density"]
    mask = data["critical_tail_mask"]
    fig.add_trace(go.Scatter(x=x[mask], y=density[mask], fill="tozeroy", line={"color": BRAND["bright"]}, fillcolor="rgba(21,164,181,.22)", name="Red bölgesi"))
    fig.add_vline(x=data["critical_value"], line_dash="dash", line_color=BRAND["red"], annotation_text="kritik F")
    if data["observed_outside"]:
        fig.add_annotation(text=f"Gözlenen F={statistic:.2f}, çizim aralığının sağında", xref="paper", yref="paper", x=0.98, y=0.9, showarrow=False)
    else:
        fig.add_vline(x=statistic, line_color=BRAND["green"], annotation_text="gözlenen F")
    fonts = plot_font_sizes(scale)
    return fig.update_layout(template="plotly_white", height=330, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                             font={"size": fonts["base"]}, hoverlabel={"font_size": fonts["hover"]},
                             legend={"font": {"size": fonts["legend"]}},
                             xaxis={"title": {"text": "F istatistiği", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}},
                             yaxis={"title": {"text": "Yoğunluk", "font": {"size": fonts["title"]}}, "tickfont": {"size": fonts["tick"]}})


def _preset_rows(preset: str, coefficient_names: tuple[str, ...]) -> tuple[tuple[dict[str, float], float], ...]:
    """Güvenli custom kısıt hazır örneklerini model katsayı düzeninde döndürür."""
    zeros = {name: 0.0 for name in coefficient_names}
    def row(values: dict[str, float], rhs: float = 0.0) -> tuple[dict[str, float], float]:
        return ({**zeros, **values}, rhs)
    presets = {
        "exper = 0": (row({"exper": 1.0}),),
        "exper = 0 ve tenure = 0": (row({"exper": 1.0}), row({"tenure": 1.0})),
        "exper = tenure": (row({"exper": 1.0, "tenure": -1.0}),),
        "exper + tenure = 0.20": (row({"exper": 1.0, "tenure": 1.0}, 0.20),),
        "educ = 0.50": (row({"educ": 1.0}, 0.50),),
        "bütün eğimler = 0": tuple(row({name: 1.0}) for name in coefficient_names if name != "const"),
        "yinelenen/geçersiz kısıt örneği": (row({"exper": 1.0}), row({"exper": 2.0})),
    }
    return presets[preset]


def _render_custom_restriction_lab(alpha: float, scale: float) -> str:
    """Tablo tabanlı R ve r oluşturucu ile güvenli custom F laboratuvarını render eder."""
    st.subheader("Kullanıcı tanımlı kısıt laboratuvarı")
    st.caption("Serbest matematik parser'ı yerine R matrisi ve r vektörünü satır satır kurun. Teknik geçerlilik, araştırma sorusunun ekonomik gerekçesinden ayrı denetlenir.")
    model_id = st.selectbox("Custom kısıt modeli", ("WAGE1", "HPRICE1"), key="konu08_custom_model")
    specs = {
        "WAGE1": ("wage1", "wage", ("educ", "exper", "tenure")),
        "HPRICE1": ("hprice1", "price", ("lotsize", "sqrft", "bdrms")),
    }
    dataset_key, dependent, explanatory = specs[model_id]
    result = fit_ols_inference(load_dataset(dataset_key), dependent, explanatory)
    names = tuple(result.coefficients.index)
    preset_options = ("exper = 0", "exper = 0 ve tenure = 0", "exper = tenure", "exper + tenure = 0.20", "educ = 0.50", "bütün eğimler = 0", "yinelenen/geçersiz kısıt örneği") if model_id == "WAGE1" else ("bütün eğimler = 0",)
    preset = st.selectbox("Hazır örnek", preset_options, key=f"konu08_custom_preset_{model_id}")
    active_key = f"konu08_active_preset_{model_id}"
    if st.session_state.get(active_key) != preset:
        rows = _preset_rows(preset, names)
        st.session_state["konu08_custom_rows"] = len(rows)
        for row_index, (weights, rhs) in enumerate(rows):
            for name in names:
                st.session_state[f"konu08_custom_{row_index}_{name}"] = weights.get(name, 0.0)
            st.session_state[f"konu08_custom_{row_index}_rhs"] = rhs
        st.session_state[active_key] = preset
    row_count = st.selectbox("Girilen kısıt satırı", (1, 2, 3), key="konu08_custom_rows")
    st.caption("Satır sayısı doğrudan q değildir; q=rank(R), yani bağımsız kısıt sayısıdır.")
    header_columns = st.columns((1,) * (len(names) + 1))
    for column, name in zip(header_columns, (*names, "sağ taraf")):
        column.markdown(f"**{name}**")
    restrictions: list[LinearRestriction] = []
    for row_index in range(int(row_count)):
        columns = st.columns((1,) * (len(names) + 1))
        weights = {name: columns[position].number_input(name, min_value=-10.0, max_value=10.0, step=0.5, key=f"konu08_custom_{row_index}_{name}", label_visibility="collapsed") for position, name in enumerate(names)}
        rhs = columns[-1].number_input("sağ taraf", min_value=-100.0, max_value=100.0, step=0.1, key=f"konu08_custom_{row_index}_rhs", label_visibility="collapsed")
        restrictions.append(LinearRestriction(weights, rhs, f"Kısıt {row_index + 1}"))
    tuple_restrictions = tuple(restrictions)
    validation = validate_restriction_system(result, tuple_restrictions)
    classification = classify_restriction_system(result, tuple_restrictions)
    st.markdown("**Matematiksel yeniden yazım**")
    labels = {"const": "sabit", **{name: variable_metadata(dataset_key, name).label for name in explanatory}}
    st.code("H0: " + "; ".join(format_linear_restriction(item, labels=labels) for item in tuple_restrictions), language="text")
    r_frame = pd.DataFrame([{name: item.weights.get(name, 0.0) for name in names} | {"r (sağ taraf)": item.rhs} for item in tuple_restrictions], index=[f"Kısıt {index + 1}" for index in range(len(tuple_restrictions))])
    st.dataframe(r_frame, width="stretch")
    st.caption("Tablodaki katsayı ağırlıkları R matrisini, son sütun r vektörünü gösterir.")
    st.dataframe(pd.DataFrame([{"Teknik durum": "Teknik olarak geçerli" if validation.is_valid else "Teknik olarak geçersiz", "Girilen satır sayısı": validation.row_count, "Bağımsız kısıt sayısı q": validation.q, "rank(R)": validation.rank_r, "rank([R|r])": validation.rank_augmented, "Kısıt sınıfı": classification.restriction_type}]), hide_index=True, width="stretch")
    for message in validation.messages:
        (st.success if validation.is_valid else st.warning)(message)
    st.info("Araştırma sorusunu doğru temsil ediyor mu? Teknik geçerlilik yalnız matematiksel sistemi doğrular. Örneğin exper+tenure=0, exper=0 ve tenure=0 şeklindeki iki ayrı kısıtla aynı hipotez değildir.")
    if not validation.is_valid:
        st.error("Custom F testi çalıştırılmadı. Kısıtları bağımsız, bilgi taşıyan ve tutarlı hale getirin.")
        return f"{model_id}:geçersiz:{preset}:{row_count}"
    test = joint_f_test(result, tuple_restrictions, alpha=alpha)
    cols = st.columns(6)
    cols[0].metric("q", test.q); cols[1].metric("F", f"{test.f_statistic:.3f}"); cols[2].metric("Pay sd", test.df_num); cols[3].metric("Payda sd", test.df_denom); cols[4].metric("Kritik F", f"{test.critical_value:.3f}"); cols[5].metric("p", format_p_value(test.p_value))
    st.success("Girilen kısıtların tümü birlikte reddedilir; en az bir kısıt örneklem bilgisiyle uyumsuzdur." if test.reject_null else "Girilen kısıtların tümünü birlikte reddetmek için seçilen düzeyde yeterli örneklem kanıtı yoktur.")
    if classification.nested_exclusion_applicable:
        restricted = tuple(name for name in explanatory if name not in classification.excluded_coefficients)
        nested = nested_exclusion_f_test(load_dataset(dataset_key), dependent, explanatory, restricted, alpha=alpha)
        st.dataframe(pd.DataFrame([{"Uygun yöntem": "Genel matrix F", "F": test.f_statistic}, {"Uygun yöntem": "Nested SSR F", "F": nested.f_from_ssr}, {"Uygun yöntem": "Nested R² F", "F": nested.f_from_r_squared}]), hide_index=True, width="stretch")
        st.caption("Bu üç sonuç aynı uygun dışlama kısıtında eşittir. R=restricted, UR=unrestricted; pay sd=q, payda sd=n−kUR−1.")
    else:
        st.info("Bu kısıt, basit değişken dışlama kısıtı değildir. SSR/R² ile değişken silme yaklaşımı kullanılmaz; genel Rβ=r matrix F kullanılır.")
    return f"{model_id}:{preset}:{row_count}:{format_linear_restriction(tuple_restrictions[0])}"


def render() -> None:
    """Konu 08 ders sırasına uygun ortak F testi ekranını görüntüler."""
    tests = _tests(); wage_test = tests["wage"]; hprice_test = tests["hprice"]; wage_result = tests["wage_result"]
    scale = float(st.session_state.get("text_scale", 1.10))
    st.markdown("<span class='topic-badge'>KONU 08</span>", unsafe_allow_html=True)
    st.header("Birden Fazla Kısıtın Sınanması: F Testi ve Büyük Örneklem Mantığı")
    st.markdown("<div class='lesson-note'><strong>Geçiş:</strong> Konu 07 katsayıları tek tek sınadı. Burada deneyim ve kıdem birlikte katkı sağlıyor mu gibi ortak soruları, kısıtlı ve kısıtsız modelleri karşılaştırarak ele alıyoruz.</div>", unsafe_allow_html=True)
    st.warning("Bu bölüm yalnız geleneksel, homoskedastisiteye dayanan nonrobust F testlerini kullanır. Dayanıklı ortak testler Konu 12 kapsamındadır.")

    st.subheader("Tekli test ile ortak hipotez aynı değildir")
    st.dataframe(pd.DataFrame([
        {"Neden": "Ortak karar", "Açıklama": "İki ayrı t testi tek bir ortak yanlış karar olasılığı oluşturmaz."},
        {"Neden": "Birlikte hareket", "Açıklama": "Katsayı tahminleri birlikte değişebilir; F bunu ortak kovaryansla hesaba katar."},
        {"Neden": "Araştırma sorusu", "Açıklama": "Soru deneyim ve kıdemin birlikte katkısıysa hipotez de ortaktır."},
    ]), hide_index=True, width="stretch")
    choice = st.selectbox("Hipotezi sınıflandırın", ("β_exper = 0", "β_exper = 0 ve β_tenure = 0", "β_exper = β_tenure"), key="konu08_restriction")
    st.info("Tek katsayı" if choice == "β_exper = 0" else "Ortak hipotez: F testi için doğrusal kısıt sistemi.")

    st.subheader("F testinin genel haritası")
    st.latex(r"H_0:R\beta=r,\qquad q=\operatorname{rank}(R)")
    st.write("R, katsayıları nasıl birleştirdiğimizi; r ise sıfır hipotezindeki hedef değerleri gösterir. q, girilen satır sayısı değil bağımsız kısıt sayısıdır.")
    st.latex(r"F=\frac{(R\widehat\beta-r)'[R\widehat{\operatorname{Var}}(\widehat\beta)R']^{-1}(R\widehat\beta-r)}{q}")
    st.info("Öğrenci diliyle: Kısıtlar altında beklenen değerlerden sapmalar, tahmin belirsizliğine göre ne kadar büyüktür?")
    st.latex(r"F=\frac{(SSR_R-SSR_{UR})/q}{SSR_{UR}/(n-k_{UR}-1)}\qquad F=\frac{(R^2_{UR}-R^2_R)/q}{(1-R^2_{UR})/(n-k_{UR}-1)}")
    st.caption("R = kısıtlı model; UR = kısıtsız model; kUR = kısıtsız modeldeki eğim sayısı; pay sd=q; payda sd=n−kUR−1. SSR/R² biçimleri yalnız uygun iç içe dışlama kısıtlarında kullanılabilir. F negatif değildir, üst kuyruk testidir; büyük F kısıtlarla uyumsuzluğa işaret eder. p-değeri H0'ın doğru olma olasılığı değildir.")

    st.subheader("Her kısıt aynı türde değildir")
    st.dataframe(pd.DataFrame([
        {"Kısıt türü": "Dışlama", "Örnek": "βexper=βtenure=0", "Değişken silerek restricted model?": "Evet", "Genel Rβ=r?": "Evet"},
        {"Kısıt türü": "Eşitlik", "Örnek": "βexper=βtenure", "Değişken silerek restricted model?": "Hayır", "Genel Rβ=r?": "Evet"},
        {"Kısıt türü": "Sıfır dışı", "Örnek": "βeduc=0.50", "Değişken silerek restricted model?": "Hayır", "Genel Rβ=r?": "Evet"},
        {"Kısıt türü": "Doğrusal birleşim", "Örnek": "βexper+βtenure=0.20", "Değişken silerek restricted model?": "Hayır", "Genel Rβ=r?": "Evet"},
        {"Kısıt türü": "Genel anlamlılık", "Örnek": "Bütün eğimler=0", "Değişken silerek restricted model?": "Evet", "Genel Rβ=r?": "Evet"},
    ]), hide_index=True, width="stretch")

    st.subheader("WAGE1: deneyim ve kıdemin ortak testi")
    meta = get_dataset_metadata("wage1")
    st.caption(f"Kaynak: {meta.source} | Gözlem birimi: {meta.observation_unit} | {meta.description}")
    alpha = st.selectbox("Anlamlılık düzeyi α", (0.10, 0.05, 0.01), index=1, key="konu08_alpha")
    st.latex(r"H_0:\beta_{exper}=0,\ \beta_{tenure}=0\qquad H_1:\text{kısıtlardan en az biri doğru değildir}")
    values = st.columns(5)
    values[0].metric("q (pay sd)", wage_test.q); values[1].metric("Payda sd", wage_test.df_denom); values[2].metric("F (SSR)", f"{wage_test.f_from_ssr:.3f}"); values[3].metric("F (R²)", f"{wage_test.f_from_r_squared:.3f}"); values[4].metric("Ortak p", format_p_value(wage_test.p_value))
    st.dataframe(pd.DataFrame([
        {"Model": "Kısıtlı", "Açıklayıcılar": "educ", "SSR (artık kareleri toplamı)": wage_test.ssr_restricted, "R²": wage_test.r_squared_restricted},
        {"Model": "Kısıtsız", "Açıklayıcılar": "educ, exper, tenure", "SSR (artık kareleri toplamı)": wage_test.ssr_unrestricted, "R²": wage_test.r_squared_unrestricted},
    ]), hide_index=True, width="stretch")
    st.caption("SSR ve R² formülleri ile genel kısıt matrisi aynı F değerini verir; karşılaştırma aynı bağımlı değişken ve aynı complete-case örneklem üzerindedir.")
    st.plotly_chart(_f_figure(wage_test.f_from_ssr, wage_test.q, wage_test.df_denom, alpha, scale), width="stretch")
    st.success("H0 reddedilir: kısıtlardan en az biri veriyle uyumsuzdur." if wage_test.p_value < alpha else "H0 reddedilemez.")

    st.subheader("Özel ortak test, genel anlamlılık ve F=t²")
    overall = overall_f_test(wage_result, alpha=alpha)
    education_t = coefficient_test(wage_result, "educ", alpha=alpha)
    education_f = __import__("core.joint_inference_utils", fromlist=["joint_f_test"]).joint_f_test(wage_result, (LinearRestriction({"educ": 1.0}, 0.0, "educ = 0"),), alpha=alpha)
    cols = st.columns(3)
    cols[0].metric("Özel F: exper, tenure", f"{wage_test.f_from_ssr:.3f}")
    cols[1].metric("Genel F: tüm eğimler", f"{overall.f_statistic:.3f}")
    cols[2].metric("Eğitim: F ve t²", f"{education_f.f_statistic:.3f} / {education_t.t_statistic ** 2:.3f}")
    st.caption("Genel test sabit terimi kısıtlamaz. F=t² yalnız aynı null değerindeki iki taraflı t testi ve q=1 için geçerlidir: " + ("doğrulandı." if single_restriction_equivalence(education_t, education_f) else "denetlenmelidir."))

    custom_context_id = _render_custom_restriction_lab(alpha, scale)

    st.subheader("HPRICE1: ayrı ve ortak bulgular çelişmek zorunda değildir")
    st.write(f"H0: β_lotsize=0 ve β_bdrms=0 için F({hprice_test.q},{hprice_test.df_denom})={hprice_test.f_from_ssr:.3f}, p {format_p_value(hprice_test.p_value)}.")
    st.caption("Bir katsayının tek başına p-değeri büyükken ortak test reddedilebilir; ortak sonuç, tüm katsayıların ayrı ayrı anlamlı olduğunu söylemez.")

    st.subheader("Normal olmayan hata altında büyük örneklem")
    sim_cols = st.columns(2)
    n_sim = sim_cols[0].selectbox("Benzetim n", (25, 100, 500), index=1, key="konu08_sim_n")
    reps = sim_cols[1].selectbox("Tekrar sayısı", (500, 1000, 4000), index=1, key="konu08_sim_reps")
    simulation = _simulation(n_sim, reps)
    row = simulation.rejection_rates.index[0]
    metrics = st.columns(4)
    metrics[0].metric("Ret oranı", f"%{100 * simulation.rejection_rates.loc[row]:.2f}"); metrics[1].metric("Standartlaştırılmış eğim ortalaması", f"{simulation.standardized_slope_means.loc[row]:.3f}"); metrics[2].metric("Std. sapma", f"{simulation.standardized_slope_stds.loc[row]:.3f}"); metrics[3].metric("Çarpıklık", f"{simulation.standardized_slope_skewness.loc[row]:.3f}")
    st.caption("DGP'de hatalar sağa çarpıktır ve gerçek iki eğim sıfırdır. Büyük n, yaklaşık çıkarımın çalışmasına yardım edebilir; “n>30 ise her şey normaldir” evrensel kural değildir.")

    st.subheader("Büyük n neyi çözmez?")
    scenario = st.selectbox("Senaryo", ("omitted_ability", "voluntary_online_sample", "wrong_quadratic_form", "repeated_firms_dependence", "correct_design_more_n", "heteroskedastic_nonrobust"), format_func=lambda x: {"omitted_ability":"Eksik yetenek", "voluntary_online_sample":"Gönüllü çevrim içi örneklem", "wrong_quadratic_form":"Yanlış karesel biçim", "repeated_firms_dependence":"Tekrarlanan firma gözlemleri", "correct_design_more_n":"Doğru tasarımda daha çok n", "heteroskedastic_nonrobust":"Heteroskedastisite ve nonrobust SH"}[x], key="konu08_scenario")
    classified = classify_large_sample_scenario(scenario)
    st.dataframe(pd.DataFrame([{"Sorunu çözer mi?": "Evet" if classified["sorunu_cozer_mi"] else "Hayır", "Ne iyileşebilir?": classified["ne_iyilesebilir"], "Ne çözülmez?": classified["ne_cozulmez"]}]), hide_index=True, width="stretch")
    st.info(str(classified["guvenli_aciklama"]))

    st.subheader("Python çıktısı ve kendini dene")
    st.code('model.f_test("exper = 0, tenure = 0")  # özel ortak test', language="python")
    context = Konu08QuestionContext("wage-special:" + custom_context_id, wage_test)
    synchronize_question_state(st.session_state, context.context_id, "konu08")
    index_key, _, answer_key = question_state_keys("konu08")
    question = generate_konu08_question(context, int(st.session_state.get(index_key, 0)))
    st.write(question.prompt)
    buttons = st.columns(2)
    if buttons[0].button("Cevabı göster", key="konu08_show_answer", type="primary", width="stretch"):
        reveal_answer(st.session_state, "konu08")
    if buttons[1].button("Yeni soru", key="konu08_next_question", type="secondary", width="stretch"):
        next_question(st.session_state, "konu08"); st.rerun()
    if st.session_state.get(answer_key, False):
        st.success("Çözüm: " + question.answer)
