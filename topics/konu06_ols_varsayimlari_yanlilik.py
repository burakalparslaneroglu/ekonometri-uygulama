"""Konu 06: EKK varsayımları, yansızlık ve model sorunları."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.assumption_diagnostics_utils import (
    DEFAULT_SIMULATION_SEED,
    NEAR_COLLINEARITY_SCENARIOS,
    calculate_vif,
    coefficient_sensitivity,
    exact_collinearity_demo,
    generate_near_collinearity_data,
    generate_synthetic_ovb_data,
    ovb_bias,
    simulate_collinearity_variability,
    simulate_repeated_ols,
    synthetic_ovb_decomposition,
    wage1_ovb_decomposition,
)
from core.data_registry import get_dataset_metadata, load_dataset, variable_metadata
from core.konu06_questions import Konu06QuestionContext, generate_konu06_question
from core.model_utils import format_number, format_numerical_difference, format_student_number
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state


@st.cache_data(show_spinner=False)
def _wage_data() -> pd.DataFrame:
    """WAGE1 verisini önbellekli yükler."""
    return load_dataset("wage1")


@st.cache_data(show_spinner=False)
def _repeated(scenario_id: str, nobs: int, repetitions: int, seed: int):
    """Ağır olmayan fakat tekrarlı benzetimi önbelleğe alır."""
    gamma = 0.0 if scenario_id == "sifir_kosullu_ortalama" else 0.8
    return simulate_repeated_ols(scenario_id=scenario_id, nobs=nobs, repetitions=repetitions, seed=seed, conditional_mean_loading=gamma)


@st.cache_data(show_spinner=False)
def _synthetic(seed: int):
    """Sentetik OVB ayrıştırmasını önbelleğe alır."""
    return synthetic_ovb_decomposition(generate_synthetic_ovb_data(seed=seed))


@st.cache_data(show_spinner=False)
def _wage_decomposition():
    """WAGE1 ortak örneklem ayrıştırmasını önbelleğe alır."""
    return wage1_ovb_decomposition(_wage_data())


@st.cache_data(show_spinner=False)
def _near(scenario_id: str, seed: int):
    """Bağlantı DGP'sini önbelleğe alır."""
    return generate_near_collinearity_data(scenario_id, seed=seed)


@st.cache_data(show_spinner=False)
def _variability(scenario_id: str, seed: int):
    """Bağlantı değişkenliği benzetimini önbelleğe alır."""
    return simulate_collinearity_variability(scenario_id, repetitions=1000, seed=seed)


def _distribution_figure(result) -> go.Figure:
    """Tahmin dağılımını gerçek parametre ve senaryo merkeziyle çizer."""
    figure = px.histogram(x=result.estimates, nbins=35, labels={"x": "Tahmin edilen eğim", "y": "Tekrar sayısı"}, color_discrete_sequence=["#107C89"])
    counts, _ = np.histogram(result.estimates, bins=35)
    top = max(int(counts.max()), 1) * 1.05

    def add_reference(value: float, name: str, color: str, dash: str) -> None:
        figure.add_scatter(
            x=[value, value], y=[0, top], mode="lines", name=name,
            line={"color": color, "dash": dash, "width": 3}, hovertemplate=f"{name}<extra></extra>",
        )

    if np.isclose(result.true_slope, result.target_center, atol=1e-10, rtol=0):
        add_reference(result.true_slope, f"Gerçek eğim = hedef merkez: {format_student_number(result.true_slope)}", "#B3392F", "dash")
    else:
        add_reference(result.true_slope, f"Gerçek yapısal eğim: {format_student_number(result.true_slope)}", "#B3392F", "dash")
        add_reference(result.target_center, f"Tahmin dağılımının hedef merkezi: {format_student_number(result.target_center)}", "#2F9E6B", "dot")
    add_reference(result.mean_estimate, f"Ortalama tahmin: {format_student_number(result.mean_estimate)}", "#07373D", "solid")
    return figure.update_layout(template="plotly_white", height=330, margin={"l": 10, "r": 10, "t": 40, "b": 10}, showlegend=True, legend_title_text="Referans çizgileri")


def _assumption_cards() -> None:
    """Dört temel varsayımı erişilebilir kartlarla gösterir."""
    cards = st.columns(4)
    content = (
        ("A1", "Parametrelerde doğrusallık", "Bilinmeyen β parametreleri doğrusal yer alır."),
        ("A2", "Rassal örnekleme", "Yatay kesit gözlemleri hedef anakütleden uygun biçimde seçilir."),
        ("A3", "Tam bağlantı yok", "Açıklayıcı değişkenler birbirinin tam doğrusal tekrarı değildir."),
        ("A4", "Sıfır koşullu ortalama", "Hata teriminin koşullu ortalaması sıfırdır."),
    )
    for column, (code, title, text) in zip(cards, content):
        column.markdown(f"<div class='lesson-note'><strong>{code} — {title}</strong><br>{text}</div>", unsafe_allow_html=True)


def _vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF çıktısını öğrenciye açık etiketlerle dönüştürür."""
    raw = calculate_vif(frame, ("x1", "x2"))
    display = raw.copy()
    display["auxiliary_r_squared"] = display["auxiliary_r_squared"].map(format_student_number)
    display["vif"] = display["vif"].map(lambda value: "Hesaplanamaz" if not np.isfinite(value) else format_student_number(value))
    display["exact_collinearity"] = display["exact_collinearity"].map({True: "Evet", False: "Hayır"})
    return display.rename(columns={"variable": "Değişken", "auxiliary_r_squared": "Yardımcı R-kare", "vif": "VIF", "exact_collinearity": "Tam bağlantı var mı?"})


def _wage_table(wage) -> pd.DataFrame:
    """WAGE1 model karşılaştırmasını öğrenciye açık hücrelerle üretir."""
    missing = "—"
    rows = [
        ("Sabit terim", wage.short_model.intercept, float(wage.middle_model.coefficients["const"]), float(wage.full_model.coefficients["const"])),
        ("Eğitim", wage.short_model.slope, float(wage.middle_model.coefficients["educ"]), float(wage.full_model.coefficients["educ"])),
        ("Deneyim", None, float(wage.middle_model.coefficients["exper"]), float(wage.full_model.coefficients["exper"])),
        ("Kıdem", None, None, float(wage.full_model.coefficients["tenure"])),
        ("R-kare", wage.short_model.r_squared, wage.middle_model.r_squared, wage.full_model.r_squared),
        ("Gözlem sayısı", wage.nobs, wage.nobs, wage.nobs),
    ]
    def cell(value: float | int | None) -> str:
        if value is None:
            return missing
        if isinstance(value, (int, np.integer)):
            return str(value)
        return format_student_number(float(value), decimals=4)
    return pd.DataFrame([
        {"Satır": name, "Yalnız eğitim": cell(short), "Eğitim + deneyim": cell(middle), "Eğitim + deneyim + kıdem": cell(full)}
        for name, short, middle, full in rows
    ])


def _sensitivity_table(sensitivity) -> pd.DataFrame:
    """Duyarlılık sonucunu değişim sütunuyla, salt-okunur metin olarak sunar."""
    original = {
        "X1 katsayısı": sensitivity.original_coefficients["x1"],
        "X2 katsayısı": sensitivity.original_coefficients["x2"],
        "X1 + X2 katsayı toplamı": sum(sensitivity.original_coefficients.values()),
        "R-kare": sensitivity.original_r_squared,
        "VIF": float(sensitivity.original_vif["vif"].iloc[0]),
        "Rank": f"{sensitivity.original_rank.rank}/{sensitivity.original_rank.columns} (tam rank: {'Evet' if sensitivity.original_rank.full_rank else 'Hayır'})",
    }
    changed = {
        "X1 katsayısı": sensitivity.perturbed_coefficients["x1"],
        "X2 katsayısı": sensitivity.perturbed_coefficients["x2"],
        "X1 + X2 katsayı toplamı": sum(sensitivity.perturbed_coefficients.values()),
        "R-kare": sensitivity.perturbed_r_squared,
        "VIF": float(sensitivity.perturbed_vif["vif"].iloc[0]),
        "Rank": f"{sensitivity.perturbed_rank.rank}/{sensitivity.perturbed_rank.columns} (tam rank: {'Evet' if sensitivity.perturbed_rank.full_rank else 'Hayır'})",
    }
    rows = []
    for label in original:
        if label == "Rank":
            change = "Tam rank korunur" if sensitivity.original_rank.full_rank and sensitivity.perturbed_rank.full_rank else "Rank değişti"
            rows.append({"Büyüklük": label, "Özgün veri": original[label], "Küçük değişiklik sonrası": changed[label], "Değişim": change})
        else:
            difference = float(changed[label]) - float(original[label])
            rows.append({"Büyüklük": label, "Özgün veri": format_student_number(float(original[label]), decimals=4), "Küçük değişiklik sonrası": format_student_number(float(changed[label]), decimals=4), "Değişim": format_student_number(difference, decimals=6)})
    return pd.DataFrame(rows)


def render() -> None:
    """Konu 06 öğretim akışını ders notu sırasıyla gösterir."""
    st.markdown("<span class='topic-badge'>KONU 06</span>", unsafe_allow_html=True)
    st.header("EKK Varsayımları, Yansızlık ve Model Sorunları")
    st.markdown("<div class='lesson-note'><strong>Geçiş:</strong> Konu 05'te çoklu regresyon katsayılarını diğer açıklayıcı değişkenler sabitken yorumladınız. Bu konuda katsayı hesaplamak ile anakütle için güvenilir yorum yapmak arasındaki varsayım köprüsünü kuruyoruz.</div>", unsafe_allow_html=True)
    st.info("Bu konu standart hata, t istatistiği, p-değeri ve güven aralığını aktif olarak kullanmaz; bunlar Konu 07'de ele alınacaktır.")
    st.warning("“Regresyon çalıştı” ifadesi, sıfır koşullu ortalamanın sağlandığını veya modelin nedensel yorum taşıdığını göstermez.")

    st.subheader("Dört temel varsayım ve farklı rolleri")
    _assumption_cards()
    st.latex(r"Y_i=\beta_0+\beta_1X_{1i}+\cdots+\beta_kX_{ki}+u_i\qquad E(u_i\mid X_{1i},\ldots,X_{ki})=0")
    st.write("A1'de X² içeren bir model parametrelerde doğrusal olabilir. A2 örneklemin hedef anakütleyle ilişkisini, A3 benzersiz katsayı hesaplanmasını, A4 ise yansızlığın merkezî koşulunu ilgilendirir.")
    st.caption("Homoskedastisite ve normallik yansızlık için gerekli değildir; çıkarım konularında daha sonra ele alınacaktır.")

    st.subheader("Tahmin, tahmin edici ve yansızlık")
    st.latex(r"E(\widehat\beta_j)=\beta_j\qquad \operatorname{Bias}(\widehat\beta_j)=E(\widehat\beta_j)-\beta_j")
    st.write("Bir örneklemdeki β̂ bir tahmindir. Yansızlık ise aynı veri üretim sürecinden tekrar tekrar örneklem alındığında tahmin edicinin merkezinin β çevresinde olmasıdır; her tekil tahminin gerçek değere eşit olması değildir.")

    st.subheader("Tekrarlı örnekleme laboratuvarı")
    first, second, third = st.columns(3)
    scenario = first.selectbox("Sıfır koşullu ortalama senaryosu", ("sifir_kosullu_ortalama", "bozulmus_sifir_kosullu_ortalama"), format_func=lambda value: "Sıfır koşullu ortalama" if value == "sifir_kosullu_ortalama" else "Bozulmuş sıfır koşullu ortalama", key="konu06_sampling_scenario")
    nobs = second.selectbox("Örneklem büyüklüğü", (40, 80, 200), index=1, key="konu06_sampling_nobs")
    repetitions = third.selectbox("Tekrar sayısı", (500, 1000, 3000), index=2, key="konu06_sampling_repetitions")
    repeated = _repeated(scenario, nobs, repetitions, DEFAULT_SIMULATION_SEED)
    metrics = st.columns(4)
    metrics[0].metric("Gerçek eğim β₁", format_number(repeated.true_slope))
    metrics[1].metric("Tahmin ortalaması", format_number(repeated.mean_estimate))
    metrics[2].metric("Tahmin ortalaması − β₁", format_number(repeated.bias))
    metrics[3].metric("Tahminlerin std. sapması", format_number(repeated.estimate_std))
    st.plotly_chart(_distribution_figure(repeated), width="stretch")
    if np.isclose(repeated.true_slope, repeated.target_center, atol=1e-10, rtol=0):
        st.caption("Bu senaryoda sıfır koşullu ortalama sağlandığı için gerçek eğim ile tahmin dağılımının hedef merkezi aynıdır. Ortalama tahmin ayrıca gösterilir.")
    else:
        st.caption("Gerçek yapısal eğim 1,5'tir; ancak sıfır koşullu ortalama bozulduğu için EKK eğim tahminleri yaklaşık 2,3 çevresinde toplanır. Ortalama tahmin üçüncü çizgidir.")
    st.warning("Benzetim varsayımın mantığını gösterir; gerçek veride gözlenmeyen hata terimi nedeniyle A4'ü doğrudan kanıtlamaz.")

    st.subheader("Eksik değişken yanlılığı: iki bağlantının çarpımı")
    st.latex(r"\operatorname{Bias}(\widehat{\widetilde\beta}_1)=\beta_2\delta_1\qquad E(\widehat{\widetilde\beta}_1)=\beta_1+\beta_2\delta_1")
    st.write("Dışlanan Z'nin hem Y ile ilişkili olması (β₂≠0) hem de modelde tutulan X ile ilişkili olması (δ₁≠0) gerekir.")
    inputs = st.columns(3)
    target = inputs[0].number_input("Hedef X katsayısı (β₁)", value=1.2, key="konu06_ovb_target")
    omitted = inputs[1].number_input("Dışlanan Z'nin Y katsayısı (β₂)", value=-2.0, key="konu06_ovb_omitted")
    auxiliary = inputs[2].number_input("Z ~ X yardımcı eğimi (δ₁)", value=0.4, key="konu06_ovb_auxiliary")
    ovb = ovb_bias(target, omitted, auxiliary)
    calculation = pd.DataFrame([{"Yanlılık katkısı β₂δ₁": ovb.bias, "Kısa katsayı merkezi": ovb.expected_short_coefficient, "Yön": ovb.direction}])
    st.dataframe(calculation.round(4), hide_index=True, width="stretch")

    synthetic = _synthetic(DEFAULT_SIMULATION_SEED)
    st.markdown("**Sentetik kısa–uzun–yardımcı regresyon uygulaması**")
    st.latex(r"Z_i=0.7X_i+v_i,\qquad Y_i=1+2X_i+3Z_i+u_i")
    synthetic_table = pd.DataFrame([
        {"Büyüklük": "Yardımcı eğim δ̂₁", "Değer": synthetic.auxiliary_slope},
        {"Büyüklük": "Beklenen OVB katkısı", "Değer": synthetic.predicted_bias},
        {"Büyüklük": "Beklenen kısa X eğimi", "Değer": synthetic.expected_short_slope},
        {"Büyüklük": "Tahmin edilen kısa X eğimi", "Değer": synthetic.short_model.slope},
        {"Büyüklük": "Uzun model X katsayısı", "Değer": float(synthetic.long_model.coefficients["x"])},
        {"Büyüklük": "Uzun model Z katsayısı", "Değer": float(synthetic.long_model.coefficients["z"])},
        {"Büyüklük": "Ayrıştırma farkı", "Değer": format_numerical_difference(synthetic.decomposition_error)},
    ])
    numeric_columns = synthetic_table.select_dtypes(include="number").columns
    synthetic_table.loc[:, numeric_columns] = synthetic_table.loc[:, numeric_columns].round(5)
    st.dataframe(synthetic_table, hide_index=True, width="stretch")
    st.caption("5.000 gözlem kısa modelin yanlış hedefini düzeltmez; mekanizmayı daha görünür yapar.")

    st.subheader("WAGE1: eğitim katsayısının ortak örneklem ayrıştırması")
    wage = _wage_decomposition()
    metadata = get_dataset_metadata("wage1")
    with st.expander("Veri kaynağı ve değişkenler", expanded=False):
        st.markdown(f"**Kaynak:** {metadata.source}")
        st.write(f"**Gözlem birimi:** {metadata.observation_unit}. {metadata.description}")
        st.dataframe(pd.DataFrame([{"Değişken": name, "Açıklama": variable_metadata("wage1", name).description, "Ölçü birimi": variable_metadata("wage1", name).unit} for name in ("wage", "educ", "exper", "tenure")]), hide_index=True, width="stretch")
    st.dataframe(_wage_table(wage), hide_index=True, width="stretch")
    contribution = float(wage.middle_model.coefficients["exper"]) * wage.auxiliary_model.slope
    st.write(f"Yardımcı regresyon: deneyim ~ eğitim eğimi = **{format_number(wage.auxiliary_model.slope)}**. Ayrıştırma: **{format_number(wage.short_model.slope)} = {format_number(float(wage.middle_model.coefficients['educ']))} + ({format_number(float(wage.middle_model.coefficients['exper']))} × {format_number(wage.auxiliary_model.slope)}) = {format_number(float(wage.middle_model.coefficients['educ']))} + {format_number(contribution)}**.")
    st.caption(f"WAGE1 ayrıştırma farkı: {format_numerical_difference(wage.decomposition_error)}. Eşitlik teorik olarak geçerlidir; tolerans kayan nokta hesabını denetler. Katsayı değişimi tek başına doğru model veya nedensellik kanıtı değildir.")

    st.subheader("Tam ve yüksek fakat tam olmayan çoklu doğrusal bağlantı")
    demo_id = st.selectbox("Tam bağlantı senaryosu", ("monthly_annual_income", "total_and_components", "exact_linear_combination", "high_but_not_exact"), format_func=lambda value: {"monthly_annual_income": "Aylık ve yıllık gelir", "total_and_components": "Toplam ve bileşenler", "exact_linear_combination": "Tam doğrusal birleşim", "high_but_not_exact": "Yüksek fakat tam olmayan bağlantı"}[value], key="konu06_exact_scenario")
    demo = exact_collinearity_demo(demo_id)
    if demo.rank_result.full_rank:
        st.success(f"Rank = {demo.rank_result.rank}/{demo.rank_result.columns}. Tam rank: Evet. Tam doğrusal bağlantı var mı?: Hayır. {demo.explanation}")
    else:
        st.error(f"Rank = {demo.rank_result.rank}/{demo.rank_result.columns}. Tam rank: Hayır. Tam doğrusal bağlantı var mı?: Evet. Tam bağlantı nedeniyle benzersiz katsayı tahminine geçilmez. {demo.explanation}")

    st.subheader("Yüksek fakat tam olmayan bağlantı, korelasyon ve varyans şişirme faktörü (VIF)")
    near_id = st.selectbox("Bağlantı düzeyi", tuple(NEAR_COLLINEARITY_SCENARIOS), format_func=lambda value: NEAR_COLLINEARITY_SCENARIOS[value][0], key="konu06_near_scenario")
    near = _near(near_id, DEFAULT_SIMULATION_SEED)
    vif = _vif_table(near.data)
    figure = px.scatter(near.data, x="x1", y="x2", labels={"x1": "X₁", "x2": "X₂"}, color_discrete_sequence=["#107C89"])
    figure.update_layout(template="plotly_white", height=330, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(figure, width="stretch")
    sample_corr = float(near.data[["x1", "x2"]].corr().iloc[0, 1])
    st.dataframe(vif, hide_index=True, width="stretch")
    st.caption(f"Örneklem korelasyonu: {sample_corr:.6f}; teorik korelasyon: {near.theoretical_correlation:.6f}; teorik VIF: {near.theoretical_vif:.2f}. {'Yuvarlanmış görünüm 1.000 olsa bile ham korelasyon tam olarak bir değildir.' if round(sample_corr, 3) == 1 else ''}")
    variability = _variability(near_id, DEFAULT_SIMULATION_SEED)
    variability_table = pd.DataFrame([{"Ortalama korelasyon": variability.mean_correlation, "Ortalama VIF": variability.mean_vif, "β̂₁ ortalaması": variability.beta1_mean, "β̂₂ ortalaması": variability.beta2_mean, "Toplam ortalaması": variability.beta_sum_mean, "β̂₁ tahminlerinin std. sapması": variability.beta1_std, "β̂₂ tahminlerinin std. sapması": variability.beta2_std, "Toplam tahmininin std. sapması": variability.beta_sum_std}])
    st.dataframe(variability_table.round(4), hide_index=True, width="stretch")
    st.caption("Bu satırlardaki standart sapmalar tekrarlı örneklerde tahminlerin standart sapmasıdır; tek örneklemden hesaplanan standart hata değildir.")
    sensitivity = coefficient_sensitivity(_near("near_exact", DEFAULT_SIMULATION_SEED).data)
    st.markdown("**Küçük veri değişikliğine duyarlılık**")
    st.dataframe(_sensitivity_table(sensitivity), hide_index=True, width="stretch")
    st.caption("Ortak tahmin uyumu neredeyse değişmezken ayrı katsayılar daha fazla değişebilir. Bu, verinin ortak katkıyı ayrı katkılardan daha iyi belirlediğini gösterir; her küçük veri değişikliği mutlaka büyük katsayı değişimi üretmez.")

    st.subheader("Model kararını araştırma sorusuna bağlamak")
    st.dataframe(pd.DataFrame([
        {"Konu": "Eksik değişken yanlılığı", "Merkez": "Yanlış değere kayabilir", "Ana soru": "Dışlanan unsur hem Y hem X ile ilişkili mi?"},
        {"Konu": "Yüksek fakat tam olmayan bağlantı", "Merkez": "Sıfır koşullu ortalama altında doğru kalabilir", "Ana soru": "Ayrı katkılar veri tarafından ayırt edilebiliyor mu?"},
    ]), hide_index=True, width="stretch")
    st.write("Sıfır koşullu ortalama varsayımı (A4) sağlanıyorsa katsayı tahminlerinin merkezi doğru kalabilir; ancak ayrı katsayı tahminleri daha değişken ve veri değişikliklerine daha duyarlı olabilir. Bu nedenle yüksek çoklu doğrusal bağlantı otomatik olarak yanlılık anlamına gelmez.")
    st.write("Karar verirken araştırma sorusunu, teoriyi, dışarıda kalan olası faktörleri, veri aralığını, korelasyon/VIF'yi ve ekonomik olarak savunulabilir modellerdeki duyarlılığı birlikte okuyun. Yüksek VIF otomatik “sil” emri değildir.")
    st.markdown("1. Araştırma sorusu ve hedef katsayı nedir?  \n2. Hangi kontroller vardır, hangileri dışarıda kalmıştır?  \n3. Dışlanan faktör hem Y hem temel X ile ilişkili olabilir mi?  \n4. Açıklayıcılar aynı bilgiyi tekrar ediyor mu?  \n5. Sonuç ilişki diliyle mi yazılmalıdır?")

    context_id = f"{scenario}:{nobs}:{repetitions}:{target}:{omitted}:{auxiliary}:{demo_id}:{near_id}"
    context = Konu06QuestionContext(context_id, repeated, ovb, wage, float(vif["VIF"].iloc[0]), not demo.rank_result.full_rank, variability)
    topic_id = "konu06"
    synchronize_question_state(st.session_state, context_id, topic_id)
    index_key, _, answer_key = question_state_keys(topic_id)
    index = int(st.session_state.get(index_key, 0))
    question = generate_konu06_question(context, index)
    st.subheader("Kendini dene")
    st.markdown(f"**Soru {index + 1}:** {question.prompt}")
    left, right = st.columns(2)
    if left.button("Cevabı göster", key="konu06_show_answer", type="primary", width="stretch"):
        reveal_answer(st.session_state, topic_id)
    if right.button("Yeni soru", key="konu06_next_question", width="stretch"):
        next_question(st.session_state, topic_id)
        st.rerun()
    if st.session_state.get(answer_key, False):
        st.success(f"**Çözüm:** {question.answer}")
    else:
        st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")
