"""Konu 12: heteroskedastisite ve dayanıklı çıkarım."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from core.data_registry import get_dataset_metadata,load_dataset
from core.joint_inference_utils import LinearRestriction
from core.konu12_questions import Konu12QuestionContext,generate_konu12_question
from core.regression_inference_utils import fit_ols_inference,format_p_value
from core.robust_inference_utils import breusch_pagan_auxiliary_details,coefficient_interval_comparison,fit_robust_inference,heteroskedastic_pattern_simulation,heteroskedasticity_tests,residual_plot_data,robust_joint_test,simulate_heteroskedastic_coverage,simulation_slope_samples,white_auxiliary_design,white_test_details
from core.session_utils import question_state_keys,synchronize_question_state
from core.ui_components import format_null_decision,render_question_actions

@st.cache_data(show_spinner=False)
def _model(covariance_type:str):
    """HPRICE1 düzey modelini istenen HC türüyle önbelleğe alır."""
    return fit_robust_inference(load_dataset("hprice1").assign(lotsize1000=lambda d:d.lotsize/1000,sqrft100=lambda d:d.sqrft/100),"price",("lotsize1000","sqrft100","bdrms"),covariance_type=covariance_type)
@st.cache_data(show_spinner=False)
def _log_model(covariance_type:str):
    """HPRICE1 log modelini istenen HC türüyle önbelleğe alır."""
    return fit_robust_inference(load_dataset("hprice1"),"lprice",("llotsize","lsqrft","bdrms"),covariance_type=covariance_type)
@st.cache_data(show_spinner=False)
def _simulation(nobs:int,repetitions:int):
    """Vektörize kapsama benzetimini önbelleğe alır."""
    return simulate_heteroskedastic_coverage(nobs=nobs,repetitions=repetitions)
def render()->None:
    """Konu 12 öğretim ekranını gösterir."""
    st.markdown("<span class='topic-badge'>KONU 12</span>",unsafe_allow_html=True)
    st.header("Heteroskedastisite ve Dayanıklı Çıkarım")
    st.info("Heteroskedastisite koşullu hata varyansıyla ilgilidir. Sıfır koşullu ortalama geçerliyse EKK katsayılarını otomatik olarak yanlı yapmaz; geleneksel standart hata ve çıkarımı etkileyebilir.")
    st.caption(f"Veri kaynağı: {get_dataset_metadata('hprice1').source}")
    st.markdown("**Bu sayfada ne yapacağız?**  Koşullu varyans → Grafik tanısı → BP → White → HC0–HC3 → Ortak test → Benzetim → Raporlama")
    st.subheader("1. Koşullu ortalama ve koşullu varyans")
    pattern=st.selectbox("Varyans örüntüsü",("sabit","artan","azalan","U biçimli"),key="konu12_pattern")
    pattern_data=heteroskedastic_pattern_simulation(pattern=pattern)
    pattern_mean_plot=pattern_data.rename(columns={"x":"X","y":"Gözlenen Y","koşullu_ortalama":"Koşullu ortalama E(Y|X)"})
    st.scatter_chart(
        pattern_mean_plot,
        x="X",
        y=["Gözlenen Y","Koşullu ortalama E(Y|X)"],
        x_label="X",
        y_label="Y düzeyi",
    )
    pattern_variance_plot=pattern_data.rename(columns={"x":"X","hata_sd":"Hata standart sapması"})
    st.line_chart(
        pattern_variance_plot,
        x="X",
        y="Hata standart sapması",
        x_label="X",
        y_label="Hata standart sapması",
    )
    st.caption("Bütün seçeneklerde E(Y|X)=1+2X aynıdır. Değişen unsur Var(u|X)'tir; heteroskedastisite koşullu ortalama denklemi değildir.")
    st.dataframe(pd.DataFrame([{"Özellik":"EKK katsayı merkezi","Sıfır koşullu ortalama geçerliyse":"Tek başına bozulmaz"},{"Özellik":"Etkinlik","Sıfır koşullu ortalama geçerliyse":"Genellikle kaybolur"},{"Özellik":"Geleneksel SH/t/F/GA","Sıfır koşullu ortalama geçerliyse":"Güvenilmez olabilir"},{"Özellik":"HC çıkarımı","Sıfır koşullu ortalama geçerliyse":"Heteroskedastisiteye dayanıklıdır"}]),hide_index=True,width="stretch")

    cov=st.selectbox("Dayanıklı kovaryans türü",("HC1","HC0","HC2","HC3"),key="konu12_covariance")
    result=_model(cov); log_result=_log_model(cov); ols=result.ols; bp,white=heteroskedasticity_tests(ols); plots=residual_plot_data(ols)
    st.subheader("2. Artık grafikleri: görsel tanı, kanıt değil")
    diagnostic=st.selectbox("Tanı görünümü",("Artık–tahmin","Mutlak artık–tahmin","Ölçek–konum"),key="konu12_diagnostic")
    plot_labels={"Artık–tahmin":("artık","Artık (bin ABD doları)"),"Mutlak artık–tahmin":("mutlak_artık","Mutlak artık (bin ABD doları)"),"Ölçek–konum":("scale_location","√|studentize artık|")}
    column,label=plot_labels[diagnostic]
    diagnostic_plot=plots.rename(columns={"tahmin":"Tahmin edilen fiyat (bin ABD doları)",column:label})
    st.scatter_chart(
        diagnostic_plot,
        x="Tahmin edilen fiyat (bin ABD doları)",
        y=label,
        x_label="Tahmin edilen fiyat (bin ABD doları)",
        y_label=label,
    )
    st.caption("Huni biçimi değişen yayılıma işaret edebilir; fakat fonksiyonel biçim ve uç gözlemler de örüntü yaratabilir. Grafik tek başına formal kanıt değildir.")

    st.subheader("3. Breusch–Pagan testi: yardımcı regresyon laboratuvarı")
    alpha=st.select_slider("Karar için α",options=(.01,.05,.10),value=.05,key="konu12_alpha")
    bp_details=breusch_pagan_auxiliary_details(ols)
    st.latex(r"H_0: Var(u_i\mid X_i)=\sigma^2,\qquad LM=nR^2_{aux}\sim\chi^2_q")
    st.markdown("**Adımlar:** EKK modelini tahmin et → artıkların karesini hesapla → û²'yi açıklayıcılara regrese et → yardımcı R²'yi al → LM=nR² hesapla.")
    st.write("Yardımcı açıklayıcılar: "+", ".join(bp_details.auxiliary_variables))
    bp_cards=st.columns(5); bp_cards[0].metric("R²_aux",f"{bp_details.auxiliary_r_squared:.4f}"); bp_cards[1].metric("nR²",f"{bp_details.lm_manual:.3f}"); bp_cards[2].metric("Statsmodels LM",f"{bp_details.lm_statsmodels:.3f}"); bp_cards[3].metric("df",bp_details.df); bp_cards[4].metric("p",format_p_value(bp_details.lm_p_value))
    bp_reject=bp_details.lm_p_value<alpha; bp_decision=format_null_decision(bp_details.lm_p_value,alpha)
    st.success(f"Breusch–Pagan kararı (α={alpha:.2f}, %{100*alpha:.0f}): {bp_decision}; " + ("heteroskedastisite lehine kanıt vardır." if bp_reject else "bu sonuç homoskedastisitenin kanıtı değildir."))

    st.subheader("4. White testi: kareler ve çapraz terimler")
    white_design,squares,crosses=white_auxiliary_design(ols); white_details=white_test_details(ols)
    st.latex(r"\hat u_i^2=\delta_0+\sum_j\delta_jX_{ji}+\sum_j\theta_jX_{ji}^2+\sum_{j<k}\phi_{jk}X_{ji}X_{ki}+v_i")
    terms=st.columns(3); terms[0].write("**Düzey terimleri**\n\n"+", ".join(white_details.base_variables)); terms[1].write("**Kare terimleri**\n\n"+", ".join(squares)); terms[2].write("**Çapraz terimler**\n\n"+", ".join(crosses))
    st.dataframe(white_design.head(8),hide_index=True,width="stretch")
    white_cards=st.columns(5); white_cards[0].metric("Sütun",white_details.column_count); white_cards[1].metric("Rank",white_details.matrix_rank); white_cards[2].metric("nR²",f"{white_details.lm_manual:.3f}"); white_cards[3].metric("df",white_details.df); white_cards[4].metric("p",format_p_value(white_details.p_value))
    white_reject=white_details.p_value<alpha; white_decision=format_null_decision(white_details.p_value,alpha)
    st.success(f"White testi kararı (α={alpha:.2f}, %{100*alpha:.0f}): {white_decision}; " + ("heteroskedastisite lehine kanıt vardır." if white_reject else "bu sonuç homoskedastisitenin kanıtı değildir."))
    st.warning("White testi daha esnektir; ancak daha çok serbestlik derecesi tüketir ve yanlış fonksiyonel biçim de reddetmeye katkı sağlayabilir.")

    st.subheader("5. BP ve White: düzey–log karşılaştırması")
    log_bp,log_white=heteroskedasticity_tests(log_result.ols)
    diagnostic_table=pd.DataFrame([{"Model":"Düzey","BP LM":bp.lm_statistic,"BP p":format_p_value(bp.p_value),"White LM":white.lm_statistic,"White p":format_p_value(white.p_value)},{"Model":"Log","BP LM":log_bp.lm_statistic,"BP p":format_p_value(log_bp.p_value),"White LM":log_white.lm_statistic,"White p":format_p_value(log_white.p_value)}])
    st.dataframe(diagnostic_table.round(4),hide_index=True,width="stretch")
    st.caption("Düzey modelinin sonuç birimi bin ABD dolarıdır; log modelinin sonucu log fiyattır. Her iki tanı geleneksel EKK artıklarına dayanır.")
    st.caption("Testlerin reddedilmemesi homoskedastisitenin kanıtı değildir. Grafik, ekonomik biçim ve robust çıkarım birlikte değerlendirilir.")

    st.subheader("6. HC0–HC3 dayanıklı kovaryans laboratuvarı")
    st.info("Katsayılar aynı • fitted değerler aynı • R² aynı • kovaryans, SH, t, p ve güven aralığı değişebilir")
    source=load_dataset("hprice1").assign(lotsize1000=lambda d:d.lotsize/1000,sqrft100=lambda d:d.sqrft/100)
    comparison=coefficient_interval_comparison(source,"price",("lotsize1000","sqrft100","bdrms"))
    selected_coef=st.selectbox("Katsayı",tuple(comparison.coefficient),index=1,key="konu12_coefficient")
    selected_row=comparison.loc[comparison.coefficient==selected_coef].iloc[0]
    comparison_display=comparison[["coefficient","estimate","nonrobust_se","HC0_se","HC1_se","HC2_se","HC3_se","nonrobust_p",f"{cov}_p"]].copy()
    comparison_display=comparison_display.rename(columns={"coefficient":"Katsayı","estimate":"Tahmin (bin ABD doları)","nonrobust_se":"Geleneksel SH","HC0_se":"HC0 SH","HC1_se":"HC1 SH","HC2_se":"HC2 SH","HC3_se":"HC3 SH","nonrobust_p":"Geleneksel p",f"{cov}_p":f"{cov} p"})
    comparison_display["Geleneksel p"]=comparison_display["Geleneksel p"].map(lambda value:format_p_value(float(value)))
    comparison_display[f"{cov} p"]=comparison_display[f"{cov} p"].map(lambda value:format_p_value(float(value)))
    st.dataframe(comparison_display.round(4),hide_index=True,width="stretch")
    st.caption(f"Model: price ~ lotsize1000 + sqrft100 + bdrms. Sonuç birimi: bin ABD doları. Karşılaştırılan kovaryans türleri: geleneksel (nonrobust) ve {cov}.")
    st.write(f"{selected_coef}: aynı tahmin={selected_row.estimate:.4f}; geleneksel SH={selected_row.nonrobust_se:.4f}; {cov} SH={selected_row[f'{cov}_se']:.4f}; {cov} GA=[{selected_row[f'{cov}_lower']:.4f}, {selected_row[f'{cov}_upper']:.4f}].")

    st.subheader("7. HPRICE1 düzey modeli ayrıntılı uygulama")
    table=pd.DataFrame({"Katsayı":ols.coefficients.index,"EKK tahmini":ols.coefficients.values,"Geleneksel SH":ols.standard_errors.values,f"{cov} SH":result.standard_errors.values,"Geleneksel p": [format_p_value(float(v)) for v in ols.p_values_two_sided_zero],f"{cov} p":[format_p_value(float(v)) for v in result.p_values_two_sided_zero]})
    st.dataframe(table.round(4),hide_index=True,width="stretch")
    st.caption(f"Model: HPRICE1 düzey fiyat modeli; sonuç birimi bin ABD doları. Nokta tahminleri, fitted değerler, R² ve artıklar aynıdır. Yalnız geleneksel ve {cov} kovaryansları arasında standart hata ve çıkarım değişir.")
    st.info(f"lotsize1000 tahmini {ols.coefficients['lotsize1000']:.4f} değişmez; geleneksel p={format_p_value(float(ols.p_values_two_sided_zero['lotsize1000']))}, HC1 p={format_p_value(float(_model('HC1').p_values_two_sided_zero['lotsize1000']))}. Belirsizlik ve karar değişebilir; ekonomik büyüklük değişmez.")

    st.subheader("8. HPRICE1 log modeli ayrıntılı uygulama")
    log_table=pd.DataFrame({"Katsayı":log_result.ols.coefficients.index,"Tahmin":log_result.ols.coefficients.values,"Geleneksel SH":log_result.ols.standard_errors.values,f"{cov} SH":log_result.standard_errors.values,"Geleneksel p":[format_p_value(float(v)) for v in log_result.ols.p_values_two_sided_zero],f"{cov} p":[format_p_value(float(v)) for v in log_result.p_values_two_sided_zero]})
    st.dataframe(log_table.round(4),hide_index=True,width="stretch"); st.caption(f"Model: HPRICE1 log fiyat modeli; kovaryans türleri geleneksel ve {cov}. llotsize ve lsqrft katsayıları esnekliktir. Bağımlı değişkenler farklı olduğu için düzey ve log model R² değerleri mekanik model seçimi için doğrudan karşılaştırılmaz.")

    st.subheader("9. Geleneksel ve dayanıklı ortak test")
    joint=robust_joint_test(result,(LinearRestriction({"lotsize1000":1},0,"lotsize1000 = 0"),LinearRestriction({"bdrms":1},0,"bdrms = 0")))
    from core.joint_inference_utils import joint_f_test
    conventional_joint=joint_f_test(ols,(LinearRestriction({"lotsize1000":1},0,"lotsize1000=0"),LinearRestriction({"bdrms":1},0,"bdrms=0")))
    joint_cards=st.columns(2); joint_cards[0].metric("Geleneksel ortak F",f"{conventional_joint.f_statistic:.3f}",f"p={format_p_value(conventional_joint.p_value)}"); joint_cards[1].metric(f"{cov} Wald/F",f"{joint.f_statistic:.3f}",f"p={format_p_value(joint.p_value)}")
    st.write(f"Aynı H₀: β_lotsize1000=β_bdrms=0; q=2, payda sd={joint.df_denom}. Kovaryans değiştiği için karar değişebilir.")
    st.warning("Robust standart hata eksik değişken yanlılığını, yanlış fonksiyonel biçimi, küme bağımlılığını veya nedensel tasarımı düzeltmez.")
    st.subheader("10. Tekrarlı örnekleme benzetimi")
    reps=st.select_slider("Tekrar sayısı",options=(500,1000,2000,4000),value=1000,key="konu12_reps")
    sim=_simulation(60,reps)
    st.dataframe(pd.DataFrame([{"Eğim ortalaması":sim.slope_mean,"Ampirik SD":sim.empirical_slope_sd,"Geleneksel ort. SH":sim.mean_nonrobust_se,"HC1 ort. SH":sim.mean_hc1_se,"HC3 ort. SH":sim.mean_hc3_se,"Geleneksel kapsama":sim.nonrobust_coverage,"HC1 kapsama":sim.hc1_coverage,"HC3 kapsama":sim.hc3_coverage}]).round(3),hide_index=True,width="stretch")
    slope_samples=simulation_slope_samples(repetitions=reps).eğim.to_numpy(); counts,edges=np.histogram(slope_samples,bins=25); centers=(edges[:-1]+edges[1:])/2
    st.bar_chart(
        pd.DataFrame({"Eğim":centers,"Frekans":counts}).set_index("Eğim"),
        x_label="Eğim tahmini",
        y_label="Frekans",
    )
    st.bar_chart(
        pd.DataFrame({"Yöntem":["Geleneksel","HC1","HC3"],"Kapsama":[sim.nonrobust_coverage,sim.hc1_coverage,sim.hc3_coverage]}).set_index("Yöntem"),
        x_label="Standart hata yöntemi",
        y_label="Kapsama oranı",
    )
    st.caption("Nominal hedef .95'tir. Eğim doğru merkezin çevresinde kalırken çok küçük geleneksel SH kapsamanın düşmesine yol açabilir.")
    st.subheader("11. Robust standart hata neyi çözmez?")
    issue=st.selectbox("Sorun",("heteroskedastisite","eksik değişken","yanlış fonksiyonel biçim","içsellik","küme bağımlılığı","seri korelasyon","ölçüm hatası"),key="konu12_limit")
    if issue=="heteroskedastisite": st.success("Bağımsız gözlemler altında HC0–HC3 türü kovaryans çözümü uygun olabilir.")
    else: st.error("Farklı tanı veya yöntem gerekir; HC0–HC3 bu sorunu çözmez ve clustering/HAC yerine geçmez.")
    st.subheader("12. Makale tablosu ve raporlama")
    st.info(f"EKK lotsize1000 katsayısı {ols.coefficients['lotsize1000']:.4f}; {cov} SH {result.standard_errors['lotsize1000']:.4f}, p={format_p_value(float(result.p_values_two_sided_zero['lotsize1000']))}. BP p={format_p_value(bp.p_value)}, White p={format_p_value(white.p_value)}. Bu düzeltme eksik değişkeni veya yanlış model biçimini çözmez.")
    st.subheader("Kısa uygulama sorusu")
    context=Konu12QuestionContext(result,bp,sim); model_id=f"H12-D|{cov}"
    synchronize_question_state(st.session_state,model_id,"konu12"); index,_,answer=question_state_keys("konu12"); q=generate_konu12_question(context,model_id,int(st.session_state.get(index,0)))
    st.write(q.prompt); render_question_actions(topic_id="konu12")
    if st.session_state.get(answer,False): st.success(f"Çözüm: {q.answer}")
