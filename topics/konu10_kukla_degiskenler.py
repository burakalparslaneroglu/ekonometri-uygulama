"""Konu 10: kukla değişkenler ve kategorik açıklayıcı değişkenler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.categorical_regression_utils import (add_categorical_columns, binary_group_summary, build_reference_dummies, category_contrast_inference, category_joint_test, compare_raw_and_controlled_dummy, dummy_design_preview, dummy_log_exact_percent, fit_binary_dummy_model, numeric_vs_dummy_coding_comparison)
from core.data_registry import get_dataset_metadata, load_dataset
from core.konu10_questions import Konu10QuestionContext, generate_konu10_question
from core.regression_inference_utils import fit_ols_inference, format_p_value
from core.session_utils import question_state_keys, synchronize_question_state
from core.ui_components import render_question_actions


@st.cache_data(show_spinner=False)
def _results() -> dict[str, object]:
    """WAGE1 tabanlı sabit Konu 10 hesaplarını önbelleğe alır."""
    data = add_categorical_columns(load_dataset("wage1"), "wage1")
    comparison = compare_raw_and_controlled_dummy(data)
    region = fit_ols_inference(data, "lwage", ("educ", "exper", "expersq", "tenure", "tenursq", "northcen", "south", "west"))
    industry = fit_ols_inference(data, "lwage", ("educ", "exper", "expersq", "tenure", "tenursq", "construc", "ndurman", "trcommpu", "trade", "services", "profserv"))
    return {"data": data, "comparison": comparison, "region": region, "industry": industry}


def render() -> None:
    """Konu 10 öğretim ekranını gösterir."""
    values = _results(); data = values["data"]; comparison = values["comparison"]; region = values["region"]; industry = values["industry"]
    assert isinstance(data, pd.DataFrame)
    st.header("KONU 10")
    st.subheader("Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler")
    st.info("Şimdiye kadar nicel değişkenlerle çalıştık. Bu bölümde cinsiyet, bölge ve sektör gibi kategorik bilgileri regresyona aktaracağız. Eğim farkları Konu 11'de ele alınır.")
    metadata = get_dataset_metadata("wage1")
    st.caption(f"Veri kaynağı: {metadata.source} | Gözlem birimi: {metadata.observation_unit}")
    st.markdown("**Bu sayfada ne yapacağız?**  Kavram → Kodlama → Grup farkı → Referans → Rank → Ortak test → Raporlama")

    st.subheader("1. Nicel mi, kategorik mi?")
    examples={"Eğitim yılı":("Nicel","Yıl","Evet","Doğrusal eğim veya gerekçeli dönüşüm"),"Saatlik ücret":("Nicel","Dolar/saat","Evet","Sonuç veya nicel açıklayıcı"),"Bölge":("Kategorik","Etiket","Hayır","Referans + m−1 kukla"),"Sektör":("Kategorik","Etiket","Hayır","Referans + m−1 kukla"),"Kadın göstergesi":("Kategorik","0/1 etiketi","Hayır","Tek kukla"),"Medeni durum":("Kategorik","Etiket","Hayır","Kukla kodlama"),"Konut türü":("Kategorik","Etiket","Hayır","Kukla kodlama"),"Yatak odası sayısı":("Nicel (ayrık)","Adet","Evet","Nicel değişken; biçim ayrıca değerlendirilebilir")}
    selected_example=st.selectbox("Bir değişken seçin",tuple(examples),key="konu10_variable_type")
    variable_type,unit,meaning,approach=examples[selected_example]
    st.dataframe(pd.DataFrame([{"Tür":variable_type,"Sayı ölçü birimi taşır mı?":unit,"Bir birim artış anlamlı mı?":meaning,"Uygun kodlama":approach}]),hide_index=True,width="stretch")
    st.dataframe(pd.DataFrame({"Değişken":list(examples),"Tür":[value[0] for value in examples.values()]}),hide_index=True,width="stretch")
    st.warning("Kuzeydoğu=1, Güney=2, Batı=3 gibi tek sayısal kod, yapay sıra, eşit mesafe ve doğrusal fark varsayar. Kategori etiketleri için bu genellikle anlamlı değildir.")

    st.subheader("2. Neden A=1, B=2, C=3 tek eğimi risklidir?")
    synthetic=pd.DataFrame({"kategori":np.repeat(["A","B","C"],8),"sonuç":np.repeat([10.0,13.0,21.0],8)+np.tile(np.linspace(-.5,.5,8),3)})
    coding=numeric_vs_dummy_coding_comparison(synthetic["kategori"],synthetic["sonuç"])
    coding_table=coding.group_means.rename(columns={"category":"Kategori","size":"n","mean":"Grup ortalaması"})
    st.dataframe(coding_table.round(3),hide_index=True,width="stretch")
    plot=coding_table.set_index("Kategori")[["Grup ortalaması"]].copy(); plot["1–2–3 doğrusal tahmin"]=[float(coding.numeric_fitted[synthetic.kategori==cat].mean()) for cat in ["A","B","C"]]
    st.bar_chart(plot)
    st.write(f"Sayısal kod SSR={coding.numeric_ssr:.3f}; kukla kodlama SSR={coding.dummy_ssr:.3f}. Tek eğim A→B ile B→C farkını eşit olmaya zorlar.")

    st.subheader("3. 0–1 kukla kodlama laboratuvarı")
    direction = st.radio("Kodlama yönü", ("D=1 kadın, D=0 erkek", "D=1 erkek, D=0 kadın"), horizontal=True, key="konu10_direction")
    if direction.startswith("D=1 kadın"):
        summary = binary_group_summary(data, "wage", "female", reference_label="Erkek", comparison_label="Kadın")
    else:
        reversed_data = data.copy(); reversed_data["male"] = 1 - reversed_data["female"]
        summary = binary_group_summary(reversed_data, "wage", "male", reference_label="Kadın", comparison_label="Erkek")
    active=data if direction.startswith("D=1 kadın") else reversed_data; group="female" if direction.startswith("D=1 kadın") else "male"; binary=fit_binary_dummy_model(active,"wage",group)
    cards=st.columns(4); cards[0].metric("Referans",summary.reference_label); cards[1].metric("β̂₀",f"{binary.coefficients['const']:.4f}"); cards[2].metric("δ̂",f"{binary.coefficients[group]:.4f}"); cards[3].metric("R²",f"{binary.r_squared:.4f}")
    st.dataframe(pd.DataFrame([{"Grup":summary.reference_label,"n":summary.reference_n,"Ortalama ücret":summary.reference_mean},{"Grup":summary.comparison_label,"n":summary.comparison_n,"Ortalama ücret":summary.comparison_mean}]).round(4),hide_index=True,width="stretch")
    st.bar_chart(pd.DataFrame({"Grup":[summary.reference_label,summary.comparison_label],"Ortalama ücret":[summary.reference_mean,summary.comparison_mean]}).set_index("Grup"))
    st.latex(r"\widehat\beta_0=\bar Y_{D=0},\qquad \widehat\delta=\bar Y_{D=1}-\bar Y_{D=0}")
    st.caption("Sabit referans grubun ortalamasını; kukla katsayısı karşılaştırma eksi referans ortalamasını verir. Kodlama değişince işaret değişir, fitted değerler değişmez.")

    st.subheader("4. Ham fark ve kontrollü fark")
    assert hasattr(comparison, "raw_model")
    models=[comparison.raw_model,comparison.controlled_level_model,comparison.controlled_log_model]
    model_names=["M1 Ham düzey","M2 Kontrollü düzey","M3 Kontrollü log"]
    st.dataframe(pd.DataFrame([{"Model":name,"female":m.coefficients['female'],"SH":m.standard_errors['female'],"t":m.t_values_zero['female'],"p":format_p_value(float(m.p_values_two_sided_zero['female'])),"%95 GA":f"[{m.confidence_intervals_95.loc['female','lower']:.3f}, {m.confidence_intervals_95.loc['female','upper']:.3f}]","n":m.nobs,"R²":m.r_squared} for name,m in zip(model_names,models,strict=True)]).round(4),hide_index=True,width="stretch")
    st.line_chart(pd.DataFrame({"Model":model_names,"Tahmin":[m.coefficients['female'] for m in models]}).set_index("Model"))
    st.success(f"Log modelinde tam yüzde dönüşüm: 100(exp(δ)−1) = %{comparison.exact_log_percent:.2f}; %95 aralık: [%{comparison.exact_log_ci_percent[0]:.2f}, %{comparison.exact_log_ci_percent[1]:.2f}].")
    st.caption("Kontrollü katsayı, seçili kontroller sabitken tahmin edilen farktır; nedensel ayrımcılık tahmini değildir.")

    st.subheader("5. Log kukla: yaklaşık ve tam yüzde")
    delta=st.slider("δ",-.60,.60,float(round(comparison.controlled_log_difference,4)),.01,key="konu10_delta")
    exact=dummy_log_exact_percent(delta); approximate=100*delta
    metric=st.columns(3); metric[0].metric("Yaklaşık yüzde",f"%{approximate:.2f}"); metric[1].metric("Tam yüzde",f"%{exact:.2f}"); metric[2].metric("Fark",f"{exact-approximate:.2f} puan")
    curve_x=np.linspace(-.6,.6,121); st.line_chart(pd.DataFrame({"δ":curve_x,"100δ":100*curve_x,"100(exp(δ)−1)":100*np.expm1(curve_x)}).set_index("δ"))
    st.caption("Bunlar yüzde değişimidir; yüzde puan değişimi değildir. Exact dönüşüm negatif ve pozitif tarafta simetrik değildir.")

    st.subheader("6. Referans kategori ve kategori karşıtlığı")
    regions=np.select([data.northcen.eq(1),data.south.eq(1),data.west.eq(1)],["Kuzey Merkez","Güney","Batı"],default="Kuzeydoğu")
    reference=st.selectbox("Referans bölge",("Kuzeydoğu","Kuzey Merkez","Güney","Batı"),key="konu10_region_reference")
    coding=build_reference_dummies(pd.Series(regions,index=data.index),reference_category=reference,prefix="region",category_order=("Kuzeydoğu","Kuzey Merkez","Güney","Batı"))
    region_data=data.join(coding.dummies); controls=("educ","exper","expersq","tenure","tenursq"); selected_region_model=fit_ols_inference(region_data,"lwage",(*controls,*coding.dummies.columns))
    st.dataframe(coding.coding_table,hide_index=True,width="stretch")
    first_category=st.selectbox("Kategori A",coding.categories,index=3,key="konu10_contrast_a"); second_category=st.selectbox("Kategori B",coding.categories,index=2,key="konu10_contrast_b")
    if first_category!=second_category:
        coefficient_map={cat:f"region_{cat}" for cat in coding.categories if cat!=reference}; contrast=category_contrast_inference(selected_region_model,first_category=first_category,second_category=second_category,reference_category=reference,coefficient_map=coefficient_map)
        st.write(f"{first_category} − {second_category}: {contrast.estimate:.4f}, SH={contrast.standard_error:.4f}, t={contrast.t_statistic:.3f}, p={format_p_value(contrast.p_value)}, %95 GA=[{contrast.confidence_interval_95[0]:.4f}, {contrast.confidence_interval_95[1]:.4f}]")
    st.caption("Referans değişince katsayı adları değişir; aynı kategori karşıtlığı, fitted değerler, artıklar, R² ve SSR değişmez.")

    st.subheader("7. Kukla değişken tuzağı ve rank laboratuvarı")
    scenario=st.radio("Tasarım",("Sabit + bütün kuklalar","Sabit + m−1 kukla","Sabitsiz + bütün kuklalar"),horizontal=True,key="konu10_rank_scenario")
    includes=scenario!="Sabitsiz + bütün kuklalar"; all_dummies=scenario!="Sabit + m−1 kukla"
    preview,trap=dummy_design_preview(pd.Series(["A","B","C","A","B","C"]),includes_intercept=includes,include_all_dummies=all_dummies)
    st.dataframe(preview,hide_index=True,width="stretch"); rank_cards=st.columns(4); rank_cards[0].metric("Sütun",trap.column_count); rank_cards[1].metric("Rank",trap.matrix_rank); rank_cards[2].metric("Tam rank", "Evet" if trap.full_rank else "Hayır"); rank_cards[3].metric("Tahmin", "Benzersiz" if trap.full_rank else "Sunulmaz")
    st.latex(r"1=D_1+D_2+\cdots+D_m"); st.caption(trap.explanation)

    st.subheader("8. Bölge ve endüstri ortak F testleri")
    region_test = category_joint_test(region, ("northcen", "south", "west"), category_name="Bölge", reference_category="Kuzeydoğu")
    industry_test = category_joint_test(industry, ("construc", "ndurman", "trcommpu", "trade", "services", "profserv"), category_name="Endüstri", reference_category="Diğer")
    region_tab,industry_tab=st.tabs(("Bölge kuklaları","Endüstri kuklaları"))
    for target,test,model,coefs,null in ((region_tab,region_test,region,("northcen","south","west"),"H₀: northcen=south=west=0"),(industry_tab,industry_test,industry,("construc","ndurman","trcommpu","trade","services","profserv"),"H₀: bütün endüstri katsayıları=0")):
        with target:
            st.markdown(f"**{null}**"); st.dataframe(pd.DataFrame({"Katsayı":coefs,"Tahmin":[model.coefficients[x] for x in coefs],"Tekli p":[format_p_value(float(model.p_values_two_sided_zero[x])) for x in coefs]}),hide_index=True,width="stretch")
            st.write(f"q={test.q}; F({test.q},{model.df_resid})={test.f_statistic:.3f}; p={format_p_value(test.p_value)}; karar: {'H₀ reddedilir' if test.reject_null else 'H₀ reddedilemez'}.")
            st.caption("Ortak reddetme, bütün katsayıların tek tek anlamlı olmasını gerektirmez.")
    st.subheader("9. Python, makale tablosu ve raporlama")
    st.code("lwage ~ educ + exper + expersq + tenure + tenursq + C(region)", language="python")
    st.info(f"Rapor: Erkek referans grubuna göre kadınların kontrollü log ücret farkı {comparison.controlled_log_difference:.4f}; geleneksel SH {comparison.controlled_log_model.standard_errors['female']:.4f}, p={format_p_value(float(comparison.controlled_log_model.p_values_two_sided_zero['female']))}. Exact fark %{comparison.exact_log_percent:.2f}. Bulgular gözlemsel ve koşulludur; tek başına nedensel etki değildir.")

    st.subheader("Kısa uygulama sorusu")
    context = Konu10QuestionContext(comparison, region_test.p_value)
    model_id = "W10-L|NE|nonrobust"
    synchronize_question_state(st.session_state, model_id, "konu10")
    index_key, _, answer_key = question_state_keys("konu10")
    question = generate_konu10_question(context, model_id, int(st.session_state.get(index_key, 0)))
    st.write(question.prompt)
    render_question_actions(topic_id="konu10")
    if st.session_state.get(answer_key, False):
        st.success(f"Çözüm: {question.answer}")
