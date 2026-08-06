"""Konu 11: etkileşim terimleri ve grup farkları."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from core.data_registry import get_dataset_metadata,load_dataset
from core.interaction_utils import add_interaction_columns,center_interaction_model,classify_interaction_structure,conditional_difference_grid,conditional_group_difference,crossing_point,fit_interaction_model,group_lines,interaction_curve_data
from core.konu11_questions import Konu11QuestionContext,generate_konu11_question
from core.model_utils import format_numerical_difference
from core.regression_inference_utils import format_p_value
from core.session_utils import question_state_keys,synchronize_question_state
from core.ui_components import format_null_decision,render_question_actions

@st.cache_data(show_spinner=False)
def _results():
    """Ders notundaki WAGE1 ve HPRICE1 etkileşim modellerini önbelleğe alır."""
    w=add_interaction_columns(load_dataset("wage1"),"wage1"); h=add_interaction_columns(load_dataset("hprice1"),"hprice1")
    return {"w":fit_interaction_model(w,"lwage","educ12","female","female_educ12",("exper","tenure")),"h":fit_interaction_model(h,"price","lotsize10k","colonial","colonial_lotsize10k",("sqrft100","bdrms")),"wdata":w}

def render()->None:
    """Konu 11 ekranını gösterir; LPM kaynak kapsamı dışında bırakılır."""
    r=_results(); w=r["w"]; h=r["h"]; data=r["wdata"]; assert isinstance(data,pd.DataFrame)
    st.markdown("<span class='topic-badge'>KONU 11</span>",unsafe_allow_html=True)
    st.header("Etkileşim Terimleri ve Grup Farkları")
    st.info("Additif kukla modelinde doğrular paraleldir. D×X etkileşimi, grup farkının X düzeyine göre değişmesine izin verir. Bu kaynakta doğrusal olasılık modeli, logit veya probit yer almaz.")
    st.caption(f"Veri kaynakları: {get_dataset_metadata('wage1').source}; {get_dataset_metadata('hprice1').source}")
    st.markdown("**Bu sayfada ne yapacağız?**  Additif model → Etkileşim cebiri → Koşullu fark → Merkezleme → WAGE1/HPRICE1 → Test haritası")
    st.subheader("1. Additif model neden paralel doğrular üretir?")
    additive_beta0=st.slider("β₀",-5.0,10.0,2.0,.5,key="konu11_add_b0"); additive_beta1=st.slider("β₁",-2.0,4.0,1.0,.1,key="konu11_add_b1"); additive_gamma0=st.slider("γ₀",-5.0,5.0,2.0,.5,key="konu11_add_g0")
    additive=group_lines(additive_beta0,additive_beta1,additive_gamma0,0); additive_grid=interaction_curve_data(additive,np.linspace(0,10,101))
    st.latex(r"Y=\beta_0+\beta_1X+\gamma_0D+u")
    st.line_chart(additive_grid.set_index("x"),x_label="X",y_label="Tahmin edilen Y"); st.write(f"D=0: Ŷ={additive_beta0:.1f}+{additive_beta1:.1f}X; D=1: Ŷ={additive_beta0+additive_gamma0:.1f}+{additive_beta1:.1f}X. Dikey fark her X'te γ₀={additive_gamma0:.1f}.")

    st.subheader("2. Etkileşimli modelin cebiri")
    sliders=st.columns(4); b0=sliders[0].number_input("β₀",value=10.0,key="konu11_b0"); b1=sliders[1].number_input("β₁",value=3.0,key="konu11_b1"); g0=sliders[2].number_input("γ₀",value=5.0,key="konu11_g0"); g1=sliders[3].number_input("γ₁",value=-1.5,key="konu11_g1")
    x_lab=st.slider("Seçili X",0.0,10.0,4.0,.5,key="konu11_x_lab"); lab_lines=group_lines(b0,b1,g0,g1); lab_grid=interaction_curve_data(lab_lines,np.linspace(0,10,101)); st.line_chart(lab_grid.set_index("x"),x_label="X",y_label="Tahmin edilen Y")
    cards=st.columns(4); cards[0].metric("D=0 sabiti",f"{lab_lines.intercept_zero:.2f}"); cards[1].metric("D=0 eğimi",f"{lab_lines.slope_zero:.2f}"); cards[2].metric("D=1 sabiti",f"{lab_lines.intercept_one:.2f}"); cards[3].metric("D=1 eğimi",f"{lab_lines.slope_one:.2f}")
    st.write(f"Seçili X'te grup farkı Δ({x_lab:.1f})=γ₀+γ₁X={g0+g1*x_lab:.3f}. Etkileşim katsayısı γ₁ eğim farkıdır; D=1 toplam eğimi değildir.")

    st.subheader("3. Dört temel grup regresyon yapısı")
    structure=st.selectbox("Yapı",("Aynı sabit – aynı eğim","Farklı sabit – aynı eğim","Aynı sabit – farklı eğim","Farklı sabit – farklı eğim"),key="konu11_structure")
    params={"Aynı sabit – aynı eğim":(0,0),"Farklı sabit – aynı eğim":(2,0),"Aynı sabit – farklı eğim":(0,1),"Farklı sabit – farklı eğim":(2,1)}[structure]
    structure_lines=group_lines(1,1,*params); st.line_chart(interaction_curve_data(structure_lines,np.linspace(0,5,51)).set_index("x"),x_label="X",y_label="Tahmin edilen Y"); st.write(f"Sınıflandırma: **{classify_interaction_structure(*params)}**. γ₀={'0' if params[0]==0 else '≠0'}, γ₁={'0' if params[1]==0 else '≠0'}.")
    st.dataframe(pd.DataFrame([{"Yapı":"Aynı sabit – aynı eğim","D":"—","X":"✓","D×X":"—","Yapısal kısıt":"γ₀=0, γ₁=0"},{"Yapı":"Farklı sabit – aynı eğim","D":"✓","X":"✓","D×X":"—","Yapısal kısıt":"γ₀≠0, γ₁=0"},{"Yapı":"Aynı sabit – farklı eğim","D":"ana etki hiyerarşi gereği modelde tutulur","X":"✓","D×X":"✓","Yapısal kısıt":"γ₀=0, γ₁≠0"},{"Yapı":"Farklı sabit – farklı eğim","D":"✓","X":"✓","D×X":"✓","Yapısal kısıt":"γ₀≠0, γ₁≠0"}]),hide_index=True,width="stretch")

    st.subheader("4. Koşullu grup farkı ve güven bandı")
    st.latex(r"Y=\beta_0+\beta_1X+\gamma_0D+\gamma_1(D\times X)+u")
    st.write(f"D=0: sabit={w.lines.intercept_zero:.4f}, eğim={w.lines.slope_zero:.4f}; D=1: sabit={w.lines.intercept_one:.4f}, eğim={w.lines.slope_one:.4f}.")
    selected=st.slider("Eğitim (yıl)",8.0,20.0,12.0,.5,key="konu11_educ")
    gap=conditional_group_difference(w.result,"female","female_educ12",selected-12)
    st.dataframe(pd.DataFrame([{"Eğitim":selected,"Kadın − erkek koşullu fark":gap.estimate,"SH":gap.standard_error,"p-değeri":format_p_value(gap.p_value),"%95 GA":f"[{gap.confidence_interval_95[0]:.3f}, {gap.confidence_interval_95[1]:.3f}]"}]),hide_index=True,width="stretch")
    gap_grid=conditional_difference_grid(w.result,"female","female_educ12",np.linspace(float(data.educ12.min()),float(data.educ12.max()),81),data_min=float(data.educ12.min()),data_max=float(data.educ12.max()))
    gap_plot=gap_grid.assign(**{"Eğitim yılı":gap_grid["x"]+12.0,"Kadın − erkek farkı":gap_grid["estimate"],"%95 GA alt":gap_grid["lower"],"%95 GA üst":gap_grid["upper"]})
    st.line_chart(
        gap_plot,
        x="Eğitim yılı",
        y=["Kadın − erkek farkı","%95 GA alt","%95 GA üst"],
        x_label="Eğitim yılı",
        y_label="Koşullu log ücret farkı",
    )
    st.latex(r"Var(\widehat\gamma_0+X\widehat\gamma_1)=Var(\widehat\gamma_0)+X^2Var(\widehat\gamma_1)+2XCov(\widehat\gamma_0,\widehat\gamma_1)")
    cross,inside=crossing_point(w.lines,data_min=float(data.educ12.min()),data_max=float(data.educ12.max()))
    st.caption("Kesişim: " + ("paralel doğrular nedeniyle sonlu kesişim yok." if cross is None else f"merkezlenmiş değer educ12={cross:.2f}; gerçek eğitim yılı={cross+12.0:.2f}; veri aralığında: {'evet' if inside else 'hayır'}"))
    if cross is not None and not inside: st.warning("Cebirsel kesişim gözlenen eğitim aralığının dışındadır; güçlü yorum ekstrapolasyon olur.")

    st.subheader("5. Merkezleme laboratuvarı")
    center=st.slider("Eğitim merkezi c",8.0,18.0,12.0,1.0,key="konu11_center")
    centered=center_interaction_model(data,"lwage","educ","female",("exper","tenure"),center=center)
    coefficient_cards=st.columns(2); coefficient_cards[0].metric("Ham female katsayısı",f"{centered.raw_result.coefficients['female']:.4f}"); coefficient_cards[1].metric(f"female katsayısı (educ={center:.0f})",f"{centered.centered_result.coefficients['female']:.4f}")
    difference_cards=st.columns(4); difference_cards[0].metric("Maks. fitted farkı",format_numerical_difference(centered.fitted_max_difference)); difference_cards[1].metric("Maks. artık farkı",format_numerical_difference(centered.residual_max_difference)); difference_cards[2].metric("R² farkı",format_numerical_difference(centered.r_squared_difference)); difference_cards[3].metric("SSR farkı",format_numerical_difference(centered.ssr_difference))
    st.caption("Merkezleme ekonomik ilişkiyi değiştirmez; ana kukla katsayısının yorumlandığı eğitim yılını değiştirir. Fitted değerler, artıklar, R² ve SSR sayısal tolerans içinde aynıdır.")

    st.subheader("6. WAGE1 kadın–eğitim etkileşimi")
    wage_table=pd.DataFrame({"Katsayı":w.result.coefficients.index,"Tahmin":w.result.coefficients.values,"SH":w.result.standard_errors.values,"t":w.result.t_values_zero.values,"p":[format_p_value(float(v)) for v in w.result.p_values_two_sided_zero],"%95 GA":[f"[{row.lower:.4f}, {row.upper:.4f}]" for row in w.result.confidence_intervals_95.itertuples()]})
    st.dataframe(wage_table,hide_index=True,width="stretch")
    st.caption("Model: lwage ~ female × educ12 + exper + tenure. Sonuç: log saatlik ücret; referans grup: erkek (female=0); kovaryans türü: geleneksel (nonrobust).")
    slopes=st.columns(2); slopes[0].metric("Erkek eğitim eğimi",f"{w.lines.slope_zero:.4f} (tam %{100*np.expm1(w.lines.slope_zero):.2f})"); slopes[1].metric("Kadın eğitim eğimi",f"{w.lines.slope_one:.4f} (tam %{100*np.expm1(w.lines.slope_one):.2f})")
    st.write(f"İki grup regresyon ilişkisinin bütünüyle eşitliği için ortak F testi: F(2,{w.result.df_resid})={w.joint_f:.3f}, p={format_p_value(w.joint_p_value)}.")
    st.warning("Etkileşim katsayısının anlamsız olması, ücret düzeylerinin aynı olduğu anlamına gelmez. Eğim, merkezde fark ve iki ilişkinin bütünüyle eşitliği farklı null hipotezleridir.")

    st.subheader("7. HPRICE1 colonial–arsa etkileşimi")
    lot=st.select_slider("Arsa alanı (kare fit)",options=(10000,15000,25000,50000),value=10000,key="konu11_lot")
    scaled=(lot-10000)/1000; hgap=conditional_group_difference(h.result,"colonial","colonial_lotsize10k",scaled)
    st.dataframe(pd.DataFrame({"Katsayı":h.result.coefficients.index,"Tahmin (bin ABD doları)":h.result.coefficients.values,"SH":h.result.standard_errors.values,"p":[format_p_value(float(v)) for v in h.result.p_values_two_sided_zero]}),hide_index=True,width="stretch")
    st.caption("Model: price ~ colonial × lotsize10k + sqrft100 + bdrms. Sonuç birimi: bin ABD doları; referans grup: colonial=0 (diğer konutlar); kovaryans türü: geleneksel (nonrobust).")
    st.write(f"{lot:,} kare fitte colonial − diğer farkı: {hgap.estimate:.4f} bin dolar; SH={hgap.standard_error:.4f}; p={format_p_value(hgap.p_value)}; %95 GA=[{hgap.confidence_interval_95[0]:.3f}, {hgap.confidence_interval_95[1]:.3f}].")
    decision_5=format_null_decision(h.joint_p_value,.05); decision_10=format_null_decision(h.joint_p_value,.10)
    st.write(f"Etkileşim p={format_p_value(float(h.result.p_values_two_sided_zero['colonial_lotsize10k']))}; ortak F(2,{h.result.df_resid})={h.joint_f:.3f}, p={format_p_value(h.joint_p_value)}. α=0.05 (%5): {decision_5}; α=0.10 (%10): {decision_10}.")

    st.subheader("8. Tekli ve ortak test haritası")
    st.dataframe(pd.DataFrame([{"Araştırma sorusu":"Eğimler eşit mi?","Null":"γ₁=0","Test":"t veya F(1,df)"},{"Araştırma sorusu":"Merkezde fark var mı?","Null":"γ₀=0","Test":"t"},{"Araştırma sorusu":"Seçili X'te fark var mı?","Null":"γ₀+Xγ₁=0","Test":"Doğrusal birleşim"},{"Araştırma sorusu":"İki ilişki bütünüyle aynı mı?","Null":"γ₀=γ₁=0","Test":"Ortak F"}]),hide_index=True,width="stretch")
    st.subheader("9. Python formül sözdizimi ve raporlama")
    st.code("lwage ~ female * educ12 + exper + tenure\nprice ~ colonial * lotsize10k + sqrft100 + bdrms",language="python")
    st.caption("`D * X`, D + X + D:X terimlerini açar. `D : X` yalnız etkileşimi ekler; ana etkileri ayrıca yazmamak hiyerarşi ve yorum sorununa yol açar.")
    st.subheader("Kısa uygulama sorusu")
    center_gap=conditional_group_difference(w.result,"female","female_educ12",0)
    context=Konu11QuestionContext(w,center_gap); model_id="W11-I|educ12"
    synchronize_question_state(st.session_state,model_id,"konu11"); index,_,answer=question_state_keys("konu11"); question=generate_konu11_question(context,model_id,int(st.session_state.get(index,0)))
    st.write(question.prompt); render_question_actions(topic_id="konu11")
    if st.session_state.get(answer,False): st.success(f"Çözüm: {question.answer}")
