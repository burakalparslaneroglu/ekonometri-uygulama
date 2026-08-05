"""Konu 02: Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.data_registry import DatasetMetadata, get_dataset_metadata, load_dataset, variable_metadata
from core.data_structure_utils import has_repeated_units, is_panel_structure, sort_time_series, summarize_panel_balance
from core.group_comparison_utils import GroupComparison, compare_two_groups
from core.scenario_registry import generate_konu02_question
from core.session_utils import next_question, question_state_keys, reveal_answer, synchronize_question_state


TOPIC_ID = "konu02"
LAB_DATASETS = ("wage1", "phillips", "cps78_85", "wagepan")


@st.cache_data(show_spinner=False)
def _load_dataset_cached(dataset_key: str) -> pd.DataFrame:
    """Wooldridge verisini Konu 02 için önbellekli biçimde yükler."""
    return load_dataset(dataset_key)


@st.cache_data(show_spinner=False)
def _jtrain_comparison() -> GroupComparison:
    """JTRAIN2 için iki grubun betimsel özetini üretir."""
    frame = load_dataset("jtrain2")
    return compare_two_groups(
        frame,
        "train",
        "re78",
        control_value=0,
        treatment_value=1,
        control_label="Kontrol grubu",
        treatment_label="Eğitim grubu",
        unit=variable_metadata("jtrain2", "re78").unit,
    )


def _metadata_rows(metadata: DatasetMetadata) -> pd.DataFrame:
    """Veri seti metadata'sını öğrenciye uygun tabloya dönüştürür."""
    rows = [
        ("Gözlem birimi", metadata.observation_unit),
        ("Veri yapısı", metadata.data_structure),
        ("Veri üretim biçimi", metadata.collection_method),
        ("Zaman boyutu", metadata.time_coverage or "Yok"),
        ("Veri sıklığı", metadata.frequency or "Uygulanmaz"),
        ("Kimlik alanı", metadata.identifier_variable or "Tanımlı değil"),
        ("Aynı birimler tekrar gözleniyor mu?", "Evet" if metadata.repeated_units else "Hayır"),
    ]
    return pd.DataFrame(rows, columns=["Başlık", "Bilgi"])


def _render_data_structure_intro() -> None:
    """Veri yapısını tanımak için temel karar kartlarını gösterir."""
    st.subheader("A. Veri yapısını tanıma")
    st.write("Bir dosyayı sınıflandırmadan önce satırların neyi temsil ettiğini ve zaman bilgisinin rolünü okuyun.")
    questions = (
        ("1. Bir satır neyi temsil ediyor?", "Çalışan, yıl, firma-yıl ya da başka bir temel birim olabilir."),
        ("2. Zaman boyutu var mı?", "Tek dönem mi, yoksa birden fazla dönem mi bulunduğunu belirleyin."),
        ("3. Zaman sırası bilgi taşıyor mu?", "Ardışık yılların sırası önemliyse zaman serisi düşünülür."),
        ("4. Aynı birimler tekrar gözleniyor mu?", "Evetse, birim ve dönem birlikte incelenmelidir."),
    )
    for column, (question, explanation) in zip(st.columns(4), questions, strict=True):
        column.markdown(f"**{question}**\n\n{explanation}")
    st.markdown(
        "**Kısa karar yolu:** Tek dönemde farklı birimler → yatay kesit verisi; ardışık dönemler → zaman serisi verisi; "
        "farklı dönemlerde farklı örneklemler → havuzlanmış yatay kesit verisi; aynı birimler farklı dönemlerde yeniden görünüyorsa → panel veri."
    )
    st.warning("`id` ve `year` adlı sütunların bulunması tek başına panel kanıtı değildir. Aynı kimliklerin farklı dönemlerde tekrarlandığını kontrol etmek gerekir.")


def _render_common_lab_content(metadata: DatasetMetadata, frame: pd.DataFrame) -> None:
    """Seçilen veri setinin ortak ön izlemesini ve metadata'sını gösterir."""
    st.markdown(f"**Kaynak:** {metadata.source}")
    st.write(metadata.description)
    metrics = st.columns(3)
    metrics[0].metric("Gözlem sayısı", f"{len(frame):,}")
    metrics[1].metric("Veri yapısı", metadata.data_structure)
    metrics[2].metric("Veri üretim biçimi", metadata.collection_method)
    st.dataframe(_metadata_rows(metadata), hide_index=True, width="stretch")
    st.dataframe(frame.head(8), hide_index=True, width="stretch")
    variable_table = pd.DataFrame(
        [
            {"Değişken": item.name, "Açıklama": item.description, "Ölçü birimi": item.unit}
            for item in metadata.variables.values()
        ]
    )
    st.dataframe(variable_table, hide_index=True, width="stretch")
    st.info(f"**Sınıflandırma gerekçesi:** {metadata.classification_reason}")


def _render_wage1_detail() -> None:
    """WAGE1'in yatay kesit özelliğini açıklar."""
    st.caption("WAGE1'de gözlem birimi çalışandır. Satırların dosyadaki sırası zaman sırası değildir; her satır farklı bir çalışanı temsil eder.")


def _render_phillips_detail(frame: pd.DataFrame) -> None:
    """PHILLIPS zaman serisinin sıralı görünümünü ve çizgilerini gösterir."""
    ordered = sort_time_series(frame, "year")
    st.caption("Yıl zaman indeksidir: veri yıl sırasına göre düzenlenmiştir. Bu grafikler birlikte hareketi gösterir; nedensel etki kanıtı değildir.")
    plot_frame = ordered.melt(id_vars="year", value_vars=["inf", "unem"], var_name="Değişken", value_name="Oran")
    plot_frame["Değişken"] = plot_frame["Değişken"].map({"inf": "Enflasyon oranı", "unem": "İşsizlik oranı"})
    figure = px.line(
        plot_frame,
        x="year",
        y="Oran",
        color="Değişken",
        labels={"year": "Yıl", "Oran": "Oran (yüzde)", "Değişken": "Gösterge"},
        color_discrete_sequence=["#107C89", "#B3392F"],
    )
    figure.update_layout(template="plotly_white", height=400, margin={"l": 10, "r": 10, "t": 25, "b": 10})
    st.plotly_chart(figure, width="stretch")


def _render_cps_detail(frame: pd.DataFrame) -> None:
    """CPS78_85'in iki kesit dönemini özetler."""
    summary = frame.groupby("year", sort=True)["educ"].agg(["count", "mean"]).reset_index()
    summary.columns = ["Örneklem yılı", "Gözlem sayısı", "Ortalama eğitim (yıl)"]
    st.dataframe(summary.round(3), hide_index=True, width="stretch")
    st.caption("1978 ve 1985 örneklemleri farklı çalışanlardan oluşabilir. Bu nedenle dönem ortalaması farkı tek başına bir politika etkisi olarak yorumlanmaz.")


def _render_wagepan_detail(frame: pd.DataFrame) -> None:
    """WAGEPAN'in tekrar eden kişi-yıl yapısını gösterir."""
    summary = summarize_panel_balance(frame, "nr", "year")
    repeated = has_repeated_units(frame, "nr")
    panel_structure = is_panel_structure(frame, "nr", "year")
    metrics = st.columns(4)
    metrics[0].metric("Kişi sayısı", summary.unit_count)
    metrics[1].metric("Dönem sayısı", summary.period_count)
    metrics[2].metric("Tekrar eden kişi kimlikleri", "Evet" if repeated else "Hayır")
    metrics[3].metric("Panel dengesi", "Dengeli" if panel_structure and summary.balanced else "Dengesiz")
    distribution = summary.periods_per_unit.value_counts().sort_index().rename_axis("Kişi başına dönem").reset_index(name="Kişi sayısı")
    st.dataframe(distribution, hide_index=True, width="stretch")
    sample_ids = frame["nr"].drop_duplicates().head(3)
    st.dataframe(frame.loc[frame["nr"].isin(sample_ids), ["nr", "year", "educ", "lwage"]].sort_values(["nr", "year"]), hide_index=True, width="stretch")
    st.caption("Her satır bir kişi-yıl birleşimidir. Denge kontrolü, her kişinin yalnızca dönem sayısını değil, gözlendiği dönem kümesini de inceler.")


def _render_dataset_lab() -> None:
    """Dört veri yapısını gerçek Wooldridge verileriyle inceletir."""
    st.subheader("B. Veri seti laboratuvarı")
    dataset_key = st.selectbox(
        "İncelenecek veri seti",
        options=LAB_DATASETS,
        format_func=lambda key: get_dataset_metadata(key).title,
        key="konu02_dataset",
    )
    metadata = get_dataset_metadata(dataset_key)
    try:
        frame = _load_dataset_cached(dataset_key)
        _render_common_lab_content(metadata, frame)
        if dataset_key == "wage1":
            _render_wage1_detail()
        elif dataset_key == "phillips":
            _render_phillips_detail(frame)
        elif dataset_key == "cps78_85":
            _render_cps_detail(frame)
        else:
            _render_wagepan_detail(frame)
    except (RuntimeError, ValueError) as error:
        st.error(f"Veri seti incelenemedi: {error}")


def _render_structure_collection_activity() -> None:
    """Veri düzeni ile veri üretim biçimi ayrımını görünür kılar."""
    st.subheader("C. Veri yapısı ile veri üretim biçimini ayırma")
    left, right = st.columns(2)
    left.markdown("**Veri yapısı**\n\nYatay kesit, zaman serisi, havuzlanmış yatay kesit veya panel veri; satırların ve zaman bilgisinin nasıl düzenlendiğini açıklar.")
    right.markdown("**Veri üretim biçimi**\n\nGözlemsel veya deneysel; açıklayıcı koşulun ya da uygulamanın nasıl oluştuğunu açıklar.")
    st.info("Panel veri deneysel veri demek değildir. Deneysel veri de panel olmak zorunda değildir: bunlar iki ayrı soruya verilen yanıtlardır.")


def _render_causality_content() -> None:
    """İlişki dili, ceteris paribus ve temel tehditleri tanıtır."""
    st.subheader("D. İlişki ve nedensellik")
    first, second = st.columns(2)
    first.markdown("**İlişki dili**\n\n“Örneklemde eğitim yılı ile saatlik ücret arasında pozitif bir ilişki gözlenmektedir.”")
    second.markdown("**Daha güçlü iddia**\n\n“Eğitim, ücreti artırır.” Bu ifade için uygun bir araştırma tasarımı gerekir.")
    st.warning("Korelasyon, görsel birliktelik veya grup farkı tek başına nedensel etki göstermez.")
    st.subheader("E. Ceteris paribus ve nedensellik tehditleri")
    cards = (
        ("Ceteris paribus", "Diğer ilgili koşullar aynıyken yapılan kavramsal karşılaştırmadır; gerçekte her şeyin sabit kaldığını söylemez."),
        ("Karıştırıcı faktör", "Hem temel açıklayıcı değişkenle hem sonuçla ilişkili olabilecek üçüncü bir unsurdur."),
        ("Ters nedensellik", "Araştırmacının öne sürdüğü yönün tersi ya da karşılıklı yön de işleyebilir."),
        ("Seçilim", "Grupların oluşumu veya uygulamaya katılım sonuçla ilişkili sistematik özelliklere bağlı olabilir."),
    )
    for column, (title, description) in zip(st.columns(4), cards, strict=True):
        column.markdown(f"**{title}**\n\n{description}")


def _render_jtrain_comparison(comparison: GroupComparison) -> None:
    """JTRAIN2 için izin verilen grup karşılaştırmasını gösterir."""
    st.subheader("F. JTRAIN2 grup karşılaştırması")
    metadata = get_dataset_metadata("jtrain2")
    st.markdown(f"**Kaynak:** {metadata.source}")
    st.write(metadata.description)
    cards = st.columns(5)
    cards[0].metric("Kontrol grubu gözlem sayısı", comparison.control_count)
    cards[1].metric("Eğitim grubu gözlem sayısı", comparison.treatment_count)
    cards[2].metric("Kontrol grubu ortalaması", f"{comparison.control_mean:.3f} {comparison.unit}")
    cards[3].metric("Eğitim grubu ortalaması", f"{comparison.treatment_mean:.3f} {comparison.unit}")
    cards[4].metric("Eğitim eksi kontrol", f"{comparison.difference:.3f} {comparison.unit}")
    st.caption("Gösterilen fark, eğitim grubu ortalamasından kontrol grubu ortalamasının çıkarılmasıyla elde edilir.")
    st.info("Rastgele atamanın gerçekten uygulandığı ve önemli uygulama sorunlarının bulunmadığı kabul edildiğinde, gruplar arasındaki ortalama fark nedensel etki bakımından gözlemsel bir karşılaştırmaya göre daha güçlü yorumlanabilir.")
    st.warning("Yine de rastgele atamanın uygulanması, başlangıç farkları, uygulamaya uyum, eksik sonuçlar ve deney örneklemi dışına genellenebilirlik ayrı ayrı değerlendirilmelidir.")


def _render_questions(comparison: GroupComparison) -> None:
    """Konu 02'nin ortak altyapılı soru panelini gösterir."""
    model_id = f"konu02:{st.session_state.get('konu02_dataset', LAB_DATASETS[0])}:veri_nedensellik"
    synchronize_question_state(st.session_state, model_id, topic_id=TOPIC_ID)
    index_key, _, answer_key = question_state_keys(TOPIC_ID)
    index = int(st.session_state.get(index_key, 0))
    question = generate_konu02_question(model_id, index, comparison)
    st.subheader("Kendini dene")
    st.markdown(f"**Soru {index + 1}:** {question.prompt}")
    st.text_area("Yanıtınız", key=f"konu02_student_response_{index}", help="Yanıtınızı yazdıktan sonra çözümle karşılaştırabilirsiniz.")
    first, second = st.columns(2)
    if first.button("Cevabı göster", type="primary", width="stretch", key="konu02_show_answer"):
        reveal_answer(st.session_state, topic_id=TOPIC_ID)
    if second.button("Yeni soru", width="stretch", key="konu02_next_question"):
        next_question(st.session_state, topic_id=TOPIC_ID)
        st.rerun()
    if st.session_state.get(answer_key, False):
        st.success(f"**Çözüm:** {question.answer}")
    else:
        st.caption("Çözümü görmek için “Cevabı göster” düğmesini kullanın.")


def render() -> None:
    """Konu 02'nin tam öğrenci arayüzünü oluşturur."""
    st.markdown("<span class='topic-badge'>KONU 02</span>", unsafe_allow_html=True)
    st.header("Konu 02 — Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus")
    st.markdown(
        "<div class='lesson-note'><strong>Amaç:</strong> Veri yapısını satır, zaman ve tekrar eden birimler üzerinden tanımak; "
        "veri üretim biçimini ayırmak; ilişki ile nedensel etki arasındaki farkı ve güvenli yorum dilini kavramak.</div>",
        unsafe_allow_html=True,
    )
    _render_data_structure_intro()
    _render_dataset_lab()
    _render_structure_collection_activity()
    _render_causality_content()
    try:
        comparison = _jtrain_comparison()
    except (RuntimeError, ValueError) as error:
        st.error(f"JTRAIN2 grup karşılaştırması oluşturulamadı: {error}")
        return
    _render_jtrain_comparison(comparison)
    st.divider()
    _render_questions(comparison)
