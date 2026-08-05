"""Konu 06 için bağlama duyarlı, deterministik sorular."""

from __future__ import annotations

from dataclasses import dataclass

from core.assumption_diagnostics_utils import (
    CollinearityVariabilityResult,
    OVBResult,
    RepeatedSamplingResult,
    Wage1OVBDecomposition,
)
from core.model_utils import format_number
from core.question_engine import GeneratedQuestion, cycle_question_type


KONU06_QUESTION_TYPES = (
    "hesaplama_ve_guvenilirlik", "yorum_duzeyi", "varsayim_eslestirme",
    "parametrelerde_dogrusal", "rassal_ornekleme", "tam_baglanti",
    "sifir_kosullu_ortalama", "hata_sifir_degil", "artik_ortogonalligi",
    "tahmin_tahmin_edici", "bes_tahmin", "tek_tahmin", "merkez_ve_degiskinlik",
    "gamma_merkez", "tekrar_yanlilik", "ovb_iki_kosul", "ovb_yon",
    "ovb_sayisal", "sentetik_ayristirma", "wage_isaret", "wage_ayristirma",
    "katsayi_degisim_siniri", "tam_ve_yuksek", "rank", "vif_hesabi",
    "vif_siniri", "yuksek_baglanti_yanlilik", "toplam_kararliligi", "duyarlilik",
    "yuksek_r2", "arastirma_sorusu", "ovb_ve_baglanti", "nedensel_dil", "ileri_kapsam",
)


@dataclass(frozen=True)
class Konu06QuestionContext:
    """Soru üretiminde kullanılan görünür sayısal bağlam."""

    context_id: str
    repeated: RepeatedSamplingResult
    ovb: OVBResult
    wage: Wage1OVBDecomposition
    vif: float
    exact_collinearity: bool
    variability: CollinearityVariabilityResult


def generate_konu06_question(context: Konu06QuestionContext, question_index: int) -> GeneratedQuestion:
    """Aynı bağlam ve indeks için aynı Konu 06 soru/çözümünü üretir."""
    kind = cycle_question_type(context.context_id, question_index, KONU06_QUESTION_TYPES, namespace="konu06")
    repeated, wage, variability = context.repeated, context.wage, context.variability
    if kind == "hesaplama_ve_guvenilirlik":
        prompt, answer = "Yazılımın EKK katsayısı üretmesi modelin güvenilir olduğunu kanıtlar mı?", "Hayır. Katsayı hesaplamak ile anakütle yorumunun güvenilir olması farklıdır; varsayımlar ayrıca değerlendirilir."
    elif kind == "yorum_duzeyi":
        prompt, answer = "“Katsayı 0,42 hesaplandı” hangi yorum düzeyidir?", "Bu hesaplama düzeyindedir. Anakütle veya nedensel yorum ek varsayım ve araştırma tasarımı gerektirir."
    elif kind == "varsayim_eslestirme":
        prompt, answer = "A1–A4 varsayımlarını rollerine göre eşleştirin.", "A1 parametrelerde doğrusallık, A2 rassal örnekleme, A3 tam çoklu doğrusal bağlantı yokluğu, A4 sıfır koşullu ortalamadır."
    elif kind == "parametrelerde_dogrusal":
        prompt, answer = "Y=β₀+β₁X+β₂X²+u modeli parametrelerde doğrusal mıdır?", "Evet. X² gözlenen yeni bir açıklayıcı değişkendir; bilinmeyen parametreler birinci kuvvette yer alır."
    elif kind == "rassal_ornekleme":
        prompt, answer = "Yalnız gönüllülerin yanıtladığı anket ilk olarak hangi varsayımı tehdit eder?", "A2 rassal örneklemeyi tehdit eder; seçilen gözlemler hedef anakütleyi sistematik biçimde temsil etmeyebilir."
    elif kind == "tam_baglanti":
        prompt, answer = "Aylık gelir ve tam olarak 12 katı yıllık gelir birlikteyse ne olur?", "Tam çoklu doğrusal bağlantı vardır; veri ayrı katsayıları benzersiz biçimde belirleyemez."
    elif kind == "sifir_kosullu_ortalama":
        prompt, answer = "E(u|X)=0 ekonomik olarak ne söyler?", "X verildiğinde hata teriminde kalan faktörlerin ortalaması sistematik biçimde pozitif ya da negatif değildir."
    elif kind == "hata_sifir_degil":
        prompt, answer = "E(u|X)=0 her gözlemde u=0 demek midir?", "Hayır. Tek tek hata terimleri pozitif veya negatif olabilir; koşullu ortalama sıfırdır."
    elif kind == "artik_ortogonalligi":
        prompt, answer = "EKK artıklarının X ile örneklem ilişkisi zayıfsa A4 kanıtlanır mı?", "Hayır. Bu, EKK normal denklemlerinin örneklem özelliğidir; A4 gözlenmeyen anakütle hata terimi hakkındadır."
    elif kind == "tahmin_tahmin_edici":
        prompt, answer = "Tahmin ile tahmin ediciyi ayırın.", "Belirli örneklemdeki β̂ bir tahmindir. Tahmin edici ise farklı örneklemlerde farklı tahminler üreten kuraldır; yansızlık onun özelliğidir."
    elif kind == "bes_tahmin":
        values = (3.5, 4.4, 3.8, 4.1, 4.2)
        prompt, answer = "3,5; 4,4; 3,8; 4,1; 4,2 tahminlerinin ortalaması nedir?", f"Ortalama = {format_number(sum(values) / len(values))}. Tek bir küçük tahmin dizisi yansızlığı kesin kanıtlamaz ya da çürütmez."
    elif kind == "tek_tahmin":
        prompt, answer = "Tek örneklemde β̂=β bulunması yansızlığı kanıtlar mı?", "Hayır. Yansızlık, tekrarlı örneklerde tahminlerin beklenen merkezine ilişkindir."
    elif kind == "merkez_ve_degiskinlik":
        prompt, answer = "Benzetimde merkez ile tahminlerin standart sapması neyi ayırır?", f"Merkez {format_number(repeated.mean_estimate)} civarındadır; tahminlerin tekrarlı örneklerde standart sapması {format_number(repeated.estimate_std)} değişkenliği gösterir. Bu tek örneklem standart hatası değildir."
    elif kind == "gamma_merkez":
        prompt, answer = "γ=0,8 iken β₁=1,5 için beklenen merkez nedir?", "Merkez 1,5+0,8=2,3 olur; sıfır koşullu ortalama bozulduğu için hedef parametreden kayar."
    elif kind == "tekrar_yanlilik":
        prompt, answer = "Tekrar sayısını artırmak yanlı merkezi neden 1,5'e döndürmez?", "Daha çok tekrar, tahminlerin yanlış merkez çevresinde toplandığını daha açık gösterir; DGP'deki sistematik kaymayı düzeltmez."
    elif kind == "ovb_iki_kosul":
        prompt, answer = "Eksik değişken yanlılığı için iki koşul nedir?", "Dışlanan Z hem Y ile ilişkili olmalı hem de modelde tutulan X ile ilişkili olmalıdır."
    elif kind == "ovb_yon":
        prompt, answer = "β₂ pozitif ve δ₁ negatifse yanlılığın yönü nedir?", "β₂δ₁ negatiftir; yanlılık aşağı yönlüdür."
    elif kind == "ovb_sayisal":
        prompt, answer = "β₁=1,2, β₂=−2 ve δ₁=0,4 iken yanlılık ve kısa katsayı merkezi nedir?", "Yanlılık −2×0,4=−0,8; kısa katsayı merkezi 1,2−0,8=0,4 olur."
    elif kind == "sentetik_ayristirma":
        prompt, answer = "Sentetik uygulamada kısa model eğimi neden yaklaşık 4,1'dir?", "X'in gerçek katsayısı 2'dir; Z dışarıda kaldığında yaklaşık 3×0,7=2,1 katkısı taşınır ve kısa eğim yaklaşık 4,1 olur."
    elif kind == "wage_isaret":
        prompt, answer = "WAGE1'de deneyimin ücret katsayısı pozitif, deneyimin eğitim yardımcı eğimi negatiftir. Kısa eğitim katsayısı hangi yönde etkilenir?", "Çarpım negatif olduğundan deneyim dışarıda kaldığında eğitim katsayısı aşağı yönlü etkilenebilir. Bu örneklem ayrıştırması nedensel kanıt değildir."
    elif kind == "wage_ayristirma":
        product = float(wage.middle_model.coefficients["exper"]) * wage.auxiliary_model.slope
        prompt, answer = "WAGE1 kısa eğitim katsayısı hangi iki parçaya ayrışır?", f"{format_number(wage.short_model.slope)} = {format_number(float(wage.middle_model.coefficients['educ']))} + ({format_number(float(wage.middle_model.coefficients['exper']))}×{format_number(wage.auxiliary_model.slope)}) = uzun eğitim katsayısı + {format_number(product)}."
    elif kind == "katsayi_degisim_siniri":
        prompt, answer = "Model değişince katsayının değişmesi tek başına neyi kanıtlamaz?", "Tek başına doğru model, eksik değişken yanlılığı veya nedensel etki kanıtlamaz; ekonomik mekanizma ve tasarım değerlendirilmelidir."
    elif kind == "tam_ve_yuksek":
        prompt, answer = "Tam bağlantı ile yüksek fakat tam olmayan bağlantının farkı nedir?", "Tam bağlantıda ayrı katsayılar benzersiz değildir. Yüksek fakat tam olmayan bağlantıda katsayılar hesaplanabilir, ancak ayrı katkılar daha duyarlı olabilir."
    elif kind == "rank":
        prompt, answer = "Rank eksik tasarım matrisi ne anlama gelir?", "Sabit ve açıklayıcı sütunlar tam rank değildir; en az bir sütun diğerlerinin tam doğrusal birleşimidir ve benzersiz katsayı tahminine geçilmez."
    elif kind == "vif_hesabi":
        prompt, answer = "Yardımcı R²=0,8 ise VIF nedir?", "VIF=1/(1−0,8)=5'tir. Bu, diğer açıklayıcılarla doğrusal tekrarın derecesine ilişkin bir uyarı ölçüsüdür."
    elif kind == "vif_siniri":
        prompt, answer = "Yüksek VIF neyi söylemez?", "Tek başına hangi değişkenin silinmesi gerektiğini, modelin yanlış olduğunu veya katsayıların yanlı olduğunu söylemez."
    elif kind == "yuksek_baglanti_yanlilik":
        prompt, answer = "Yüksek fakat tam olmayan bağlantı katsayıları otomatik olarak yanlı yapar mı?", "Hayır. Sıfır koşullu ortalama sağlanıyorsa merkez korunabilir; sorun ayrı katsayıların değişkenliği ve duyarlılığıdır."
    elif kind == "toplam_kararliligi":
        prompt, answer = "Neden katsayı toplamı ayrı katsayılardan daha kararlı kalabilir?", "X1 ve X2'nin ortak hareketi veri tarafından iyi belirlenirken ortak katkının iki değişkene dağılımı zayıf belirlenebilir."
    elif kind == "duyarlilik":
        prompt, answer = "Küçük bir veri değişikliği altında neyi birlikte kontrol edersiniz?", "Ayrı katsayıları, toplamı, R-kareyi, VIF'yi ve rank'ı birlikte karşılaştırırım; tek bir metriğe mekanik karar yüklemem."
    elif kind == "yuksek_r2":
        prompt, answer = "Yüksek R² ayrı katsayıların güvenilir ya da nedensel olduğunu garanti eder mi?", "Hayır. R² ortak örneklem uyumunu özetler; sıfır koşullu ortalamayı veya ayrı katkıların iyi ayrıştırıldığını kanıtlamaz."
    elif kind == "arastirma_sorusu":
        prompt, answer = "Yüksek bağlantı varken değişken seçimi nasıl yapılır?", "Araştırma sorusu, teori, veri aralığı ve duyarlılık birlikte değerlendirilir. İktisadi olarak gerekli kontrol yalnız VIF nedeniyle mekanik biçimde silinmez."
    elif kind == "ovb_ve_baglanti":
        prompt, answer = "Eksik değişken yanlılığı ile yüksek bağlantıyı merkez bakımından ayırın.", "OVB katsayının yanlış merkezde toplanmasına yol açabilir. Yüksek bağlantıda sıfır koşullu ortalama altında merkez doğru kalabilir, fakat ayrı tahminler daha değişken olur."
    elif kind == "nedensel_dil":
        prompt, answer = "“Eğitim ücreti artırır” ifadesini güvenli ilişki diliyle yazın.", "Bu modelde eğitim yılı daha yüksek çalışanların tahmin edilen saatlik ücreti daha yüksektir; gözlemsel veri bu cümleyi tek başına nedensel etki yapmaz."
    else:
        prompt, answer = "Homoskedastisite ve normallik bu konunun neresindedir?", "Yansızlık için gerekli değildir; belirsizlik hesaplarıyla ilgili ayrıntılar sonraki konuda ele alınacaktır."
    return GeneratedQuestion(kind, prompt, answer, question_index)
