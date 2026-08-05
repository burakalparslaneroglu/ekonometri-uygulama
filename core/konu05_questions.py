"""Konu 05 için model sonucuna bağlı deterministik sorular."""

from __future__ import annotations

from core.data_registry import RegressionModelSpec, VariableMetadata
from core.model_utils import format_number
from core.multiple_regression_utils import MultipleOLSResult, multiple_observation_result, profile_prediction_difference
from core.question_engine import GeneratedQuestion, cycle_question_type

KONU05_QUESTION_TYPES = (
    "roller", "denklem", "ceteris_paribus", "birim_fark", "profil_farki", "katkilar",
    "gozlem", "artik_isareti", "sabitler", "basit_coklu", "fwl_adim", "fwl_esitlik",
    "duzeltilmis_r2", "r2_karsilastirma", "makale_tablosu", "kontroller", "hprice_birim",
    "nedensel_dil", "kontrol_secimi", "hepsini_ekle",
)


def generate_konu05_question(model_id: str, question_index: int, result: MultipleOLSResult, spec: RegressionModelSpec, metadata: dict[str, VariableMetadata]) -> GeneratedQuestion:
    """Aynı model kimliği ve sıra için aynı Konu 05 sorusunu üretir."""
    kind = cycle_question_type(model_id, question_index, KONU05_QUESTION_TYPES, namespace="konu05")
    focal, controls = spec.focal_explanatory, spec.controls
    focal_label, y_label = metadata[focal].label, metadata[spec.dependent].label
    control_text = ", ".join(metadata[item].label for item in controls) or "modelde başka açıklayıcı değişken yoktur"
    beta = float(result.coefficients[focal])
    position = question_index % result.nobs
    observation = multiple_observation_result(result, position)
    if kind == "roller":
        prompt = "Bu modelde bağımlı değişken, temel açıklayıcı değişken ve kontroller hangileridir?"
        answer = f"Bağımlı değişken {y_label}; temel açıklayıcı değişken {focal_label}; kontroller {control_text}."
    elif kind == "denklem":
        terms = " ".join(f"+ {format_number(float(result.coefficients[key]))} {key}" for key in result.explanatory)
        prompt = "Tahmin edilen çoklu regresyon denklemini yazın."
        answer = f"Ŷ = {format_number(float(result.coefficients['const']))} {terms}."
    elif kind == "ceteris_paribus":
        prompt = f"{focal_label} katsayısını tam ceteris paribus cümlesiyle yorumlayın."
        answer = f"{control_text} sabitken, {focal_label} bir {metadata[focal].unit} daha yüksek birimin tahmin edilen {y_label} değeri örneklemde {format_number(abs(beta))} {metadata[spec.dependent].unit} {'daha yüksek' if beta >= 0 else 'daha düşüktür'}; bu nedensel iddia değildir."
    elif kind == "birim_fark":
        prompt = f"{focal_label} 4 birim farklıysa, diğer açıklayıcı değişkenler aynıyken tahmin farkı nedir?"
        answer = f"4 × {format_number(beta)} = {format_number(4 * beta)} {metadata[spec.dependent].unit}. {control_text} sabittir."
    elif kind in {"profil_farki", "katkilar"}:
        a = {key: float(result.design_data.iloc[0][key]) for key in result.explanatory}
        b = dict(a); b[focal] += 2
        comparison = profile_prediction_difference(result, a, b)
        prompt = "Yalnızca temel açıklayıcı değişken iki birim arttığında iki profilin tahmin farkı nedir?" if kind == "profil_farki" else "Profil farkını katsayı katkılarına ayırın."
        answer = f"B−A={format_number(float(comparison['difference']))}. Katkılar: " + ", ".join(f"{key}: {format_number(value)}" for key, value in comparison['contributions'].items()) + "."
    elif kind == "gozlem":
        prompt = f"{position + 1}. gözlem için tahmin edilen değer ve artığı bulun."
        answer = f"Ŷ={format_number(float(observation['predicted']))}; û=Y−Ŷ={format_number(float(observation['residual']))} {metadata[spec.dependent].unit}."
    elif kind == "artik_isareti":
        prompt = f"{position + 1}. gözlemin artığının işareti ne söyler?"
        answer = "Artık pozitiftir; gerçekleşen değer tahminin üzerindedir." if float(observation['residual']) > 0 else "Artık negatiftir; gerçekleşen değer tahminin altındadır."
    elif kind == "sabitler":
        prompt = f"{focal_label} katsayısı yorumlanırken hangi değişkenler sabit tutulur?"
        answer = f"{control_text} sabit tutulur; bu, gerçek dünyada değişkenleri fiziksel olarak eşitlemek demek değildir."
    elif kind == "basit_coklu":
        prompt = "Basit ve çoklu modelde aynı katsayı neden farklı olabilir?"
        answer = "Modeller farklı karşılaştırmalar kurar: çoklu model, içerdiği kontrollerin doğrusal katkısını ayırır. Katsayı değişimi tek başına yanlılık veya nedensellik kanıtı değildir."
    elif kind in {"fwl_adim", "fwl_esitlik"}:
        prompt = "Kısmi regresyonun üç adımını sıralayın." if kind == "fwl_adim" else "Kısmi regresyon eğimi tam modeldeki katsayıyla nasıl ilişkilidir?"
        answer = "Önce Y'yi kontrollere, sonra temel X'i aynı kontrollere göre tahmin edip iki artığı ilişkilendiririz." if kind == "fwl_adim" else "Artıklar regresyonunun eğimi tam çoklu modeldeki temel katsayıya eşittir; bu yalnızca gözlenen kontrollerin doğrusal katkısını ayırır."
    elif kind == "duzeltilmis_r2":
        prompt = "R-kare ve düzeltilmiş R-kare arasındaki temel fark nedir?"
        answer = "Aynı örneklemde yeni değişken eklemek R-kareyi düşürmez; düzeltilmiş R-kare ek katsayı kullanımını dikkate alır ve düşebilir."
    elif kind == "r2_karsilastirma":
        prompt = "İki modelin R-kareleri ne zaman doğrudan karşılaştırılabilir?"
        answer = "Yalnızca bağımlı değişken ve kullanılan örneklem aynıysa. wage ile ln(wage) modelleri mekanik olarak sıralanmaz."
    elif kind == "makale_tablosu":
        prompt = "Makale tipi regresyon tablosunu hangi sırayla okursunuz?"
        answer = "Bağımlı değişkeni, gözlem sayısını, açıklayıcıları ve kontrolleri, katsayı birimlerini, sonra R-kare ve düzeltilmiş R-kareyi kontrol ederim."
    elif kind == "kontroller":
        prompt = "“Kontroller: Evet” satırı neden tek başına yetersizdir?"
        answer = "Hangi değişkenlerin kontrol edildiğini söylemez; katsayının hangi koşullu karşılaştırmayı temsil ettiğini anlayabilmek için kontroller açıkça yazılmalıdır."
    elif kind == "hprice_birim":
        prompt = "HPRICE1'de sqrft100 ve lotsize1000 hangi ölçekleri temsil eder?"
        answer = "sqrft100 100 kare fit, lotsize1000 1.000 kare fit birimidir; dönüşümler sırasıyla sqrft/100 ve lotsize/1000'dir."
    elif kind == "nedensel_dil":
        prompt = "“Eğitim ücreti artırır” cümlesini güvenli ilişki diliyle yeniden yazın."
        answer = "Modeldeki diğer açıklayıcı değişkenler sabitken eğitim yılı daha yüksek çalışanların tahmin edilen saatlik ücreti örneklemde daha yüksektir; bu tek başına nedensel sonuç değildir."
    elif kind == "kontrol_secimi":
        prompt = "Bir kontrol değişkenini değerlendirirken hangi temel sorular sorulur?"
        answer = "Araştırma sorusu, ekonomik gerekçe, temel değişkenle ve Y ile ilişki, zamansal sıra, ölçüm kalitesi ve sabit tutmanın uygunluğu değerlendirilir."
    else:
        prompt = "“Mümkün olan her değişkeni modele eklemek her zaman daha iyidir.” doğru mu?"
        answer = "Hayır. Kontroller araştırma sorusu, teori ve zamansal sıraya göre seçilir; daha çok değişken otomatik olarak daha güvenilir veya nedensel model oluşturmaz."
    return GeneratedQuestion(kind, prompt, answer, question_index)
