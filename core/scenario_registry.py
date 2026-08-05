"""Konu 01'in ders notuna dayalı senaryo ve soru kayıtları."""

from __future__ import annotations

from dataclasses import dataclass

from core.question_engine import GeneratedQuestion, cycle_question_type


KONU01_QUESTION_TYPES = (
    "missing_component",
    "model_layers",
    "error_term",
    "research_stages",
    "empirical_purpose",
    "safe_language",
    "wage1_metadata",
)


@dataclass(frozen=True)
class Scenario:
    """Ders notundan türetilen kısa, sabit bir öğretim senaryosu."""

    key: str
    title: str
    prompt: str


KONU01_SCENARIOS = {
    "education_wage": Scenario(
        key="education_wage",
        title="Eğitim ve ücret",
        prompt="İncelenen çalışan örnekleminde eğitim yılı ile saatlik ücret arasındaki bağlantı",
    ),
    "advertising_sales": Scenario(
        key="advertising_sales",
        title="Reklam ve satış",
        prompt="Firmalarda reklam harcaması ile satışlar arasındaki bağlantı",
    ),
}


def generate_konu01_question(model_id: str, question_index: int) -> GeneratedQuestion:
    """Konu 01 kapsamına uygun, deterministik uygulama sorusu üretir."""
    question_type = cycle_question_type(model_id, question_index, KONU01_QUESTION_TYPES, namespace="konu01")
    if question_type == "missing_component":
        prompt = (
            "“Eğitim ile ücret arasındaki ilişki nedir?” sorusu ölçülebilir bir araştırma sorusu için "
            "hangi temel bileşenleri açıkça belirtmemektedir?"
        )
        answer = (
            "En az gözlem birimi, sonuç değişkeni, temel açıklayıcı değişken, anakütle/yer/dönem ve araştırma amacı "
            "açıklaştırılmalıdır. Örneğin amaç yalnızca örneklem ilişkisini betimlemek olabilir."
        )
    elif question_type == "model_layers":
        prompt = (
            "Eğitimin becerileri artırıp ücretle ilişkili olabileceğini açıklayan ifade; ücret = β₀ + β₁ eğitim "
            "ifadesi; ve ücretᵢ = β₀ + β₁ eğitimᵢ + uᵢ ifadesi sırasıyla hangi model düzeylerini gösterir?"
        )
        answer = "Sırasıyla ekonomik model, matematiksel model ve ekonometrik modeldir."
    elif question_type == "error_term":
        prompt = "Ücret–eğitim ekonometrik modelindeki hata teriminde yer alabilecek üç unsur yazınız."
        answer = (
            "Örnekler: bireysel yetenek ve motivasyon, meslek veya sektör, işverenin ücret politikası, "
            "ölçülemeyen eğitim kalitesi ve ölçüm hataları. Hata terimi doğrudan gözlenmez ve yalnızca veri giriş hatası değildir."
        )
    elif question_type == "research_stages":
        prompt = "Ampirik araştırmanın başlangıçtaki üç aşamasını doğru sırayla yazınız."
        answer = (
            "Araştırma sorusunu ve amacı belirleme; ekonomik mekanizmayı kurma; anakütleyi, veriyi ve değişkenleri tanımlama. "
            "Araştırma süreci geri bildirimli olduğundan aşamalar gerektiğinde yeniden gözden geçirilebilir."
        )
    elif question_type == "empirical_purpose":
        prompt = "“Bir şehirde kiraların ortanca değeri nedir?” sorusu hangi ampirik amacı temsil eder?"
        answer = "Bu bir betimleme sorusudur; örneklemin veya anakütlenin düzeyini ve dağılımını özetlemeyi amaçlar."
    elif question_type == "safe_language":
        prompt = (
            "WAGE1 örnekleminde eğitim yılı yüksek çalışanların ücreti daha yüksek görünüyorsa, "
            "hangi ifade güvenlidir: “eğitim ücreti kesin olarak artırır” mı, yoksa ilişki dili mi?"
        )
        answer = (
            "Güvenli ifade ilişki dilidir: “Örneklemde eğitim yılı daha yüksek çalışanların saatlik ücreti ortalama olarak daha yüksektir.” "
            "Bu gözlemsel ilişki tek başına nedensel etki göstermez."
        )
    else:
        prompt = "WAGE1 veri setinde gözlem birimi nedir; wage ve educ hangi ölçü birimleriyle tanımlanır?"
        answer = (
            "Gözlem birimi çalışandır. wage saatlik ücret olup ABD doları/saat cinsindendir; educ tamamlanan eğitim yılıdır."
        )
    return GeneratedQuestion(question_type=question_type, prompt=prompt, answer=answer, index=question_index)
