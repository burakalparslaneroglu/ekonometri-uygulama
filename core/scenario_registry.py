"""Konu 01'in ders notuna dayalı senaryo ve soru kayıtları."""

from __future__ import annotations

from dataclasses import dataclass

from core.question_engine import GeneratedQuestion, cycle_question_type
from core.group_comparison_utils import GroupComparison


KONU01_QUESTION_TYPES = (
    "missing_component",
    "model_layers",
    "error_term",
    "research_stages",
    "empirical_purpose",
    "safe_language",
    "wage1_metadata",
)

KONU02_QUESTION_TYPES = (
    "data_structure",
    "observation_unit",
    "time_dimension",
    "panel_or_pooled",
    "structure_or_collection",
    "observational_or_experimental",
    "safe_language",
    "ceteris_paribus",
    "causal_threat",
    "group_difference",
    "appropriate_interpretation",
    "generalizability",
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


def generate_konu02_question(
    model_id: str,
    question_index: int,
    group_comparison: GroupComparison,
) -> GeneratedQuestion:
    """Konu 02 kapsamındaki veri ve nedensellik sorusunu deterministik üretir."""
    question_type = cycle_question_type(model_id, question_index, KONU02_QUESTION_TYPES, namespace="konu02")
    if question_type == "data_structure":
        prompt = "Aynı 300 firmanın 2018–2025 yıllarında tekrar gözlendiği dosyanın veri yapısı nedir?"
        answer = "Panel veridir; ayırt edici özellik aynı firmaların birden fazla dönemde tekrar gözlenmesidir."
    elif question_type == "observation_unit":
        prompt = "WAGEPAN dosyasında bir satır yalnızca çalışanı mı, yoksa hangi birleşimi temsil eder?"
        answer = "Bir satır çalışan-yıl birleşimini temsil eder. Aynı çalışan farklı yıllarda yeniden görünür."
    elif question_type == "time_dimension":
        prompt = "PHILLIPS dosyasında yıl neden sadece bir etiket değil, zaman boyutunun parçasıdır?"
        answer = "Her satır bir yılı temsil eder ve yılların ardışık sırası ekonomik bilgi taşır; veri yıllık zaman serisidir."
    elif question_type == "panel_or_pooled":
        prompt = "1978 ve 1985'te farklı çalışan örneklemleri seçilmiş, aynı kişiler izlenmemiştir. Bu yapı panel mi, havuzlanmış yatay kesit mi?"
        answer = "Havuzlanmış yatay kesittir. Birden fazla dönem vardır; fakat aynı çalışanların tekrar gözlendiği gösterilmemiştir."
    elif question_type == "structure_or_collection":
        prompt = "“Panel veri” ile “deneysel veri” aynı tür sınıflandırma mıdır?"
        answer = "Hayır. Panel veri, satırların ve zamanın nasıl düzenlendiğini; deneysel veri ise uygulama atamasının nasıl üretildiğini açıklar."
    elif question_type == "observational_or_experimental":
        prompt = "Katılımcıların eğitim ve kontrol grubuna rastgele atanması hangi veri üretim biçimine işaret eder?"
        answer = "Deneysel veri üretim biçimine işaret eder. Rastgele atamanın gerçekten uygulanıp uygulanmadığı yine kontrol edilmelidir."
    elif question_type == "safe_language":
        prompt = "Gözlemsel WAGE1 örnekleminde eğitim yılı yüksek çalışanların ücreti daha yüksek görünüyorsa, hangi yorum güvenlidir?"
        answer = "“Örneklemde eğitim yılı ile saatlik ücret arasında pozitif bir ilişki gözlenmektedir” ifadesi güvenlidir. Bu görünüm tek başına nedensel etki göstermez."
    elif question_type == "ceteris_paribus":
        prompt = "Ceteris paribus düşüncesi “gerçekte her şey sabittir” mi demektir?"
        answer = "Hayır. Diğer ilgili koşullar aynıyken bir değişkenin sonuçla bağlantısını düşünmeye yarayan kavramsal karşılaştırmadır."
    elif question_type == "causal_threat":
        prompt = "Daha yüksek satış beklentisi firmaları daha fazla reklam vermeye yöneltiyorsa, reklamın satışla bağlantısında hangi sorun öne çıkar?"
        answer = "Ters nedensellik öne çıkar: sonuçla ilgili beklenti, açıklayıcı değişkeni de etkiliyor olabilir."
    elif question_type == "group_difference":
        prompt = f"JTRAIN2'de eğitim grubunun ortalama 1978 reel kazancı ile kontrol grubunun ortalama 1978 reel kazancı arasındaki fark nedir?"
        answer = (
            f"Eğitim grubu eksi kontrol grubu farkı {group_comparison.difference:.3f} {group_comparison.unit}'dır. "
            "Bu, iki grubun ortalamalarının betimsel farkıdır."
        )
    elif question_type == "appropriate_interpretation":
        prompt = "WAGE1 ve JTRAIN2 karşılaştırmalarından hangisinde, tasarım koşulları sağlanırsa, daha güçlü nedensel yorum düşünülebilir?"
        answer = "JTRAIN2'de rastgele atama gerçekten uygulanmış ve önemli uygulama sorunu yoksa grup farkı daha güçlü yorumlanabilir. WAGE1 için güvenli dil gözlenen ilişkidir."
    else:
        prompt = "Bir deney örneklemindeki grup farkı, otomatik olarak tüm çalışanlar ve tüm dönemler için geçerli sayılabilir mi?"
        answer = "Hayır. Sonucun başka kişi, kurum, yer veya dönemlere genellenebilirliği örneklem kapsamına ve bağlama bağlıdır."
    return GeneratedQuestion(question_type=question_type, prompt=prompt, answer=answer, index=question_index)
