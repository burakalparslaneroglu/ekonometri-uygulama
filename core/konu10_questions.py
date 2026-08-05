"""Konu 10 için model sonuçlarından türetilen deterministik sorular."""
from __future__ import annotations

from dataclasses import dataclass

from core.categorical_regression_utils import RawControlledComparison
from core.question_engine import GeneratedQuestion, cycle_question_type

KONU10_QUESTION_TYPES = ("reference", "raw_gap", "conditional_gap", "log_percent", "dummy_trap", "joint_test", "causality", "numeric_code", "coding_reverse", "no_intercept")

@dataclass(frozen=True)
class Konu10QuestionContext:
    """Konu 10 sorularının sayılarını taşıyan bağlam."""
    comparison: RawControlledComparison
    region_p_value: float

def generate_konu10_question(context: Konu10QuestionContext, model_id: str, question_index: int) -> GeneratedQuestion:
    """Aynı bağlam ve sırada aynı soruyu üretir."""
    kind = cycle_question_type(model_id, question_index, KONU10_QUESTION_TYPES, namespace="konu10")
    c = context.comparison
    items = {
        "reference": ("Referans grup ne anlama gelir?", "Referans grup yalnızca katsayıların karşılaştırıldığı kodlama tabanıdır; normal veya üstün bir grup değildir."),
        "raw_gap": ("WAGE1'de ham kadın-erkek ücret farkı nedir?", f"Kadın eksi erkek ham farkı {c.raw_difference:.4f} dolar/saattir. Bu gözlemsel fark tek başına nedensel ayrımcılık tahmini değildir."),
        "conditional_gap": ("Kontroller eklendiğinde female katsayısı neyi özetler?", f"Eğitim, deneyim ve kıdem sabit tutulduğunda tahmin edilen düzey farkı {c.controlled_level_difference:.4f}'tür; bu koşullu bir ilişkidir."),
        "log_percent": ("Log ücret modelinde female katsayısını tam yüzdeye çeviriniz.", f"100(exp(δ)-1) ile tam yüzde farkı %{c.exact_log_percent:.2f}'dir; δ×100 yalnızca yaklaşıktır."),
        "dummy_trap": ("Sabit terim varken tüm m kategori kuklası neden kullanılamaz?", "Çünkü 1=D1+…+Dm tam doğrusal bağlantı yaratır. Sabitle m−1 kukla veya sabitsiz m kukla kullanılabilir."),
        "joint_test": ("Bir kategori için tekli t testi ile ortak F testi nasıl ayrılır?", "Tekli t testi bir katsayıyı, ortak F testi kategoriye ait bağımsız katsayıların tümünü birlikte sınar."),
        "causality": ("Gruplar arası tahmin edilen fark nedensel etki midir?", "Hayır. Araştırma tasarımı gerekli tanımlama koşullarını sağlamadıkça bulgu ilişki ve tahmin edilen fark olarak yorumlanır."),
        "numeric_code": ("A=1, B=2, C=3 tek sayısal eğimi hangi kısıtı dayatır?", "A→B ile B→C tahmin farkını eşit ve sıralı kabul eder; kategori etiketlerinde bu genellikle gerekçesizdir."),
        "coding_reverse": ("0/1 kodu ters çevrilince ne değişir?", "Sabit ve kukla katsayısının işaret/yorumu değişir; fitted değerler, artıklar, SSR ve R² değişmez."),
        "no_intercept": ("Sabitsiz modelde bütün kategori kuklaları neyi verir?", "Her katsayı ilgili kategorinin sabitini veya yalnız kategori modeli söz konusuysa grup ortalamasını verir; tasarım tam ranklı olabilir."),
    }
    prompt, answer = items[kind]
    return GeneratedQuestion(kind, prompt, answer, question_index)
