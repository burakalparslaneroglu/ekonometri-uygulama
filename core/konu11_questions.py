"""Konu 11 için deterministik etkileşim soruları."""
from __future__ import annotations
from dataclasses import dataclass
from core.interaction_utils import ConditionalDifference, InteractionModelSummary
from core.question_engine import GeneratedQuestion, cycle_question_type

KONU11_QUESTION_TYPES=("parallel","slope","gap","centering","joint","syntax","boundary","structures","covariance","three_tests")

@dataclass(frozen=True)
class Konu11QuestionContext:
    """Soru üretimi için Konu 11 model çıktıları."""
    wage: InteractionModelSummary
    gap_at_12: ConditionalDifference

def generate_konu11_question(context: Konu11QuestionContext, model_id:str, question_index:int)->GeneratedQuestion:
    """Etkileşim modelinin hesaplanmış değerlerinden soru üretir."""
    kind=cycle_question_type(model_id,question_index,KONU11_QUESTION_TYPES,namespace="konu11"); w=context.wage
    items={
      "parallel":("Additif kukla modeli ne varsayar?","Kukla yalnız sabiti değiştirir; iki grubun X eğimleri paraleldir."),
      "slope":("Kadın grubunun eğitim eğimi nasıl hesaplanır?",f"D=1 eğimi β_educ12+γ_interaction={w.lines.slope_one:.4f}; etkileşim katsayısı tek başına D=1 toplam eğimi değildir."),
      "gap":("12 yıl eğitimde koşullu grup farkı nedir?",f"Merkezleme nedeniyle bu fark female katsayısıdır: {context.gap_at_12.estimate:.4f}; p={context.gap_at_12.p_value:.4f}."),
      "centering":("Merkezleme neyi değiştirir?","Seçilen merkezde ana kukla katsayısının anlamını değiştirir; fitted değerleri, artıkları, R²'yi ve etkileşim katsayısını değiştirmez."),
      "joint":("İki doğrunun tümüyle eşitliği nasıl sınanır?",f"Ana kukla ve etkileşim katsayısının birlikte sıfır olduğu F testiyle: F={w.joint_f:.3f}, p={w.joint_p_value:.4f}."),
      "syntax":("Python formülünde * ile : arasındaki fark nedir?","D*X ana etkileri ve D:X etkileşimini birlikte açar; D:X yalnız etkileşimi belirtir ve ana etkiler ayrıca yazılmalıdır."),
      "boundary":("Kesişim noktası örneklem dışında ise ne yapılır?","Bu nokta için güçlü yorum yapılmaz; ekstrapolasyon veri desteğinin dışına taşar."),
      "structures":("Dört grup regresyon yapısını hangi iki parametre ayırır?","Ana kukla γ₀ sabit farkını, etkileşim γ₁ eğim farkını belirler; ikisinin sıfır olup olmaması dört yapıyı verir."),
      "covariance":("Koşullu farkın SH hesabında neden covariance terimi vardır?","γ̂₀ ve γ̂₁ aynı modelden tahmin edildiği için bağımsız kabul edilemez; 2XCov(γ̂₀,γ̂₁) varyansa eklenir."),
      "three_tests":("Etkileşim modelindeki üç ayrı araştırma sorusu nedir?","Eğim eşitliği γ₁=0, merkezde fark γ₀=0 ve iki ilişkinin bütünüyle eşitliği γ₀=γ₁=0 farklı testlerdir."),
    }
    prompt,answer=items[kind]
    return GeneratedQuestion(kind,prompt,answer,question_index)
