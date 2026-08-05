"""Konu 12 için deterministik heteroskedastisite soruları."""
from __future__ import annotations
from dataclasses import dataclass
from core.question_engine import GeneratedQuestion,cycle_question_type
from core.robust_inference_utils import HeteroskedasticityTest,RobustInferenceResult,SimulationSummary
KONU12_QUESTION_TYPES=("variance","coefficient","diagnostic","bp_null","hc","joint","simulation","limits","bp_steps","bp_lm","white_terms","decision_change")
@dataclass(frozen=True)
class Konu12QuestionContext:
    robust:RobustInferenceResult; bp:HeteroskedasticityTest; simulation:SimulationSummary
def generate_konu12_question(context:Konu12QuestionContext,model_id:str,question_index:int)->GeneratedQuestion:
    """Konu 12'nin hesaplanmış sonuçlarına bağlı soru üretir."""
    kind=cycle_question_type(model_id,question_index,KONU12_QUESTION_TYPES,namespace="konu12"); r=context.robust
    items={
      "variance":("Heteroskedastisite neyin değişmesidir?","Koşullu hata varyansının Var(u|X) açıklayıcı değişken düzeylerine göre değişmesidir; koşullu ortalama ile aynı kavram değildir."),
      "coefficient":("HC1 kullanmak OLS katsayısını değiştirir mi?","Hayır. Nokta tahmini aynı EKK katsayısıdır; kovaryans, standart hata, t, p ve güven aralığı değişebilir."),
      "diagnostic":("Artık–tahmin grafiği neyi kanıtlar?","Yayılım örüntüsü için tanı sağlar, ancak tek başına heteroskedastisiteyi, doğru modeli veya nedenselliği kanıtlamaz."),
      "bp_null":("Breusch–Pagan testinin sıfır hipotezi nedir?",context.bp.null_hypothesis+" Reddedilememe homoskedastisitenin kanıtı değildir."),
      "hc":("HC3 her zaman en iyi seçenek midir?","Hayır. HC0–HC3 farklı sonlu örneklem düzeltmeleridir; kullanılan tür açıkça raporlanmalıdır."),
      "joint":("Robust ortak testin hipotezi değişir mi?","Hayır; aynı doğrusal kısıtlar sınanır, yalnız kovaryans hesabı değişir."),
      "simulation":("Benzetimde robust kapsama neyi ölçer?",f"Bu DGP'de HC1 %95 aralık kapsaması %{100*context.simulation.hc1_coverage:.1f}; tekrar örneklemede aralıkların gerçek eğimi kapsama sıklığını ölçer."),
      "limits":("Robust standart hata hangi sorunları çözmez?","Eksik değişken yanlılığı, yanlış fonksiyonel biçim, küme bağımlılığı ve nedensel tasarım sorunlarını çözmez."),
      "bp_steps":("Breusch–Pagan testinin yardımcı regresyon adımları nelerdir?","EKK artıklarını al, karelerini oluştur, û²'yi özgün açıklayıcılara regrese et ve yardımcı R²'den LM=nR² hesapla."),
      "bp_lm":("BP testinde LM ve serbestlik derecesi nasıl bulunur?",f"LM=nR²_aux; q yardımcı regresyondaki sabit dışı açıklayıcı sayısıdır. Bu modelde q={context.robust.ols.n_explanatory}."),
      "white_terms":("White yardımcı regresyonu hangi terimleri içerir?","Özgün açıklayıcıların düzeyleri, kareleri ve ikili çapraz çarpımları; bağımlı ya da sabit sütunlar rank denetimiyle çıkarılır."),
      "decision_change":("Geleneksel ve robust p kararının değişmesi katsayıyı küçültür mü?","Hayır. Aynı nokta tahmini korunur; belirsizlik ölçüsü ve buna bağlı test kararı değişir."),
    }
    prompt,answer=items[kind]; return GeneratedQuestion(kind,prompt,answer,question_index)
