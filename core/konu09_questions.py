"""Konu 09 için deterministik fonksiyonel biçim soruları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.functional_form_utils import Wage1ModelComparison
from core.question_engine import GeneratedQuestion, cycle_question_type


KONU09_QUESTION_TYPES = (
    "parametre", "olcek_y", "olcek_x", "invariance", "standard_beta", "causal_rank", "log_approx", "log_exact",
    "quadratic", "marginal", "turning", "range", "hierarchy", "joint_f", "centering", "model_selection", "r2_limit", "causality",
    "ols_curve", "beta_squared", "scale_se", "standard_formula", "sample_dependence", "log_multi", "log_negative",
    "elasticity", "quadratic_discrete", "quadratic_sign", "quadratic_zero", "turning_type", "wage_educ", "exper_10",
    "exper_20", "exper_turn", "tenure_turn", "small_square", "m1_r2", "m4_adj_r2", "nested_q", "ssr_comparison",
    "article_expersq", "wrong_form", "high_r2", "centering_correlation", "fitted_invariance",
)


@dataclass(frozen=True)
class Konu09QuestionContext:
    """WAGE1 M1–M4 sonuçlarıyla soru üretim bağlamı."""

    context_id: str
    comparison: Wage1ModelComparison


def generate_konu09_question(context: Konu09QuestionContext, question_index: int) -> GeneratedQuestion:
    """Aynı bağlam ve indeks için aynı Konu 09 sorusunu üretir."""
    kind = cycle_question_type(context.context_id, question_index, KONU09_QUESTION_TYPES, namespace="konu09")
    m4, joint = context.comparison.models["M4"], context.comparison.quadratic_joint_test
    items = {
        "parametre": ("Y=β0+β1X+β2X²+u modeli parametrelerde doğrusal mıdır?", "Evet. X² veri sütunudur; β katsayıları birinci kuvvette yer alır. β1² içeren model parametrelerde doğrusal değildir."),
        "olcek_y": ("Y 100 ile çarpılırsa katsayı ve standart hata ne olur?", "Y'nin katsayıları ve standart hataları 100 ile çarpılır; ilgili t, p, F ve R² değişmez."),
        "olcek_x": ("X 10'a bölünürse X katsayısı ne olur?", "Yeni X birimi eski birimin onda biri olduğundan katsayı ve standart hata 10 ile çarpılır; t, p, F ve R² değişmez."),
        "invariance": ("Ölçek değişiminde hangi çıkarım büyüklükleri değişmez?", "Aynı örneklem ve eşdeğer modelde t, p-değeri, F ve R² değişmez; katsayıların sayısal birimi değişir."),
        "standard_beta": ("Standartlaştırılmış katsayı nasıl elde edilir?", "Ham eğim, X'in standart sapmasıyla çarpılıp Y'nin standart sapmasına bölünür; standart katsayı bir standart sapmalık değişimle ilişkilidir."),
        "causal_rank": ("En büyük standartlaştırılmış katsayı nedensel önem sırası mıdır?", "Hayır. Bu karşılaştırma birimden arındırılmış koşullu ilişkiyi özetler; araştırma tasarımı olmadan nedensel sıralama değildir."),
        "log_approx": ("Log-düzey modelde β=0.06 ve X bir birim artarsa yaklaşık yüzde değişim nedir?", "Yaklaşık değişim %6'dır; bu birinci dereceden yaklaşıktır."),
        "log_exact": ("Log-düzey modelde tam yüzde değişim nasıl bulunur?", "100×[exp(β×ΔX)−1] ile bulunur; yaklaşık yüzde ile özellikle büyük değişimlerde aynı değildir."),
        "quadratic": ("Karesel modelde düzey katsayısı neden tek başına yorumlanmaz?", "X'in etkisi β1+2β2X ile X düzeyine bağlıdır; düzey ve kare katsayıları birlikte okunmalıdır."),
        "marginal": ("Karesel modelde marjinal etki nedir?", "Koşullu tahmin eğiminin belirli X düzeyindeki değeridir: β1+2β2X. Bu ders bağlamında nedensel marjinal etki değildir."),
        "turning": ("Dönüm noktası formülü nedir?", "β2 sıfır değilse −β1/(2β2). β2 negatifse tepe, pozitifse dip oluşturur."),
        "range": ("Dönüm noktası veri aralığı dışındaysa nasıl yorumlanmalıdır?", "Örneklem dışında güçlü yorum yapılmamalıdır; modelin o bölgedeki uzantısı gözlenen veri tarafından desteklenmez."),
        "hierarchy": ("Hiyerarşi ilkesi karesel modelde ne gerektirir?", "X² eklenirse X de modelde kalır; aksi halde katsayıların ve biçimin yorumu bozulur."),
        "joint_f": ("Karesel terimlerin ortak testi neyi sınar?", f"H0: expersq=0 ve tenursq=0. M1–M4 karşılaştırmasında F={joint.f_from_ssr:.3f}; ortak ret yalnız en az bir karesel kısıtın uyumsuz olduğunu gösterir."),
        "centering": ("Merkezleme neyi değiştirir ve neyi değiştirmez?", "Referans noktası ve doğrusal katsayının okunduğu düzey değişir; fitted değerler, artıklar, SSR ve R² eşdeğer modelde değişmez. İçselliği çözmez."),
        "model_selection": ("Model seçimi tek bir ölçüte indirgenebilir mi?", "Hayır. Teori, işaretler, veri aralığı, basitlik, SSR/R²/düzeltilmiş R² ve ortak F birlikte değerlendirilir."),
        "r2_limit": ("Farklı bağımlı değişkenlere ait R² değerleri karşılaştırılabilir mi?", "Hayır. Bağımlı değişken veya dönüşümü değiştiğinde R² karşılaştırması otomatik model seçimi sağlamaz."),
        "causality": ("Fonksiyonel biçim için daha iyi örneklem kanıtı nedensellik kanıtlar mı?", "Hayır. Fonksiyonel biçim ve uyum değerlendirmesi gözlemsel ilişkiyi açıklar; nedensel tasarımın yerini almaz."),
    }
    items.update({
        "ols_curve": ("Eğri görünen model neden EKK ile tahmin edilebilir?", "X² veya ln(X) veri dönüşümüdür; bilinmeyen katsayılar birinci kuvvetteyse model parametrelerde doğrusaldır."),
        "beta_squared": ("β1²X modeli neden bu dersin doğrusal EKK modeli değildir?", "Bilinmeyen parametre karesel girer; model parametrelerde doğrusal değildir."),
        "scale_se": ("Ölçek dönüşümünde standart hata neden katsayıyla birlikte değişir?", "Aynı birim dönüşümü tahminin örnekleme belirsizliği ölçeğine de uygulanır; t oranı bu yüzden değişmez."),
        "standard_formula": ("Standart katsayının ham katsayıyla bağı nedir?", "β* = β × sX / sY. Yalnız yorum için Y ve X birlikte standartlaştırılır."),
        "sample_dependence": ("Standartlaştırılmış katsayılar neden örnekleme bağlıdır?", "Standart sapmalar örneklemden örnekleme değişebilir; katsayı sıralaması da değişebilir."),
        "log_multi": ("Log-düzeyde dört birim değişim için tam yüzde nasıl hesaplanır?", "100×[exp(4β)−1] kullanılır; tek yıllık yaklaşık yüzdeyi dörtle çarpmak yalnız yaklaşıktır."),
        "log_negative": ("β=-0.10 için tam yüzde değişim neden -10 değildir?", "Tam dönüşüm doğrusal değildir: 100×[exp(-0.10)-1] yaklaşık -9.52'dir."),
        "elasticity": ("Log-log katsayısı nasıl yorumlanır?", "X yüzde 1 arttığında Y'nin yaklaşık yüzde β kadar değiştiği esneklik olarak yorumlanır."),
        "quadratic_discrete": ("Karesel modelde X'ten X+1'e tam fark formülü nedir?", "β1+β2(2X+1); bu, türev olan marjinal etkiden küçük farkla ayrılır."),
        "quadratic_sign": ("β2 negatifse eğri ne yapar?", "Aşağı doğru bükülür; marjinal etki X arttıkça azalır."),
        "quadratic_zero": ("β2=0 ise karesel model neye iner?", "X bakımından düz çizgili doğrusal modele iner; dönüm noktası tanımlı değildir."),
        "turning_type": ("Dönüm noktasının tepe mi dip mi olduğu nasıl belirlenir?", "Kare katsayısı negatifse tepe, pozitifse dip olur."),
        "wage_educ": ("M4'te eğitim katsayısı yaklaşık nasıl yüzde yorumlanır?", f"β={m4.coefficients['educ']:.4f}; bir yıl eğitim artışı yaklaşık %{100*m4.coefficients['educ']:.2f}, tam olarak %{100*(np.exp(m4.coefficients['educ'])-1):.2f} daha yüksek ücretle ilişkilidir."),
        "exper_10": ("M4'te 10 yıl deneyimde yaklaşık marjinal yüzde ilişki nedir?", f"Yaklaşık %{100*(m4.coefficients['exper'] + 20*m4.coefficients['expersq']):.2f}.") ,
        "exper_20": ("M4'te 20 yıl deneyimde yaklaşık marjinal yüzde ilişki nedir?", f"Yaklaşık %{100*(m4.coefficients['exper'] + 40*m4.coefficients['expersq']):.2f}.") ,
        "exper_turn": ("Deneyim dönüm noktası yaklaşık kaç yıldır?", f"−β1/(2β2) ile yaklaşık {-m4.coefficients['exper']/(2*m4.coefficients['expersq']):.2f} yıldır; veri aralığıyla birlikte okunmalıdır."),
        "tenure_turn": ("Kıdem dönüm noktası yaklaşık kaç yıldır?", f"Yaklaşık {-m4.coefficients['tenure']/(2*m4.coefficients['tenursq']):.2f} yıldır; örneklem dışıysa güçlü yorum yapılmaz."),
        "small_square": ("Küçük kare katsayısı neden önemsiz olmak zorunda değildir?", "X² ile çarpıldığı için X büyüdükçe katkısı artar; marjinal etki ve veri aralığı birlikte değerlendirilir."),
        "m1_r2": ("M1 R² neyi gösterir?", f"M1 için R²={context.comparison.models['M1'].r_squared:.3f}; yalnız aynı bağımlı değişken ve örneklemde uyum karşılaştırmasının bir parçasıdır."),
        "m4_adj_r2": ("M4 düzeltilmiş R² neyi ek olarak hesaba katar?", f"M4 için düzeltilmiş R²={m4.adjusted_r_squared:.3f}; ek terimlerin karmaşıklığını kısmen cezalandırır."),
        "nested_q": ("M1 ile M4 arasındaki nested karşılaştırmada q kaçtır?", f"İki karesel terim eklendiği için q={joint.q}.") ,
        "ssr_comparison": ("M4'ün SSR'si M1'den neden küçük/eşittir?", "M4, M1'i kare katsayılarını sıfıra eşitleyerek içerir; daha geniş EKK modelinin SSR'si artamaz."),
        "article_expersq": ("Makale tablosundaki expersq satırı neyi bildirir?", "Deneyim karesinin katsayısını bildirir; deneyim düzey terimiyle birlikte fonksiyonel biçimi belirler."),
        "wrong_form": ("Yanlış fonksiyonel biçim hangi örüntüyü üretebilir?", "Eğri ortalama ilişkiye düz çizgi uydurmak bazı bölgelerde sistematik yüksek, bazı bölgelerde düşük tahmin üretebilir."),
        "high_r2": ("Yüksek R² modelin kesin doğru olduğunu gösterir mi?", "Hayır. R² doğru biçimi, nedenselliği veya örneklem dışı tahmini garanti etmez."),
        "centering_correlation": ("Merkezleme X ile X² korelasyonunu ne yapabilir?", "Azaltabilir; bu sayı yorumlamayı kolaylaştırabilir ama temel ekonomik ilişkiyi değiştirmez."),
        "fitted_invariance": ("Eşdeğer merkezlemede fitted değerler neden değişmez?", "Yeni sütunlar ham X ve X²'nin doğrusal dönüşümüdür; aynı tahmin uzayını temsil eder."),
    })
    prompt, answer = items[kind]
    return GeneratedQuestion(kind, prompt, answer, question_index)
