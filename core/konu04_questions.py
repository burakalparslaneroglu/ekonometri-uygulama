"""Konu 04 için model sonucuna bağlı deterministik sorular."""

from __future__ import annotations

from core.model_utils import FunctionalForm, ObservationDecomposition, SimpleOLSResult, SumOfSquares, format_number
from core.question_engine import GeneratedQuestion, cycle_question_type


KONU04_QUESTION_TYPES = (
    "toplam_sapma", "sapma_ayristirma", "kareler_tanimi", "eksik_kareler", "r_kare_mkt",
    "r_kare_hkt", "r_kare_yorumu", "yanlis_r_kare_yorumu", "birim_katsayi", "birim_r_kare",
    "fonksiyonel_bicim", "fonksiyonel_yorum", "log_uygunluk", "r_kare_karsilastirma",
)


def generate_konu04_question(
    model_id: str, question_index: int, result: SimpleOLSResult, sums: SumOfSquares,
    decomposition: ObservationDecomposition, form: FunctionalForm, dependent_label: str, explanatory_label: str,
) -> GeneratedQuestion:
    """Konu 04 kapsamındaki soruyu aynı kimlik ve sıra için yeniden üretir."""
    question_type = cycle_question_type(model_id, question_index, KONU04_QUESTION_TYPES, namespace="konu04")
    r2 = sums.model / sums.total
    if question_type == "toplam_sapma":
        prompt = f"Seçili gözlemde {dependent_label}={format_number(decomposition.observed)} ve ortalama={format_number(decomposition.mean_observed)}. Toplam sapma nedir?"
        answer = f"Toplam sapma Yᵢ−Ȳ={format_number(decomposition.observed)}−{format_number(decomposition.mean_observed)}={format_number(decomposition.total_deviation)}'tür."
    elif question_type == "sapma_ayristirma":
        prompt = "Bir gözlemdeki toplam sapma hangi iki bileşene ayrılır?"
        answer = f"Yᵢ−Ȳ=(Ŷᵢ−Ȳ)+ûᵢ={format_number(decomposition.model_deviation)}+{format_number(decomposition.residual_deviation)}={format_number(decomposition.total_deviation)}."
    elif question_type == "kareler_tanimi":
        prompt = "Gözlenen değerlerle tahmin edilen değerler arasındaki kareli farkların toplamı hangi büyüklüktür?"
        answer = "Hata/artık kareleri toplamıdır (HKT): Σûᵢ². EKK'nin en küçük yaptığı ölçüt budur."
    elif question_type == "eksik_kareler":
        prompt = f"TKT={format_number(sums.total)} ve HKT={format_number(sums.error)} ise MKT nedir?"
        answer = f"TKT=MKT+HKT olduğundan MKT={format_number(sums.total)}−{format_number(sums.error)}={format_number(sums.model)}."
    elif question_type == "r_kare_mkt":
        prompt = f"MKT={format_number(sums.model)} ve TKT={format_number(sums.total)} iken R-kareyi hesaplayın."
        answer = f"R²=MKT/TKT={format_number(sums.model)}/{format_number(sums.total)}={format_number(r2)}."
    elif question_type == "r_kare_hkt":
        prompt = f"HKT={format_number(sums.error)} ve TKT={format_number(sums.total)} iken R-kareyi hesaplayın."
        answer = f"R²=1−HKT/TKT=1−{format_number(sums.error)}/{format_number(sums.total)}={format_number(r2)}."
    elif question_type == "r_kare_yorumu":
        prompt = "Bu modelin R-kare değeri örneklem içinde nasıl yorumlanır?"
        answer = f"Örneklemde {dependent_label} değişkeninin ortalama çevresindeki değişiminin yaklaşık %{format_number(100*r2)}'i, {explanatory_label} ile kurulan bu doğrusal model tarafından izlenmektedir. Bu nedensel bir yorum değildir."
    elif question_type == "yanlis_r_kare_yorumu":
        prompt = "“R-kare yüksek olduğundan X, Y'ye neden olur.” ifadesi doğru mudur?"
        answer = "Hayır. R-kare örneklem uyumunu özetler; nedenselliği, modelin mutlaka doğru olduğunu veya iktisadi önemi kanıtlamaz."
    elif question_type == "birim_katsayi":
        prompt = "Y yalnızca 1000 ile çarpılan yeni birimle yazılırsa sabit ve eğim ne olur?"
        answer = "Her iki katsayı 1000 ile çarpılır. Ekonomik ilişki değişmez; yalnızca Y'nin yazıldığı ölçü birimi değişir."
    elif question_type == "birim_r_kare":
        prompt = "Yalnızca doğrusal bir ölçü birimi dönüşümünde R-kare değişir mi?"
        answer = "Hayır. Doğrusal birim dönüşümü tahmin edilen ekonomik ilişkiyi ve R-kareyi değiştirmez."
    elif question_type == "fonksiyonel_bicim":
        prompt = "ln(Y)=β₀+β₁ln(X)+u denkleminin fonksiyonel biçimi nedir?"
        answer = "Log–log biçimidir; eğim katsayısı esneklik olarak okunur."
    elif question_type == "fonksiyonel_yorum":
        prompt = f"Seçili {form} modelinde eğim katsayısı nasıl yorumlanır?"
        if form == "düzey-düzey":
            answer = f"{explanatory_label} bir birim arttığında {dependent_label} tahmin edilen olarak {format_number(result.slope)} birim değişir."
        elif form == "log-düzey":
            answer = f"{explanatory_label} bir birim arttığında {dependent_label} yaklaşık %{format_number(100*result.slope)} değişir; bu ilişki yorumudur."
        elif form == "düzey-log":
            answer = f"{explanatory_label} yüzde 1 arttığında {dependent_label} yaklaşık {format_number(result.slope/100)} birim değişir."
        else:
            answer = f"{explanatory_label} yüzde 1 arttığında {dependent_label} yaklaşık %{format_number(result.slope)} değişir; eğim esnekliktir."
    elif question_type == "log_uygunluk":
        prompt = "Bir değişken sıfır veya negatif değer içeriyorsa doğal logaritması doğrudan alınabilir mi?"
        answer = "Hayır. Doğal logaritma yalnızca pozitif değerlerde tanımlıdır; sorunlu gözlemler görünür biçimde raporlanmalıdır."
    else:
        prompt = "Düzey Y modeli ile logaritmik Y modelinin R-kareleri neden doğrudan sıralanmamalıdır?"
        answer = "Bağımlı değişkenler farklı ölçeklerde olduğundan açıklanan değişkenlik aynı büyüklük değildir. Model seçimi yalnız en yüksek R-kareye indirgenemez."
    return GeneratedQuestion(question_type=question_type, prompt=prompt, answer=answer, index=question_index)
