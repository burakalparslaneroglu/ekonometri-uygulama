"""Konu 07 için model sonucuna bağlı, deterministik öğretim soruları."""

from __future__ import annotations

from dataclasses import dataclass

from core.question_engine import GeneratedQuestion, cycle_question_type
from core.regression_inference_utils import CoefficientInference, OLSInferenceResult, format_p_value


KONU07_QUESTION_TYPES = (
    "nokta_ve_belirsizlik", "ayni_katsayi_farkli_sh", "ornekleme_dagilimi",
    "standart_hata", "uc_degiskinlik", "sh_sezgisi", "null_sifir_degil",
    "iki_tarafli_hipotez", "yon_onceden", "alpha_tip1", "tip2_hata",
    "test_istatistigi", "df_resid", "kritik_deger", "p_degeri_anlami",
    "p_sifir_degildir", "guven_araligi", "guven_duzeyi", "test_ga",
    "model_sayisi", "deneyim_kuyruk", "python_cikti", "nonrobust",
    "yildiz", "iktisadi_onem", "olcekleme", "nedensellik", "raporlama",
    "buyuk_katsayi", "kucuk_katsayi", "artik_degiskinligi", "x_degiskinligi",
    "r_kare_sezgisi", "orneklem_buyuklugu", "negatif_hipotez", "karar_dili",
    "negatif_t", "p_alpha_karari", "p_032", "ga_kurma", "sh_yari",
    "ga_null_uyeligi", "wage_egitim_null", "wage_deneyim", "ci_sutunlari",
    "makale_parantez", "wage_lwage", "hprice_olcek", "bdrms_belirsizlik",
    "buyuk_orneklem", "kucuk_orneklem", "kapsama_yorumu", "kapsama_kacirma",
    "yanlis_p", "yanlis_h0", "tek_katsayi_siniri",
)


@dataclass(frozen=True)
class Konu07QuestionContext:
    """Soru metnini hesaplanan model ve test sonucuna bağlar."""

    context_id: str
    model_id: str
    result: OLSInferenceResult
    inference: CoefficientInference
    coefficient_label: str
    dependent_label: str
    explanatory_unit: str


def _number(value: float, decimals: int = 3) -> str:
    """Öğrenci görünümü için yalın sayısal biçim üretir."""
    return f"{value:.{decimals}f}"


def generate_konu07_question(context: Konu07QuestionContext, question_index: int) -> GeneratedQuestion:
    """Aynı bağlam ve indeks için aynı soru ve çözümü üretir."""
    kind = cycle_question_type(context.context_id, question_index, KONU07_QUESTION_TYPES, namespace="konu07")
    test, result = context.inference, context.result
    beta, se = _number(test.estimate), _number(test.standard_error)
    extra_questions = {
        "buyuk_katsayi": ("Buyuk bir katsayi neden otomatik olarak kesin degildir?", "Kesinlik katsayi buyuklugunden degil, standart hata ve ornekleme belirsizliginden okunur."),
        "kucuk_katsayi": ("Kucuk bir katsayi neden otomatik olarak onemsiz degildir?", "Onem, olcu birimi ve anlamli degisim miktariyla degerlendirilir."),
        "artik_degiskinligi": ("Artik degiskenligi yukselirse SH icin beklenen yon nedir?", "Diger kosullar ayniyken SH genellikle yukseltir."),
        "x_degiskinligi": ("Xj degiskenligi yukselirse SH icin beklenen yon nedir?", "Diger kosullar ayniyken SH genellikle duser."),
        "r_kare_sezgisi": ("Yardimci Rj kare yukselirse SH neden yukselebilir?", "Xj'nin bagimsiz degiskenligi azalir; ayri katki daha zor ayrisir."),
        "orneklem_buyuklugu": ("Benzer DGP altinda n artarsa SH icin beklenen yon nedir?", "Daha cok bilgiyle SH genellikle duser."),
        "negatif_hipotez": ("Negatif yonlu testte hipotezler nasil yazilir?", f"H0: beta >= {_number(test.null_value)}; H1: beta < {_number(test.null_value)}."),
        "karar_dili": ("H0 reddedilemediginde dogru karar dili nedir?", "H0 reddedilemez denir; H0 kabul edildi veya kanitlandi denmez."),
        "negatif_t": ("Negatif t istatistigi neyi gosterir?", "Tahminin null degerinden negatif yonde standart hata cinsinden uzakta oldugunu gosterir."),
        "p_alpha_karari": ("p-degeri ile alpha kullanilarak karar nasil verilir?", "Kati kural p<alpha ise H0 reddedilir; p=alpha oldugunda reddedilemez."),
        "p_032": ("p=0.032 icin %10, %5 ve %1 kararlarini verin.", "%10 ve %5'te H0 reddedilir; %1'de reddedilemez."),
        "ga_kurma": ("Iki tarafli guven araliginin formulu nedir?", "beta sapka arti/eksi t kritik deger carpim standart hata."),
        "sh_yari": ("SH yariya duserse hata payi ne olur?", "Kritik deger sabitken hata payi da yariya duser."),
        "ga_null_uyeligi": ("Null deger GA icindeyse iki tarafli test karari nedir?", "Ayni alpha duzeyinde H0 reddedilemez."),
        "wage_egitim_null": ("WAGE1 egitimde 0, 0.50, 0.75 null kararlarini karsilastirin.", "0 reddedilir; 0.50 reddedilemez; 0.75 reddedilir."),
        "wage_deneyim": ("WAGE1 deneyimde iki ve tek tarafli p neden farklidir?", "Iki tarafli test iki kuyrugu, onceden belirlenmis tek tarafli test tek kuyrugu kullanir."),
        "ci_sutunlari": ("Statsmodels [0.025, 0.975] sutunlari nedir?", "%95 iki tarafli guven araliginin alt ve ust sinirlaridir."),
        "makale_parantez": ("Makale tablosundaki parantezler nasil okunmalidir?", "Tablo notu kontrol edilmeden standart hata oldugu varsayilmaz."),
        "wage_lwage": ("wage ve lwage katsayilari neden ayni birimde degildir?", "Bagimli degisken duzey ile dogal log olarak farkli olculmustur."),
        "hprice_olcek": ("HPRICE1'de sqrft ve price hangi ham birimlerdedir?", "sqrft kare fit, price bin ABD dolari; Konu 05 donusumleri burada kullanilmaz."),
        "bdrms_belirsizlik": ("Yatak odasi GA'si sifiri kapsarsa yorum nedir?", "Ayri katki dar bicimde belirlenememistir; etki yoktur denmez."),
        "buyuk_orneklem": ("Buyuk orneklemde kucuk katsayi neden anlamli olabilir?", "n standard hatayi kucultebilir; bu iktisadi buyuklugu artirmaz."),
        "kucuk_orneklem": ("Kucuk orneklemde buyuk nokta tahmini neden belirsiz kalabilir?", "SH buyuk ve GA genis olabilir."),
        "kapsama_yorumu": ("%95 kapsama nasil dogru yorumlanir?", "Bu, yontemin uzun donem kapsama oranidir; sabit araligin olasiligi degildir."),
        "kapsama_kacirma": ("Bazi %95 araliklarin gercek degeri kacirmasi basarisizlik midir?", "Hayir; %95 tek tek tum araliklarin kapsamasini degil uzun donem oranini anlatir."),
        "yanlis_p": ("p=0.03 H0'in dogru olma olasiligi midir?", "Hayir; H0 dogru kabul edildiginde ucluk olasiligidir."),
        "yanlis_h0": ("H0 reddedilemedi, dolayisiyla kanitlandi mi?", "Hayir; veri H0 aleyhine yeterli kanit vermemistir."),
        "tek_katsayi_siniri": ("Tek katsayi t testi hangi soruyu cevaplayamaz?", "Birden fazla katsayinin birlikte sifir olup olmadigini; Konu 08'de F testi ele alir."),
    }
    if kind in extra_questions:
        prompt, answer = extra_questions[kind]
        return GeneratedQuestion(kind, prompt, answer, question_index)
    if kind == "nokta_ve_belirsizlik":
        prompt = f"{context.coefficient_label} için {beta} nokta tahmini tek başına neyi söylemez?"
        answer = "Tahminin anakütledeki belirsizliğini söylemez; bunun için standart hata, test veya güven aralığı gerekir."
    elif kind == "ayni_katsayi_farkli_sh":
        prompt = "Aynı 0,60 katsayısında 0,05 yerine 0,40 standart hata görülürse ilk yorum nasıl değişir?"
        answer = "Nokta tahmini değişmez; ikinci tahmin çok daha belirsizdir ve güven aralığı daha geniş olur."
    elif kind == "ornekleme_dagilimi":
        prompt = "Örnekleme dağılımı nedir?"
        answer = "Aynı veri üretim sürecinden tekrar tekrar örneklem alındığında tahmin edicinin aldığı değerlerin dağılımıdır."
    elif kind == "standart_hata":
        prompt = "Katsayının standart hatası hangi değişkenliği tahmin etmeye çalışır?"
        answer = "Katsayı tahmininin tekrarlı örneklemlerdeki değişkenliğini tek örneklemden tahmin etmeye çalışır."
    elif kind == "uc_degiskinlik":
        prompt = "Regresyon tablosundaki 'std err' Y'nin standart sapması mı, artıkların standart sapması mı, katsayının standart hatası mı?"
        answer = "Katsayı tahmininin standart hatasıdır; Y'nin toplam yayılımı veya artıkların gözlem düzeyindeki yayılımı değildir."
    elif kind == "sh_sezgisi":
        prompt = "Diğer koşullar benzerken açıklayıcı değişkenin örneklemdeki değişkenliği artarsa standart hata için beklenen yön nedir?"
        answer = "Standart hata genellikle azalır; katsayının eğimi daha fazla bağımsız X değişkenliğiyle daha kesin ayrıştırılabilir."
    elif kind == "null_sifir_degil":
        prompt = f"H0: β={_number(test.null_value)} hipotezindeki null değer neden sıfır olmak zorunda değildir?"
        answer = "Araştırma sorusu teorik ya da pratik olarak sıfır dışındaki bir referans değeri sınayabilir."
    elif kind == "iki_tarafli_hipotez":
        prompt = f"{context.coefficient_label} için iki taraflı hipotezi yazın."
        answer = f"H0: β = {_number(test.null_value)}; H1: β ≠ {_number(test.null_value)}."
    elif kind == "yon_onceden":
        prompt = "Tek taraflı testin yönü ne zaman belirlenmelidir?"
        answer = "Katsayının işareti veya p-değeri görüldükten sonra değil, araştırma sorusu ya da teori temelinde önceden belirlenmelidir."
    elif kind == "alpha_tip1":
        prompt = f"α={test.alpha:.2f} neyi sınırlar?"
        answer = "Sıfır hipotezi doğruyken onu reddetme, yani I. tür hata riskini sınırlar."
    elif kind == "tip2_hata":
        prompt = "II. tür hata nedir?"
        answer = "Sıfır hipotezi yanlışken onu reddedememektir; test gücü bunun karşılığı olan reddetme olasılığıdır."
    elif kind == "test_istatistigi":
        prompt = f"Bu modelde t = ({beta} − {_number(test.null_value)}) / {se} kaçtır?"
        answer = f"t = {_number(test.t_statistic)}. Tahmin null değerinden bu kadar standart hata uzaktadır."
    elif kind == "df_resid":
        prompt = f"n={result.nobs} ve k={result.n_explanatory} iken artık serbestlik derecesi nedir?"
        answer = f"n−k−1 = {result.nobs}−{result.n_explanatory}−1 = {result.df_resid}."
    elif kind == "kritik_deger":
        prompt = f"Kritik değer {_number(test.critical_value)} ve t={_number(test.t_statistic)} iken karar yalnız t büyüklüğüyle nasıl okunur?"
        answer = "İki taraflı testte |t| kritik değeri aşarsa H0 reddedilir; tek taraflı testte önceden seçilen kuyruğun yönü de gerekir."
    elif kind == "p_degeri_anlami":
        prompt = f"p={format_p_value(test.p_value)} ne değildir?"
        answer = "H0'ın doğru olma olasılığı, bulgunun tesadüf olma olasılığı veya etkinin büyüklüğü değildir."
    elif kind == "p_sifir_degildir":
        prompt = "Çıktıda p=0.000 görünmesi neden doğru bir öğrenci gösterimi değildir?"
        answer = "Yuvarlama çok küçük pozitif bir p-değerini sıfır gibi gösterebilir; görünüm '< 0.001' olmalıdır."
    elif kind == "guven_araligi":
        low, high = test.confidence_lower, test.confidence_upper
        prompt = f"{context.coefficient_label} için iki taraflı güven aralığı neyi özetler?"
        answer = f"Seçilen güven düzeyinde null ile uyumlu değerler kümesini özetler; bu bağlamda [{_number(float(low))}; {_number(float(high))}] hesaplanır."
    elif kind == "guven_duzeyi":
        prompt = "Güven düzeyi %95'ten %99'a çıktığında, diğer her şey aynıyken aralık ne olur?"
        answer = "Kritik değer yükseldiği için aralık genişler. Parametre sabittir; örneklemden örnekleme değişen aralıklardır."
    elif kind == "test_ga":
        prompt = "İki taraflı testte null değer güven aralığının dışındaysa karar nedir?"
        answer = "Aynı anlamlılık düzeyinde H0 reddedilir. Bu eşdeğerlik tek taraflı testler için iddia edilmez."
    elif kind == "model_sayisi":
        prompt = "Bu konunun tek katsayı testi sınırı hangi konuyu dışarıda bırakır?"
        answer = "Birden fazla katsayının birlikte sınanmasını ve F testini; bunlar Konu 08'in kapsamıdır."
    elif kind == "deneyim_kuyruk":
        prompt = "Pozitif t istatistiğinde 'greater' p-değeri ile iki taraflı p-değeri arasındaki ilişki ne zaman yarı yarıya olur?"
        answer = "Simetrik t dağılımında gözlenen işaret önceden seçilen alternatif yönüyle uyumluysa; bu ilişki koşulsuz bir kural değildir."
    elif kind == "python_cikti":
        prompt = "Statsmodels çıktısındaki `coef`, `std err`, `t` ve `P>|t|` alanlarını sırasıyla nasıl okursunuz?"
        answer = "Katsayı tahmini, katsayının standart hatası, sıfır nulluna göre t istatistiği ve iki taraflı p-değeridir."
    elif kind == "nonrobust":
        prompt = "`Covariance Type: nonrobust` bu modülde neyi bildirir?"
        answer = "Geleneksel homoskedastisiteye dayanan EKK standart hatalarının kullanıldığını bildirir; dayanıklı standart hatalar Konu 12'de ele alınır."
    elif kind == "yildiz":
        prompt = "Regresyon tablosundaki yıldızlar hangi bilgiye dayanır ve neyi göstermez?"
        answer = "Tablo notundaki iki taraflı p eşiğine dayanır; etki büyüklüğünü veya nedenselliği göstermez."
    elif kind == "iktisadi_onem":
        prompt = "İstatistiksel anlamlılık ile iktisadi önem neden ayrı değerlendirilir?"
        answer = "Küçük p, tahmin edilen farkın araştırma bağlamında büyük veya önemli olduğunu söylemez; birim ve anlamlı değişim miktarı ayrıca incelenir."
    elif kind == "olcekleme":
        prompt = f"{context.coefficient_label} katsayısını dört {context.explanatory_unit} değişim için nasıl ölçeklersiniz?"
        answer = f"Nokta tahmini ve güven aralığının iki sınırı 4 ile çarpılır; negatif değişimde aralık uçları yeniden sıralanır."
    elif kind == "nedensellik":
        prompt = "Küçük p-değeri gözlemsel WAGE1/HPRICE1 modelinde nedenselliği kanıtlar mı?"
        answer = "Hayır. Sonuç, modeldeki ceteris paribus ilişkiyi özetler; nedensel yorum araştırma tasarımı ve ek varsayımlar gerektirir."
    else:
        prompt = "Tek katsayı sonucunu bütünleşik raporda hangi bileşenlerle sunarsınız?"
        answer = "Model ve araştırma sorusu, katsayı ve birim, standart hata, test kararı, güven aralığı, iktisadi büyüklük ve nedensellik sınırı birlikte yazılır."
    return GeneratedQuestion(kind, prompt, answer, question_index)
