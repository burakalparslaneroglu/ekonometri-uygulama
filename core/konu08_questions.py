"""Konu 08 için deterministik, kapsam-sınırlı öğretim soruları."""

from __future__ import annotations

from dataclasses import dataclass

from core.joint_inference_utils import NestedModelFResult
from core.question_engine import GeneratedQuestion, cycle_question_type


KONU08_QUESTION_TYPES = (
    "tekli_ortak", "q", "alternatif", "nested", "ayni_orneklem", "ssr_f", "r2_f", "df", "ust_kuyruk",
    "ortak_ret", "ozel_genel", "t_kare", "buyuk_n", "buyuk_n_sinir", "nedensellik", "nonrobust",
    "esitlik_kisiti", "toplam_kisiti", "sifir_disinda", "ssr_sirasi", "formul_uyumu", "alpha", "kritik_deger",
    "wage_n", "wage_exper_p", "hprice_q", "hprice_bdrms", "sabit_kisiti", "statsmodels_f", "t_tek_taraf",
    "tutarlilik", "asimptotik", "normal_olmayan", "buyuk_n_omitted", "buyuk_n_secim", "buyuk_n_bicim",
    "buyuk_n_bagimlilik", "buyuk_n_hetero", "raporlama", "ekonomik_onem", "p_anlami", "f_isaret",
    "robust_sinir", "makale_okuma", "complete_case",
    "technical_validity", "zero_row", "dependent_rank", "inconsistent_rhs", "rows_vs_q", "sum_vs_two_zero",
    "equality_matrix", "nonzero_null", "nested_when", "matrix_only", "matrix_q", "denominator_df", "custom_tail",
    "custom_rejection", "custom_classification",
)


@dataclass(frozen=True)
class Konu08QuestionContext:
    """Gerçek ortak F sonucu ile soru üretimi için bağlam."""

    context_id: str
    test: NestedModelFResult


def generate_konu08_question(context: Konu08QuestionContext, question_index: int) -> GeneratedQuestion:
    """Aynı bağlam ve indeks için aynı Konu 08 sorusunu üretir."""
    kind = cycle_question_type(context.context_id, question_index, KONU08_QUESTION_TYPES, namespace="konu08")
    test = context.test
    items = {
        "tekli_ortak": ("İki ayrı t testi neden tek bir ortak hipotez testi değildir?", "Çünkü ortak test katsayıların birlikte değişkenliğini ve tek ortak araştırma sorusunu aynı F istatistiğinde değerlendirir."),
        "q": ("Bu ortak testte kısıt sayısı q kaçtır?", f"q={test.q}; kısıt sayısı kısıtsız ve kısıtlı eğim sayıları farkıdır."),
        "alternatif": ("Ortak F testinin alternatif hipotezi nasıl yazılır?", "Kısıtlardan en az biri doğru değildir. Bu, bütün katsayıların ayrı ayrı anlamlı olduğu anlamına gelmez."),
        "nested": ("İç içe model koşulu nedir?", "Kısıtlı model, kısıtsız modelden bazı eğimleri sıfıra eşitleyerek elde edilmelidir."),
        "ayni_orneklem": ("Kısıtlı ve kısıtsız model neden aynı complete-case örneklemini kullanmalıdır?", "SSR ve R² farkları yalnız kısıtların etkisini yansıtabilsin diye; örneklem farkı karşılaştırmayı bozar."),
        "ssr_f": ("SSR ile F hesabında temel karşılaştırma nedir?", f"(SSR_R−SSR_UR)/q, kısıtsız modelin SSR_UR/{test.df_denom} artık varyans tahminine bölünür."),
        "r2_f": ("R² ile F hesabı neyi kullanır?", "Kısıtlar kaldırıldığında R²'deki artışı, kısıtsız modelin açıklanmayan payına göre ölçekler."),
        "df": ("F testinde pay ve payda serbestlik dereceleri nedir?", f"Pay sd=q={test.q}; payda sd kısıtsız modelin artık sd'sidir: {test.df_denom}."),
        "ust_kuyruk": ("F testi neden üst kuyruk testidir?", "Kısıtlar veriye kötü uyarsa SSR artar ve F büyür; kanıt büyük, negatif olmayan F değerlerinde aranır."),
        "ortak_ret": ("Ortak H0 reddedilirse güvenli yorum nedir?", "Kısıtlardan en az biri veriyle uyumsuzdur. Bu, tüm katsayıların tek tek anlamlı olduğunu veya modelin nedensel olduğunu göstermez."),
        "ozel_genel": ("Özel ortak test ile genel anlamlılık testi arasındaki fark nedir?", "Özel test seçili kısıtları, genel test ise sabit dışındaki tüm eğimlerin birlikte sıfır olmasını sınar."),
        "t_kare": ("F=t² eşitliği hangi koşulda geçerlidir?", "Aynı null değerine ait iki taraflı t testi ve tek bir doğrusal kısıt (q=1) olduğunda."),
        "buyuk_n": ("Tutarlılık ile yansızlık aynı kavram mıdır?", "Hayır. Tutarlılık tahminin n büyüdükçe gerçek parametreye yaklaşmasıdır; yansızlık sonlu örneklem beklentisiyle ilgilidir."),
        "buyuk_n_sinir": ("Büyük örneklem eksik değişken yanlılığını otomatik çözer mi?", "Hayır. Daha büyük n örnekleme belirsizliğini azaltabilir; yanlış merkezi veya araştırma tasarımını düzeltmez."),
        "nedensellik": ("Küçük ortak p-değeri nedensellik kanıtı mıdır?", "Hayır. Gözlemsel modelde bu, seçilen kısıtların koşullu ilişkiyle uyumunu sınar; nedensel tasarım ayrıca gerekir."),
        "nonrobust": ("Bu konudaki F testinin kapsam sınırı nedir?", "Geleneksel homoskedastisiteye dayanan nonrobust F testidir; dayanıklı ortak testler Konu 12'ye bırakılır."),
    }
    items.update({
        "esitlik_kisiti": ("β_exper=β_tenure hipotezi nasıl doğrusal yazılır?", "β_exper−β_tenure=0 olarak yazılır ve q=1'dir."),
        "toplam_kisiti": ("β_exper+β_tenure=1 hipotezi F testiyle sınanabilir mi?", "Evet. Sıfır dışındaki sağ taraf genel doğrusal kısıta engel değildir; q=1'dir."),
        "sifir_disinda": ("Kısıtlı model her zaman değişken silerek mi kurulur?", "Hayır. Katsayı eşitliği veya sıfır dışı kısıtlar genel Rβ=r testi gerektirir."),
        "ssr_sirasi": ("Kısıtsız modelde SSR neden daha büyük olamaz?", "Kısıtsız model kısıtlı modelin parametre uzayını içerir; EKK daha geniş kümede en küçük SSR'yi seçer."),
        "formul_uyumu": ("SSR ve R² F formülleri ne zaman eşdeğerdir?", "Aynı bağımlı değişken, aynı örneklem, sabitli iç içe modeller ve aynı kısıtlar altında."),
        "alpha": ("p=α olduğunda katı karar kuralı nedir?", "Kural p<α olduğundan H0 reddedilemez."),
        "kritik_deger": ("F kritik değerinin sağında olmak neyi bildirir?", "Gözlenen F üst kuyruk red bölgesindedir ve H0 reddedilir."),
        "wage_n": ("WAGE1 ortak testinde gözlem sayısı kaçtır?", f"Her iki model aynı complete-case örneklemde n={test.unrestricted_result.nobs} gözlem kullanır."),
        "wage_exper_p": ("WAGE1'de deneyimin ayrı p'sinin yaklaşık 0.064 olması ortak reddi çürütür mü?", "Hayır. Ayrı ve ortak hipotezler farklıdır; ortak test iki kısıtı birlikte sınar."),
        "hprice_q": ("HPRICE1 örneğinde arsa ve yatak odası dışlanırsa q kaçtır?", "İki bağımsız dışlama kısıtı olduğu için q=2'dir."),
        "hprice_bdrms": ("Yatak odasının ayrı p'si büyükken HPRICE1 ortak testi neden reddedilebilir?", "Arsa ve yatak odasının birlikte dışlanması farklı bir hipotezdir; arsa katkısı ortak sonucu taşıyabilir."),
        "sabit_kisiti": ("Genel F testinde sabit terim neden kısıtlanmaz?", "Genel test, modeldeki eğimlerin birlikte sıfır olup olmadığını sınar; kısıtlı model sabit içerir."),
        "statsmodels_f": ("Statsmodels özetindeki F-statistic her zaman seçili özel ortak test midir?", "Hayır. Ana özet sabit dışındaki tüm eğimlerin sıfır olduğu genel F testini raporlar."),
        "t_tek_taraf": ("Tek taraflı t testiyle F=t² eşdeğerliği iddia edilir mi?", "Hayır. Eşdeğerlik aynı nulla ait iki taraflı t testi ve q=1 içindir."),
        "tutarlilik": ("Tutarlılık sonlu örneklemde yansızlık mı demektir?", "Hayır. Tutarlılık n büyüdükçe yaklaşmayı, yansızlık beklenen değeri anlatır."),
        "asimptotik": ("Asimptotik normallik neye temel sağlar?", "Uygun standartlaştırılmış tahminlerin büyük n'de yaklaşık normal davranmasına ve yaklaşık çıkarıma temel sağlar."),
        "normal_olmayan": ("Normal olmayan hata büyük örneklem çıkarımını otomatik imkânsız kılar mı?", "Hayır; uygun koşullarda yaklaşık çıkarım mümkün olabilir, fakat kalite DGP'ye bağlıdır."),
        "buyuk_n_omitted": ("Büyük n eksik yeteneği gözlemlemeyi sağlar mı?", "Hayır. Eksik değişken yanlılığı daha kesin ama yanlış merkeze yakın tahminler üretebilir."),
        "buyuk_n_secim": ("Gönüllü çevrim içi örneklemde büyük n neyi düzeltmez?", "Seçilim mekanizmasını ve hedef anakütleye genellenebilirlik sorununu düzeltmez."),
        "buyuk_n_bicim": ("Büyük n yanlış fonksiyonel biçimi otomatik düzeltir mi?", "Hayır. Yanlış biçim sürer; büyük örneklem yalnız sorunu daha görünür yapabilir."),
        "buyuk_n_bagimlilik": ("Aynı firmadan tekrarlanan gözlemler için daha çok satır yeterli midir?", "Hayır. Gözlemler arası bağımlılık uygun yöntem ve standart hata yaklaşımı gerektirir."),
        "buyuk_n_hetero": ("Büyük n heteroskedastisite altında nonrobust F'yi garanti eder mi?", "Hayır. Dayanıklı çıkarım seçimi Konu 12 kapsamındadır."),
        "raporlama": ("Ortak F sonucu nasıl kısa raporlanır?", f"Hipotez, F({test.q},{test.df_denom})={test.f_from_ssr:.2f}, p-değeri, karar ve nedensellik sınırı birlikte yazılır."),
        "ekonomik_onem": ("Küçük ortak p iktisadi önem gösterir mi?", "Hayır. Etki büyüklüğü, birim ve ekonomik bağlam ayrıca değerlendirilmelidir."),
        "p_anlami": ("F testindeki p-değeri H0'ın doğru olasılığı mıdır?", "Hayır. H0 ve model varsayımları altında en az gözlenen kadar büyük F olasılığıdır."),
        "f_isaret": ("F istatistiği neden negatif gösterilmez?", "Kısıtların kuadratik uyumsuzluk ölçüsüdür; teorik olarak negatif olamaz."),
        "robust_sinir": ("Bu ekranda neden robust F seçeneği yoktur?", "Pedagojik sıra gereği dayanıklı ortak testler Konu 12'ye bırakılmıştır."),
        "makale_okuma": ("Makale tablosunda F satırını okumadan önce ne kontrol edilir?", "Hangi sıfır hipotezini ve hangi standart hata/kovaryans türünü kullandığı tablo notundan kontrol edilir."),
        "complete_case": ("Eksik gözlemler F karşılaştırmasında nasıl ele alınmalıdır?", "Tüm gerekli değişkenlerin birleşimi üzerinden tek complete-case örneklem hazırlanmalıdır."),
        "technical_validity": ("Teknik olarak geçerli custom kısıt sistemi için temel koşul nedir?", "Satırlar bilgi taşımalı, sistem tutarlı olmalı ve R'nin satırları bağımsız olmalıdır."),
        "zero_row": ("0·β0+⋯+0·βk=0 satırı neden geçerli test kısıtı değildir?", "Hiç bilgi eklemez; q'yu artırmaz ve kullanıcıdan bu satırı düzeltmesi istenir."),
        "dependent_rank": ("βexper=0 ve 2βexper=0 neden iki kısıt değildir?", "İkinci satır ilkinin katıdır; rank(R)=1 ve yeni bağımsız bilgi yoktur."),
        "inconsistent_rhs": ("Aynı sol taraf için βexper=0 ve βexper=1 neden çalıştırılmaz?", "rank(R)<rank([R|r]) olur; sistem aynı anda sağlanamaz ve tutarsızdır."),
        "rows_vs_q": ("Girilen satır sayısı ile q arasındaki fark nedir?", "Satır sayısı kullanıcının girdiği eşitlik sayısıdır; q=rank(R) yalnız bağımsız eşitlikleri sayar."),
        "sum_vs_two_zero": ("βexper+βtenure=0, iki ayrı sıfır kısıtıyla aynı mıdır?", "Hayır. Toplam kısıtı tek doğrusal birleşimi, iki sıfır kısıtı iki ayrı parametreyi sınar."),
        "equality_matrix": ("βexper=βtenure kısıtı Rβ=r biçiminde nasıldır?", "R satırı deneyim için 1, kıdem için −1; r=0 olacak şekilde yazılır."),
        "nonzero_null": ("βeduc=0.50 kısıtında değişken silinir mi?", "Hayır. Bu sıfır dışı null kısıtıdır; genel matrix F kullanılır."),
        "nested_when": ("SSR/R² nested karşılaştırması ne zaman ayrıca uygundur?", "Her kısıt ayrı bir eğimi sıfıra eşitliyor, restricted model değişken silerek kuruluyor ve örneklem aynı kalıyorsa."),
        "matrix_only": ("Katsayı eşitliğinde neden yalnız matrix F kullanılır?", "Bir değişkeni silmek βexper=βtenure eşitliğini uygulamaz; kısıt parametreleri birlikte bağlar."),
        "matrix_q": ("Genel matrix F formülündeki q nedir?", "R matrisinin rank'ı, yani bağımsız kısıt sayısıdır."),
        "denominator_df": ("Custom F testinin payda serbestlik derecesi nedir?", "Kısıtsız tahmin sonucunun artık serbestlik derecesidir."),
        "custom_tail": ("Custom F'de büyük istatistik neden üst kuyrukta kanıttır?", "Kısıtların belirsizliğe göre büyük uyumsuzluğunu ölçer; F negatif olmaz."),
        "custom_rejection": ("Geçerli custom sistemde H0 reddedilirse güvenli sonuç nedir?", "Girilen kısıtların tümü birlikte veriyle uyumsuzdur; en az bir kısıt reddedilir."),
        "custom_classification": ("Custom kısıt sınıflandırması ne için kullanılır?", "Uygun test yöntemini öğretmek için: dışlama mı, eşitlik mi, sıfır dışı mı yoksa genel doğrusal kısıt mı olduğunu ayırır."),
    })
    prompt, answer = items[kind]
    return GeneratedQuestion(kind, prompt, answer, question_index)
