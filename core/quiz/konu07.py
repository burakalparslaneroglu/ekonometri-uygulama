"""Konu 7 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 7.1–7.13 ve Mini Quiz
7.1–7.13 maddelerini tekrar etmez: A–D modellerinin katsayı ve standart hata tablosu; standart hatayı belirleyen beş
değişim ve Tablo 7.1'in karşılaştırılması; eğitim, reklam, esneklik ve iş eğitimi hipotezlerinin kurulması; 0,40/0,10,
−0,30/0,20, 1,20/0,60 ve 0,08/0,02 için t hesapları ile WAGE1'de H₀: β = 0,70; p = 0,240, 0,048, 0,009 ve 0,050
kararları; 0,50/0,10 ve −0,20/0,15 için güven aralıkları; Tablo 7.4 ile 0,60 ve 0,80 testleri; verimlilik, vergi, ilaç
ve p = 0,08 sonrası yön seçimi; Python çıktısı ve Tablo 7.5'in okuma soruları; HPRICE1'de 100 birimlik büyüklük hesabı;
altı yanlış cümlenin düzeltilmesi; §7.13'teki konut tablosunun yedi maddesi ve mini quizlerdeki sorular. Aynı becerileri
yeni bağlamlarla ve yeni sayılarla sınar. Standart hatalar klasik EKK standart hatalarıdır; birden fazla kısıtın
birlikte sınanması (F testi) Konu 8'in, heteroskedastisiteye dayanıklı çıkarım Konu 12'nin, logaritmik modellerde tam
yüzde dönüşümü Konu 9'un konusudur; sorular bunları kullanmaz.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol, beta_hat_aliases
from core.quiz.model import (
    Equation,
    FillBlanks,
    MultipleChoice,
    NumberBlank,
    Question,
    QuestionSet,
    TrueFalse,
)


def _note(section: str, *objects: str) -> NoteRef:
    return NoteRef(section, 0, tuple(objects))


_BETA_HAT = ("\\widehat\\beta", "\\hat\\beta", "β̂", "betahat", "bhat", *beta_hat_aliases(1), "\\beta", "β")
_SE_OF = tuple(f"{name}({hat})" for name in ("se", "SE", "SH", "sh") for hat in (*_BETA_HAT, "b"))
"""se(β̂) yazımları (\\operatorname{se}(\\hat\\beta_1) dahil; metin komutları okunmadan önce silinir): ``s`` sembolüne
çevrilir. Uzun yazımlar önce eşleştiği için ``se`` ve β̂ yazımlarıyla karışmaz."""
_SE = (*_SE_OF, "se", "SE", "SH", "sh")
_DELTA_X = ("\\Delta x", "\\Deltax", "\\Delta X", "\\DeltaX", "Δ x", "Δx", "ΔX", "Delta x", "deltax", "dx")
_P_TWO = ("p_iki", "p_2", "p₂", "p2")


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="nokta-tahmini-estimand-estimator-estimate", note=_note("7.1"),
        prompt=(
            "Bir araştırmacı 800 firmalık bir örneklemde satışların reklam harcaması ve fiyat üzerine regresyonunu EKK "
            "ile tahmin ediyor ve reklam katsayısını $\\widehat\\beta_{\\text{reklam}} = 0{,}35$ buluyor. Bu 0,35 "
            "aşağıdakilerden hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Anakütledeki bilinmeyen parametre $\\beta_{\\text{reklam}}$ (tahmin edilmek istenen büyüklük)",
                "Bu örneklemden hesaplanan nokta tahmini: tahmin edicinin bu veride aldığı değer",
                "Tahmin edici: her örneklemde katsayıyı hesaplayan EKK formülü",
                "Tahminin örnekleme belirsizliğini ölçen standart hata",
            ),
            correct=1,
        ),
        explanation=(
            "Üç kavram ayrılır: anakütle parametresi $\\beta_{\\text{reklam}}$ bilinmeyen hedeftir (estimand); EKK "
            "formülü tahmin edicidir (estimator); 0,35 ise bu formülün bu örneklemde aldığı değerdir, yani nokta "
            "tahmini (estimate). Başka bir örneklemde tahmin değişir, parametre değişmez; tahminin belirsizliğini "
            "standart hata ölçer. İstatistiksel çıkarım 0,35'ten ve belirsizliğinden yola çıkarak parametre hakkında "
            "sonuç üretir (§7.1)."
        ),
    ),
    Question(
        key="k02", concept="standart-hata-diger-yayilim-olculerinden-farkli", note=_note("7.2", "(7.2)"),
        prompt=(
            "WAGE1 ücret modelinde (ücret ~ eğitim + deneyim + kıdem) üç sayı hesaplanıyor: saatlik ücretin örneklem "
            "standart sapması 3,693; artıkların standart sapması $\\widehat\\sigma = 3{,}084$; eğitim katsayısının "
            "standart hatası 0,051. “Aynı büyüklükte başka bir örneklem seçilseydi eğitim katsayısının tahmini tipik "
            "olarak ne kadar değişirdi?” sorusunu hangisi cevaplar?"
        ),
        answer=MultipleChoice(
            (
                "3,693: saatlik ücretin örneklemdeki toplam değişkenliği",
                "3,084: modelin gözlem düzeyindeki tipik hatası",
                "0,051: eğitim katsayısı tahmininin örnekleme belirsizliği",
                "0,599: eğitim katsayısının nokta tahmini",
            ),
            correct=2,
        ),
        explanation=(
            "Bağımlı değişkenin standart sapması Y'nin toplam değişkenliğini, $\\widehat\\sigma$ modelin gözlem "
            "düzeyindeki tipik hatasını, standart hata ise katsayı tahmininin örnekleme belirsizliğini ölçer. Denklem "
            "7.2'ye göre standart hata $\\widehat\\sigma$'yı kullanır ama onu eğitimin bağımsız değişkenliğine böler; "
            "bu yüzden üç sayı çok farklıdır ve yalnız 0,051 örneklemden örnekleme değişimi özetler (§7.2)."
        ),
    ),
    Question(
        key="k03", concept="kritik-deger-artik-serbestlik-derecesinden", note=_note("7.4", "Tablo 7.3"),
        prompt=(
            "25 gözlemli bir örneklemde sabit terimli ve dört açıklayıcı değişkenli bir model tahmin ediliyor. Tek bir "
            "katsayı için yüzde 5 iki taraflı $t$ testinde hangi kritik değer kullanılır?"
        ),
        answer=MultipleChoice(
            (
                "2,086 (serbestlik derecesi 20)",
                "2,060 (serbestlik derecesi 25)",
                "2,069 (serbestlik derecesi 23)",
                "1,960 (standart normal dağılım)",
            ),
            correct=0,
        ),
        explanation=(
            "Sabit terimli çoklu regresyonda serbestlik derecesi $n - k - 1 = 25 - 4 - 1 = 20$; Tablo 7.3'e göre "
            "$t_{0{,}025;\\,20} = 2{,}086$. 25 gözlem sayısıdır, serbestlik derecesi değildir; 23, tek açıklayıcı "
            "değişkenli modelin serbestlik derecesi olurdu. 1,960 yalnız çok büyük serbestlik derecesinde uygundur; "
            "küçük örneklemde onu kullanmak testi gereğinden sık reddettirir (§7.4)."
        ),
    ),
    Question(
        key="k04", concept="p-degeri-en-kucuk-anlamlilik-duzeyi", note=_note("7.5", "(7.7)"),
        prompt=(
            "Bir çalışmada bir katsayı için iki taraflı $p$-değeri 0,018 olarak raporlanmıştır. Aşağıdaki "
            "yorumlardan hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Sıfır hipotezinin doğru olma olasılığı %1,8'dir",
                "Bulgunun tesadüfen ortaya çıkmış olma olasılığı %1,8'dir",
                "Katsayının iktisadi önemi %98,2 düzeyindedir",
                "$H_0$, α = 0,02 düzeyinde reddedilir; α = 0,01 düzeyinde reddedilemez",
            ),
            correct=3,
        ),
        explanation=(
            "p-değeri, $H_0$ ve model varsayımları doğruyken gözlenen kadar uç bir t istatistiği elde etme "
            "olasılığıdır; $H_0$'ın doğru olma ya da bulgunun tesadüf olma olasılığı değildir ve iktisadi önemi "
            "ölçmez. $H_0$'ın reddedilebileceği en küçük anlamlılık düzeyi olarak okunabilir: p < α kuralıyla "
            "(Denklem 7.7) 0,018 < 0,02 olduğu için reddedilir, 0,018 > 0,01 olduğu için reddedilemez (§7.5)."
        ),
    ),
    Question(
        key="k05", concept="tek-tarafli-kritik-deger", note=_note("7.8"),
        prompt=(
            "Teori ve önceden kaydedilmiş analiz planı bir katsayının pozitif olmasını öngörüyor: "
            "$H_0: \\beta \\leq 0$, $H_1: \\beta > 0$. Büyük örneklemde $t = 1{,}80$ bulunuyor. Yüzde 5 düzeyinde karar "
            "ve gerekçesi hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Reddedilir: tek taraflı kritik değer 1,645 ve 1,80 > 1,645",
                "Reddedilemez: kritik değer 1,96 ve 1,80 < 1,96",
                "Reddedilemez: tek taraflı p = 2 × 0,036 = 0,072 > 0,05",
                "Reddedilir: iki taraflı p = 0,072 değeri 0,10'dan küçüktür",
            ),
            correct=0,
        ),
        explanation=(
            "Tek taraflı yüzde 5 testinde α = 0,05'in tamamı sağ kuyruğa konur; kritik değer standart normalin 0,95 "
            "noktasıdır: 1,645. 1,96 iki taraflı testin kritik değeridir. Tek taraflı p = P(T > 1,80) ≈ 0,036 < 0,05; "
            "iki kuyruğu sayan 0,072 iki taraflı p-değeridir ve yüzde 5'te reddetmezdi. Yön sonuç görülmeden "
            "belirlendiği için tek taraflı karar geçerlidir; 0,10 ise seçilen düzey değildir (§7.8)."
        ),
    ),
    Question(
        key="k06", concept="parantezde-t-istatistigi", note=_note("7.10"),
        prompt=(
            "Bir makale tablosunun notunda “Parantez içinde t istatistikleri verilmiştir.” yazıyor. Tabloda bir katsayı "
            "0,84, altında da (2,10) görülüyor. Bu katsayının standart hatası kaçtır?"
        ),
        answer=MultipleChoice(("2,10", "1,764", "0,40", "0,84"), correct=2),
        explanation=(
            "Parantezin içeriği tablo notundan okunur; burada t istatistiğidir. $t = \\widehat\\beta / "
            "\\operatorname{se}(\\widehat\\beta)$ olduğundan standart hata 0,84/2,10 = 0,40'tır. Parantezi standart "
            "hata sanan okur standart hatayı 2,10, t'yi 0,84/2,10 = 0,40 sanar ve katsayının sıfırdan ayrışmadığı "
            "sonucuna varır; not okunmadan tablo yorumlanamaz (§7.10)."
        ),
    ),
    Question(
        key="k07", concept="raporda-yorum-siniri", note=_note("7.13"),
        prompt=(
            "Bir öğrencinin raporu: “Reklam katsayısı 0,35 olarak tahmin edilmiştir (SH = 0,10; t = 3,50; p < 0,001; "
            "yüzde 95 GA [0,15; 0,55]). Fiyat sabitken 100.000 TL daha fazla reklam harcaması yapan firmaların "
            "satışları ortalama 0,35 milyon TL daha yüksektir.” İyi bir raporun beş bileşeninden hangisi eksiktir?"
        ),
        answer=MultipleChoice(
            (
                "Katsayının büyüklüğü ve ölçü birimiyle yorumu",
                "Yorum sınırı: gözlemsel veride nedensellik uyarısı",
                "Standart hata: tahminin örnekleme belirsizliği",
                "Güven aralığı: veriyle uyumlu parametre değerleri",
            ),
            correct=1,
        ),
        explanation=(
            "§7.13'e göre iyi bir rapor beş bileşeni birlikte taşır: katsayı büyüklüğü, standart hata, test sonucu, "
            "güven aralığı ve yorum sınırı. Paragrafta ilk dördü vardır; fakat reklam harcaması firmalar arasında "
            "rastgele atanmadığı için bulgunun nedensel etki olarak yorumlanmadığı yazılmamıştır. Notlardaki örnek "
            "paragraf da bu sınırla biter (§7.13)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="alfa-birinci-tur-hata-ust-siniri", note=_note("7.3", "Tablo 7.2"),
        prompt=(
            "Anlamlılık düzeyini α = 0,05 seçmek, sıfır hipotezi yanlışken onu reddedememe olasılığının en çok %5 "
            "olmasını sağlar."
        ),
        answer=TrueFalse(False),
        explanation=(
            "α, $H_0$ doğruyken onu reddetme olasılığının (I. tür hata) üst sınırıdır. $H_0$ yanlışken onu "
            "reddedememek II. tür hatadır (Tablo 7.2). α bu olasılığı sınırlamaz: II. tür hata olasılığı etkinin "
            "büyüklüğüne, örneklem büyüklüğüne ve veri değişkenliğine bağlıdır; α küçüldükçe de artar. Konu 7 Sezgi "
            "Deney 2'de β₁ = 0,3 ve n = 30 iken II. tür hata oranı yaklaşık üçte ikidir (§7.3)."
        ),
    ),
    Question(
        key="d02", concept="aralik-rastgele-parametre-sabit", note=_note("7.6"),
        prompt=(
            "Tekrarlı örnekleme yorumunda yüzde 95 güven aralığının sınırları sabittir; örneklemden örnekleme değişen, "
            "anakütle katsayısıdır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Tersidir: anakütle parametresi sabittir, örneklemden örnekleme değişen güven aralığıdır. “Yüzde 95”, aynı "
            "yöntemle kurulan aralıkların uzun dönemde yaklaşık %95'inin sabit parametreyi kapsayacağını anlatır. Tablo "
            "7.1'in benzetiminde 5.000 aralığın 0,9460'ı gerçek eğimi kapsar; Şekil 7.3'teki aralıkların her biri "
            "farklıdır (§7.6)."
        ),
    ),
    Question(
        key="d03", concept="guven-duzeyi-ile-test-duzeyi", note=_note("7.7", "(7.9)"),
        prompt=(
            "Bir katsayının yüzde 99 güven aralığı sıfırı kapsamıyorsa, aynı katsayı için yüzde 5 iki taraflı "
            "$H_0: \\beta = 0$ testi de reddedilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Denklem 7.9'a göre yüzde 99 aralık sıfırı kapsamıyorsa $H_0: \\beta = 0$ yüzde 1 iki taraflı düzeyde "
            "reddedilir, yani p < 0,01'dir; o hâlde p < 0,05 de sağlanır. Aynı merkezli yüzde 95 aralık yüzde 99 "
            "aralığın içinde kaldığı için o da sıfırı kapsamaz (§7.7)."
        ),
    ),
    Question(
        key="d04", concept="cikti-t-sutunu-sifir-hipotezi-icin", note=_note("7.9"),
        prompt=(
            "WAGE1 ücret modelinde (§7.9 çıktısı) araştırmacı eğitim katsayısı için $H_0: \\beta_{\\text{eğitim}} = "
            "0{,}50$ hipotezini sınamak istiyor. Statsmodels çıktısındaki `t` ve `P>|t|` sütunları bu hipotezin test "
            "istatistiğini ve p-değerini doğrudan verir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Çıktıdaki `t` ve `P>|t|` sütunları $H_0: \\beta_j = 0$ içindir: t = coef/std err. "
            "$H_0: \\beta = 0{,}50$ için t'yi araştırmacı hesaplar: (0,599 − 0,50)/0,0513 ≈ 1,93; çıktıdaki 11,679 "
            "değil. Çıktı okuma sırasının beşinci adımı budur: sınanan hipotezin gerçekten $\\beta_j = 0$ olup "
            "olmadığını doğrulamak (§7.9)."
        ),
    ),
    Question(
        key="d05", concept="anlamsiz-katsayi-genis-aralik", note=_note("7.11"),
        prompt=(
            "Katılımcıların rastgele atandığı bir değerlendirmede bir iş eğitimi programının aylık kazanca etkisi "
            "1.750 TL tahmin edilmiş, yüzde 95 güven aralığı [−250; 3.750] TL bulunmuştur. Katsayı yüzde 5 düzeyinde "
            "anlamlı değildir; buna rağmen veri, iktisadi olarak büyük pozitif etkilerle de uyumludur."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Aralık sıfırı kapsadığı için $H_0: \\beta = 0$ yüzde 5 iki taraflı düzeyde reddedilemez (Denklem 7.9). "
            "Ama aralık 3.750 TL gibi büyük pozitif etkileri de içerir. Doğru sonuç “etki yok” değil, “tahmin "
            "belirsiz” biçimindedir; HPRICE1'deki yatak odası katsayısı da (p = 0,128, aralık [−4,07; 31,77]) aynı "
            "durumdadır (§7.11)."
        ),
    ),
    Question(
        key="d06", concept="cok-sayida-test-tesadufi-anlamlilik", note=_note("7.12"),
        prompt=(
            "Ücretle gerçekte hiçbir ilişkisi olmayan 20 değişken, birbirinden bağımsız 20 ayrı yüzde 5 testiyle "
            "sınanırsa en az birinin “anlamlı” çıkma olasılığı %5'tir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Her test, $H_0$ doğruyken %5 olasılıkla reddeder. 20 bağımsız testte hiçbirinin reddetmeme olasılığı "
            "0,95²⁰ ≈ 0,36; en az birinin reddetme olasılığı ≈ 0,64'tür. Çok sayıda katsayı arasından yalnız anlamlı "
            "olanı seçmek tesadüfi anlamlılık olasılığını artırır; birden fazla kısıtın birlikte sınanması sonraki "
            "bölümün konusudur (§7.12)."
        ),
    ),
    Question(
        key="d07", concept="test-gucu-orneklemle-artar", note=_note("7.3"),
        prompt=(
            "Diğer koşullar aynıyken örneklem büyüdükçe, yanlış bir sıfır hipotezini reddetme olasılığı (testin gücü) "
            "genellikle artar."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Test gücü, yanlış bir $H_0$'ı reddetme olasılığıdır. Örneklem büyüdükçe standart hata küçülür; aynı gerçek "
            "etki daha büyük |t| değerleri üretir ve reddetme sıklaşır. Notlara göre örneklem büyüklüğü, etkinin "
            "büyüklüğü ve veri değişkenliği gücü etkiler. Konu 7 Sezgi Deney 2'de β₁ = 0,3 iken kuramsal güç n = 30'da "
            "yaklaşık %34, n = 100'de yaklaşık %83'tür (§7.3)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="t-sinanan-degere-baglidir", note=_note("7.4", "(7.5)"),
        prompt=(
            "Bir log–log talep modelinde gelir esnekliği $\\widehat\\beta = 0{,}84$, standart hatası 0,06 tahmin "
            "edilmiştir. $H_0: \\beta = 1$ için $t$ istatistiği **(1)**, $H_0: \\beta = 0$ için $t$ istatistiği **(2)** "
            "olur. (Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(-2.67, 0.005, "−2,67"), NumberBlank(14.0, 0.005, "14,00"))),
        explanation=(
            "Denklem 7.5: $t = (\\widehat\\beta - a)/\\operatorname{se}(\\widehat\\beta)$. (0,84 − 1)/0,06 ≈ −2,67 ve "
            "(0,84 − 0)/0,06 = 14,00. Büyük örneklemde iki hipotez de yüzde 5 iki taraflı testte reddedilir "
            "(|t| > 1,96): esneklik hem sıfırdan hem birden farklıdır. Testin cevabı sınanan değere bağlıdır (§7.4)."
        ),
    ),
    Question(
        key="b02", concept="yuzde-99-guven-araligi", note=_note("7.6", "(7.8)"),
        prompt=(
            "$\\widehat\\beta = 0{,}45$, $\\operatorname{se}(\\widehat\\beta) = 0{,}15$ ve büyük örneklemde yüzde 99 "
            "kritik değeri 2,576'dır. Yüzde 99 güven aralığının alt sınırı **(1)**, üst sınırı **(2)** olur. "
            "(Virgülden sonra üç basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.064, 0.0005, "0,064"), NumberBlank(0.836, 0.0005, "0,836"))),
        explanation=(
            "Denklem 7.8: 0,45 ± 2,576 × 0,15 = 0,45 ± 0,3864, yani [0,064; 0,836]. Aralık sıfırı kapsamaz; "
            "$H_0: \\beta = 0$ yüzde 1 iki taraflı düzeyde de reddedilir. Aynı tahmin ve standart hatayla yüzde 95 "
            "aralığı (kritik değer 1,96) [0,156; 0,744] olurdu: güven düzeyi yükseldikçe aralık genişler (§7.6)."
        ),
    ),
    Question(
        key="b03", concept="iki-tarafli-p-iki-kuyrugu-sayar", note=_note("7.5", "Şekil 7.2"),
        prompt=(
            "Büyük örneklemde iki katsayı için $t_1 = -2{,}33$ ve $t_2 = 1{,}65$ bulunmuştur. Standart normal "
            "dağılımda $P(Z < -2{,}33) \\approx 0{,}0099$ ve $P(Z > 1{,}65) \\approx 0{,}0495$'tir. Birinci katsayının "
            "iki taraflı $p$-değeri **(1)**, ikincisininki **(2)** olur. (Virgülden sonra üç basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.020, 0.0005, "0,020"), NumberBlank(0.099, 0.0005, "0,099"))),
        explanation=(
            "İki taraflı testte “daha uç” hem pozitif hem negatif yöndeki büyük |t| değerlerini kapsar; p-değeri iki "
            "kuyruğun alanıdır (Şekil 7.2'deki taralı alanlar): 2 × 0,0099 ≈ 0,020 ve 2 × 0,0495 = 0,099. İşaret "
            "önemli değildir: t = +2,33 de aynı p'yi verir. Birinci sonuç yüzde 5'te, ikincisi yalnız yüzde 10'da "
            "reddedilir (§7.5)."
        ),
    ),
    Question(
        key="b04", concept="buyukluk-ve-araligi-olcekleme", note=_note("7.11"),
        prompt=(
            "Bir kira modelinde metro istasyonuna uzaklığın (km) katsayısı −45 TL, yüzde 95 güven aralığı [−60; −30] "
            "TL'dir. Diğer değişkenler sabitken 2 km daha uzak konutlar için tahmin edilen kira farkının nokta tahmini "
            "**(1)** TL, aralığın alt sınırı **(2)** TL olur. (Sayıları işaretiyle yazın.)"
        ),
        answer=FillBlanks((NumberBlank(-90.0, 0.5, "−90"), NumberBlank(-120.0, 0.5, "−120"))),
        explanation=(
            "Katsayı ve güven aralığının iki sınırı aynı sabitle çarpılır: 2 × (−45) = −90 TL ve 2 × [−60; −30] = "
            "[−120; −60] TL. Büyüklüğü aralığıyla ve iktisadi bağlamla (ör. ortalama kirayla karşılaştırarak) birlikte "
            "raporlamak, katsayının yalnız “anlamlı” olduğunu söylemekten daha açıklayıcıdır (§7.11)."
        ),
    ),
    Question(
        key="b05", concept="standart-hata-karekok-n-ile-kuculur", note=_note("7.2", "(7.2)"),
        prompt=(
            "400 kişilik bir ankette eğitim katsayısının standart hatası 0,080'dir. Veri üretim yapısı benzer kalırken "
            "standart hatanın yaklaşık $1/\\sqrt{n}$ ile orantılı küçüldüğünü kabul edin. n = 1.600 olursa standart "
            "hata yaklaşık **(1)** olur; standart hatayı 0,020'ye indirmek için yaklaşık **(2)** gözlem gerekir."
        ),
        answer=FillBlanks((NumberBlank(0.040, 0.0005, "0,040"), NumberBlank(6400.0, 0.5, "6.400"))),
        explanation=(
            "Denklem 7.2'nin paydasındaki $\\sqrt{\\sum (X_{ji} - \\bar X_j)^2}$ gözlem sayısıyla yaklaşık "
            "$\\sqrt{n}$ oranında büyür. Gözlem sayısı 4 katına çıkınca standart hata yarıya iner: 0,080/2 = 0,040. "
            "Standart hatayı dörtte bire indirmek için n'nin 16 katına, 6.400'e çıkması gerekir. Standart hata gözlem "
            "sayısıyla doğrusal değil, karekökle küçülür (§7.2)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="reddetme-esigi-katsayi-olceginde", note=_note("7.4", "(7.6)"),
        prompt=(
            "Büyük örneklemde iki taraflı $H_0: \\beta = a$ testi kritik değer $c$ ile yapılıyor; standart hata $s$'dir. "
            "$H_0$'ı reddettiren ve $a$'dan büyük olan tahminlerin alt eşiğini (bu değerin üstündeki her "
            "$\\widehat\\beta$ reddettirir) $a$, $c$ ve $s$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat\\beta^{*}",
            symbols=(
                Symbol("a", "a", "sıfır hipotezindeki değer", -1, 1),
                Symbol("c", "c", "kritik değer", 1.5, 3),
                Symbol("s", "s", "standart hata", 0.05, 0.5, aliases=_SE),
            ),
            answer="a + c*s",
            shown="a + c\\,s",
        ),
        explanation=(
            "Karar kuralı (Denklem 7.6): |t| > c, yani $|\\widehat\\beta - a| > c\\,s$. Pozitif tarafta "
            "$\\widehat\\beta > a + c\\,s$ olduğunda $H_0$ reddedilir; negatif taraftaki eşik $a - c\\,s$'dir. Örneğin "
            "a = 0, s = 0,12 ve c = 1,96 için eşik 0,235'tir: notlardaki 0,30 tahmini bu eşiği aştığı için t = 2,50 > "
            "1,96 (§7.4)."
        ),
    ),
    Question(
        key="e02", concept="guven-araligindan-standart-hata", note=_note("7.6", "(7.8)"),
        prompt=(
            "Bir makale yalnız yüzde 95 güven aralığını $[L;\\ U]$ raporluyor; aralık Denklem 7.8 ile kritik değer $c$ "
            "kullanılarak kurulmuş. Katsayının standart hatasını $L$, $U$ ve $c$ cinsinden yazın (paydayı parantezle "
            "yazın)."
        ),
        answer=Equation(
            lhs="\\operatorname{se}(\\widehat\\beta)",
            symbols=(
                Symbol("L", "L", "güven aralığının alt sınırı", -1, 0.5),
                Symbol("U", "U", "güven aralığının üst sınırı", 0.6, 2),
                Symbol("c", "c", "kritik değer", 1.5, 3),
            ),
            answer="(U - L)/(2*c)",
            shown="\\frac{U - L}{2c}",
        ),
        explanation=(
            "Aralık $\\widehat\\beta \\pm c\\,\\operatorname{se}(\\widehat\\beta)$ olduğundan genişliği $U - L = "
            "2c\\,\\operatorname{se}(\\widehat\\beta)$'dir; buradan se = (U − L)/(2c). Katsayı aralığın orta noktasıdır: "
            "(L + U)/2. WAGE1 eğitim aralığı [0,498; 0,700] ve c ≈ 1,965 için se ≈ 0,202/3,93 ≈ 0,051 (§7.6)."
        ),
    ),
    Question(
        key="e03", concept="artik-standart-sapmasi-serbestlik-derecesi", note=_note("7.2", "(7.3)"),
        prompt=(
            "Sabit terimli ve $k$ açıklayıcı değişkenli bir model $n$ gözlemle tahmin ediliyor; artıkların kareleri "
            "toplamı $\\text{HKT} = \\sum \\widehat u_i^2$'dir (Bölüm 4). Hata teriminin standart sapması σ'nın "
            "tahmini $\\widehat\\sigma$'yı HKT, $n$ ve $k$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat\\sigma",
            symbols=(
                Symbol("hkt", "\\text{HKT}", "artıkların kareleri toplamı", 100, 2000,
                       aliases=("HKT", "hkt", "SSR", "\\sum\\widehat u_i^2", "\\sum\\hat u_i^2")),
                Symbol("n", "n", "gözlem sayısı", 30, 600),
                Symbol("k", "k", "açıklayıcı değişken sayısı", 1, 6),
            ),
            answer="sqrt(hkt/(n - k - 1))",
            shown="\\sqrt{\\frac{\\text{HKT}}{n - k - 1}}",
        ),
        explanation=(
            "Denklem 7.3: $\\widehat\\sigma^2 = \\sum \\widehat u_i^2/(n - k - 1)$; payda artık serbestlik "
            "derecesidir. WAGE1 ücret modelinde HKT = 4966,303 ve n − k − 1 = 526 − 3 − 1 = 522 olduğundan "
            "$\\widehat\\sigma = \\sqrt{4966{,}303/522} \\approx 3{,}084$. $\\widehat\\sigma$, Denklem 7.2'de standart "
            "hatanın payındadır (§7.2)."
        ),
    ),
    Question(
        key="e04", concept="ters-yonde-tek-tarafli-p", note=_note("7.8"),
        prompt=(
            "Önceden belirlenmiş alternatif $H_1: \\beta > 0$'dır; fakat gözlenen $t$ negatiftir. Aynı $t$ için iki "
            "taraflı $p$-değeri $p_2$ ise tek taraflı $p$-değerini $p_2$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="p_{\\text{tek}}",
            symbols=(Symbol("p", "p_2", "iki taraflı p-değeri", 0.01, 0.99, aliases=_P_TWO),),
            answer="1 - p/2",
            shown="1 - \\frac{p_2}{2}",
        ),
        explanation=(
            "İki taraflı p her iki kuyruğu sayar: $p_2 = 2P(T > |t|)$. t negatifken sağ kuyruk olasılığı "
            "$P(T > t) = 1 - P(T < t) = 1 - p_2/2$'dir; WAGE1 deneyim katsayısında sol kuyruğu seçen biri için "
            "1 − 0,064/2 ≈ 0,968 (Uygulama Adım 6). Gözlenen t alternatifin yönündeyse tek taraflı p, $p_2/2$ olur "
            "(0,032). Tek taraflı test ters yöndeki büyük bir etkiyi alternatifin kanıtı saymaz (§7.8)."
        ),
    ),
    Question(
        key="e05", concept="log-duzey-yuzde-fark-araligi", note=_note("7.10", "Tablo 7.5"),
        prompt=(
            "Log–düzey bir ücret modelinde (bağımlı değişken ln(ücret)) bir açıklayıcı değişkenin katsayısı "
            "$\\widehat\\beta$, standart hatası $s$, güven aralığının kritik değeri $c$'dir. Diğer değişkenler sabitken "
            "bu değişkendeki $\\Delta x$ birimlik farkla ilişkili yaklaşık yüzde ücret farkı için güven aralığının alt "
            "sınırını (yüzde olarak, §7.10'daki yaklaşık yorumla) yazın."
        ),
        answer=Equation(
            lhs="\\text{Alt sınır (\\%)}",
            symbols=(
                Symbol("b", "\\widehat\\beta", "log–düzey modelin katsayısı", -0.2, 0.2, aliases=_BETA_HAT),
                Symbol("c", "c", "kritik değer", 1.5, 3),
                Symbol("s", "s", "standart hata", 0.001, 0.05, aliases=_SE),
                Symbol("d", "\\Delta x", "açıklayıcı değişkendeki fark", 1, 5, aliases=_DELTA_X),
            ),
            answer="100*(b - c*s)*d",
            shown="100\\,(\\widehat\\beta - c\\,s)\\,\\Delta x",
        ),
        explanation=(
            "Güven aralığının sınırları katsayı ölçeğinde $\\widehat\\beta \\pm c\\,s$'dir (Denklem 7.8); yaklaşık "
            "yüzde yorumu (§7.10) her sınırı 100·Δx ile çarpar. Tablo 7.5'in Sütun (2)'sinde kıdem katsayısı 0,022, "
            "standart hatası 0,003: dört yıllık kıdem farkı için nokta tahmini yaklaşık %8,8, yüzde 95 aralığı "
            "yaklaşık [%6,4; %11,2] (c ≈ 1,965). Yaklaşım küçük katsayılarda iyi çalışır; daha hassas dönüşüm Konu "
            "9'da işlenir (§7.10)."
        ),
    ),
)


KONU07_QUIZ = QuestionSet(
    topic_key="konu07",
    title="Konu 7: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"7.{number}" for number in range(1, 14)),  # §7.14 bölüm özetidir
)
