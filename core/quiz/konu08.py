"""Konu 8 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 8.1–8.13 ve Mini Quiz
8.1–8.13 maddelerini tekrar etmez: altı hipotezde tek katsayı–ortak hipotez ayrımı; beş iktisadi sorunun kısıta
çevrilmesi; dört sıfır hipotezinde kısıtlı model; n = 120, k = 4, SSR 400 ve 440 ile R² 0,60 ve 0,56 hesapları;
F(2,90), F(3,150), F(1,80) ve F(4,300) kararları; WAGE1 ortak testinin sekiz maddesi; n = 100, k = 4, R² = 0,20 genel
testi; t = 2,50, F(1,80) = 9 ve t = −3 sorularıyla F = t²; HPRICE1 ortak testinin altı maddesi; büyük örneklem
mantığının altı maddesi; yetenek, gönüllü anket, karesel model, 50 firma ve heteroskedastisite senaryoları; Tablo
8.6'nın sekiz maddesi; altı yanlış cümlenin düzeltilmesi ve mini quizlerdeki sorular. Aynı becerileri yeni bağlamlarla
ve yeni sayılarla sınar. Ortak testler klasik EKK varsayımlarına dayanır; heteroskedastisiteye dayanıklı ortak test
Konu 12'nin, logaritmik modellerde tam yüzde dönüşümü Konu 9'un konusudur; sorular bunları kullanmaz.
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


_BETA_HAT = ("\\widehat\\beta", "\\hat\\beta", "β̂", "betahat", "bhat", *beta_hat_aliases(1))
_SE_OF = tuple(f"{name}({hat})" for name in ("se", "SE", "SH", "sh") for hat in (*_BETA_HAT, "b"))
"""se(β̂) yazımları (\\operatorname{se}(\\hat\\beta_1) dahil; metin komutları okunmadan önce silinir): ``s`` sembolüne
çevrilir. Uzun yazımlar önce eşleştiği için ``se`` ve β̂ yazımlarıyla karışmaz."""
_SE = (*_SE_OF, "se", "SE", "SH", "sh")
_R2_UR = ("R_UR^2", "R_UR²", "R²_UR", "R2_UR", "R2UR", "R^2_UR", "R_ur^2", "R_ur²", "R²_ur", "R2_ur", "R2ur")
_R2_R = ("R_R^2", "R_R²", "R²_R", "R2_R", "R2R", "R^2_R", "R_r^2", "R_r²", "R²_r", "R2_r", "R2r")
"""R² yazımları; ``R^2_{UR}`` önce ``R_UR^2`` biçimine çevrilir (``core.quiz.expression.parse``)."""


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="kisit-sayisi-bagimsiz-esitlikler", note=_note("8.2"),
        prompt=(
            "Beş açıklayıcı değişkenli bir modelde sıfır hipotezi $H_0: \\beta_1 = \\beta_2 = \\beta_3$ ve "
            "$\\beta_4 = 0$ biçimindedir. Kısıt sayısı $q$ kaçtır?"
        ),
        answer=MultipleChoice(("1", "2", "3", "4"), correct=2),
        explanation=(
            "Kısıt sayısı bağımsız eşitliklerin sayısıdır. $\\beta_1 = \\beta_2 = \\beta_3$ iki eşitlikle yazılır "
            "($\\beta_1 - \\beta_2 = 0$ ve $\\beta_2 - \\beta_3 = 0$); üçüncü bir eşitlik ($\\beta_1 = \\beta_3$) "
            "bunlardan türetilir. $\\beta_4 = 0$ bir kısıt daha ekler: q = 3. Hipotezde dört katsayı adının geçmesi dört "
            "kısıt olduğu anlamına gelmez (§8.2)."
        ),
    ),
    Question(
        key="k02", concept="ayni-uyum-kaybi-daha-fazla-kisit", note=_note("8.4"),
        prompt=(
            "Aynı kısıtsız model için iki ortak test düşünün: birinde tek kısıt, ötekinde dört kısıt vardır ve ikisinde "
            "de $SSR_R - SSR_{UR}$ farkı aynıdır. Dört kısıtlı testin F istatistiği tek kısıtlı testinkine göre "
            "nasıldır?"
        ),
        answer=MultipleChoice(
            (
                "Dört katıdır: kısıt arttıkça kanıt birikir",
                "Dörtte biridir: uyum kaybı dört kısıta bölünür",
                "Aynıdır: F yalnız SSR farkına bağlıdır",
                "Yarısıdır: F, q'nun kareköküne bölünür",
            ),
            correct=1,
        ),
        explanation=(
            "F'nin payı kısıt başına ortalama uyum kaybıdır: $(SSR_R - SSR_{UR})/q$. Payda, yani kısıtsız modelin artık "
            "değişkenliği iki testte aynı olduğundan dört kısıtlı testin F'si tek kısıtlınınkinin dörtte biridir. Daha "
            "fazla kısıt doğal olarak daha fazla uyum kaybı yaratabileceği için kayıp kısıt sayısına göre ölçeklenir "
            "(§8.4)."
        ),
    ),
    Question(
        key="k03", concept="kucuk-f-buyuk-ortak-p", note=_note("8.5"),
        prompt=(
            "WAGE1 modelinde başka bir ortak hipotez için $F(2, 522) = 0{,}02$ bulunsaydı ortak $p$-değeri yaklaşık kaç "
            "olurdu?"
        ),
        answer=MultipleChoice(
            (
                "0,02: F küçük olduğu için p de küçüktür",
                "0,01: F testi iki kuyruğu birlikte sayar",
                "Hesaplanamaz: F istatistiği 1'den küçük olamaz",
                "0,98: dağılımın neredeyse tamamı 0,02'nin sağındadır",
            ),
            correct=3,
        ),
        explanation=(
            "F testi üst kuyruk testidir: p = P(F(2, 522) > 0,02) ≈ 0,98. Çok küçük F, kısıtların yarattığı uyum "
            "kaybının artık değişkenliğine göre ihmal edilebilir olduğunu gösterir; veri $H_0$ ile uyumludur ve $H_0$ "
            "reddedilemez. F negatif olamaz ama 1'den küçük olabilir (§8.5)."
        ),
    ),
    Question(
        key="k04", concept="esitlik-kisiti-sonucunun-yorumu", note=_note("8.6", "Kod 8.1"),
        prompt=(
            "Kod 8.1'deki WAGE1 modelinde `model.f_test(\"exper = tenure\")` yazıldığında $F(1, 522) \\approx 24{,}58$ "
            "ve $p < 0{,}001$ bulunuyor. Bu sonuç ne gösterir?"
        ),
        answer=MultipleChoice(
            (
                "Eğitim sabitken deneyim ve kıdem katsayılarının eşit olduğu hipotezi reddedilir",
                "Deneyim ve kıdem katsayılarının toplamının sıfır olduğu hipotezi reddedilir",
                "Deneyim ve kıdem katsayılarının ikisi de ayrı ayrı yüzde 5'te anlamlıdır",
                "Kıdemin ücrete nedensel etkisi deneyiminkinden kesin olarak büyüktür",
            ),
            correct=0,
        ),
        explanation=(
            "`f_test(\"exper = tenure\")` tek bir doğrusal kısıtı sınar: $H_0: \\beta_{\\text{deneyim}} = "
            "\\beta_{\\text{kıdem}}$ (q = 1). Reddedilmesi, eğitim sabitken bir yıllık ek deneyim ile bir yıllık ek "
            "kıdemin ücretle kısmi ilişkilerinin eşit olmadığını gösterir (0,0223 ile 0,1693). Birlikte sıfır olma "
            "hipotezi başka bir testtir (F = 53,31, q = 2); deneyimin ayrı p-değeri 0,064'tür ve gözlemsel veride "
            "sonuç nedensel etki olarak okunmaz (§8.6)."
        ),
    ),
    Question(
        key="k05", concept="r2-bicimi-yuvarlamaya-duyarli", note=_note("8.9", "Tablo 8.2"),
        prompt=(
            "Tablo 8.2'deki üç basamaklı $R^2$ değerleri (0,621 ve 0,672) $R^2$ formülüne konursa $F \\approx 6{,}53$ "
            "bulunur; $SSR$ formülü ise 6,61 verir. Farkın nedeni nedir?"
        ),
        answer=MultipleChoice(
            (
                "R² değerlerinin üç basamağa yuvarlanması",
                "İki formülün farklı hipotezleri sınaması",
                "Kısıtlı modelin farklı gözlemlerle tahmin edilmesi",
                "R² formülünün yalnız genel teste uygulanabilmesi",
            ),
            correct=0,
        ),
        explanation=(
            "Aynı bağımlı değişken ve aynı 88 gözlemle iki formül cebirsel olarak eşdeğerdir; tam R² değerleriyle "
            "(0,62080 ve 0,67236) R² formülü de 6,61'i verir. Fark, paydaki iki yakın R² arasındaki küçük farkın "
            "(≈ 0,05) yuvarlamaya duyarlı olmasından gelir: üçüncü basamaktaki küçük hatalar bu farkı yaklaşık %1 "
            "değiştirir. Bu yüzden §8.6'da R² formülü beş basamaklı R² ile yazılır (§8.9)."
        ),
    ),
    Question(
        key="k06", concept="normal-olmayan-hatada-yaklasik-gecerlilik", note=_note("8.10", "Tablo 8.3"),
        prompt=(
            "Bir ücret örnekleminde 2.000 çalışan vardır ve artıklar belirgin biçimde sağa çarpıktır. Eğitim "
            "katsayısının geleneksel $t$ testi için aşağıdakilerden hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Test kullanılamaz: geleneksel t testi yalnız normal dağılan hatalarla geçerlidir",
                "n > 30 olduğu için test, küçük örneklemdeki gibi tam ve kesin biçimde geçerlidir",
                "Düzenlilik koşulları altında t yaklaşık standart normaldir; test yaklaşık geçerlidir",
                "Çarpıklık yalnız katsayı tahminini yanlı yapar; test sonucunu hiç etkilemez",
            ),
            correct=2,
        ),
        explanation=(
            "Asimptotik normallik: uygun biçimde standartlaştırılmış tahmin edicinin dağılımı örneklem büyüdükçe "
            "standart normale yaklaşır; bu yüzden büyük örneklemde t ve F testleri hatalar normal olmasa da yaklaşık "
            "geçerlidir (Tablo 8.3'te çarpıklığı 2 olan hatalarla ret oranları %5 çevresindedir). Sonuç yaklaşıktır, "
            "kesin değildir; evrensel bir “n > 30” eşiği yoktur (§8.10)."
        ),
    ),
    Question(
        key="k07", concept="genel-f-ile-f-test-ayni-hipotezde-esit", note=_note("8.12", "Tablo 8.5"),
        prompt=(
            "WAGE1 ücret modelinde (üç açıklayıcı değişken) Statsmodels ana özetindeki `Prob (F-statistic)` ile bir "
            "`f_test` çıktısındaki ortak p-değeri hangi durumda aynı olur?"
        ),
        answer=MultipleChoice(
            (
                "f_test'e modeldeki tek bir katsayı için kısıt yazıldığında",
                "Örneklem çok büyük olduğunda, yani yaklaşık olarak",
                "f_test'e bütün eğimlerin sıfır olduğu kısıtlar yazıldığında",
                "Hiçbir zaman: iki çıktı farklı F dağılımlarına dayanır",
            ),
            correct=2,
        ),
        explanation=(
            "`F-statistic` ve `Prob (F-statistic)` her zaman genel anlamlılık testine aittir: bütün eğimler birlikte "
            "sıfır. `f_test` kullanıcının yazdığı kısıtları sınar; ikisi yalnız aynı hipotez yazıldığında çakışır: WAGE1'de "
            "`f_test(\"educ = 0, exper = 0, tenure = 0\")` da F = 76,87 verir. Tablo 8.5'teki ayrım budur (§8.12)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="tahminler-birlikte-hareket-eder", note=_note("8.1"),
        prompt=(
            "İki açıklayıcı değişken güçlü biçimde pozitif korelasyonluysa, katsayı tahminleri tekrarlı örneklemede "
            "genellikle ters yönde birlikte hareket eder: biri yüksek tahmin edildiğinde öteki düşük tahmin edilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "§8.1'deki ikinci gerekçe: tahminler arasındaki örnekleme ilişkisi ortak belirsizliğin parçasıdır. İki "
            "değişken birlikte hareket ettiğinde veri ortak katkıyı iyi, ayrı katkıları zayıf belirler; biri fazla "
            "tahmin edildiğinde öteki eksik tahmin edilir. Konu 8 Sezgi Deney 2'nin saçılım grafiği ρ = 0,9'da bu ters "
            "ilişkiyi gösterir; ayrı t testleri onu tek bir kararda birleştiremez (§8.1)."
        ),
    ),
    Question(
        key="d02", concept="ic-ice-olmayan-modeller", note=_note("8.3"),
        prompt=(
            "`wage ~ educ + exper` (eğitim ve deneyim) modeli ile `wage ~ educ + tenure` (eğitim ve kıdem) modeli bu "
            "bölümdeki klasik F karşılaştırmasıyla doğrudan karşılaştırılabilir; biri kısıtlı, öteki kısıtsız model "
            "sayılır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Klasik F karşılaştırması iç içe modeller içindir: kısıtlı model kısıtsız modelden doğrusal kısıtlarla elde "
            "edilebilmelidir. İki modelin her biri ötekinde olmayan bir değişken içerir; hiçbiri ötekinin kısıtlı hâli "
            "değildir. İkisi de eğitim, deneyim ve kıdemi içeren kısıtsız modelle (`wage ~ educ + exper + tenure`) ayrı "
            "ayrı karşılaştırılabilir (§8.3)."
        ),
    ),
    Question(
        key="d03", concept="f-paydasi-kisitsiz-modelin-artik-varyansi", note=_note("8.4"),
        prompt=(
            "F istatistiğinin paydası $SSR_{UR}/(n - k - 1)$, kısıtsız modelin artık varyans tahminidir; kısıtlı "
            "modelin artıkları paydaya girmez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Payda kısıtsız modeldeki gözlem başına artık değişkenliğini temsil eder; n − k − 1 kısıtsız modelin artık "
            "serbestlik derecesidir. Kısıtlı model yalnız paya, uyum kaybı $SSR_R - SSR_{UR}$ üzerinden girer. WAGE1'de "
            "payda 4966,303/522 ≈ 9,51'dir (§8.4)."
        ),
    ),
    Question(
        key="d04", concept="dusuk-r2-ile-yuksek-genel-f", note=_note("8.7"),
        prompt=(
            "n = 1.000 gözlem ve k = 2 açıklayıcı değişkenli bir modelde R² = 0,05'tir. Bu modelde genel F testi yüzde 5 "
            "düzeyinde reddedilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$F = (R^2/k)/[(1 - R^2)/(n - k - 1)] = (0{,}05/2)/(0{,}95/997) \\approx 26{,}2$; F(2, 997) için yüzde 5 "
            "kritik değer yaklaşık 3'tür ve p çok küçüktür. Büyük örneklemde düşük R² ile de eğimlerin birlikte sıfır "
            "olduğu hipotezi güçlü biçimde reddedilebilir. Genel F modelin iyi uyduğunu ya da doğru olduğunu göstermez "
            "(§8.7)."
        ),
    ),
    Question(
        key="d05", concept="f-iki-tarafli-t-testine-denk", note=_note("8.8"),
        prompt=(
            "Tek kısıtta $F = t^2$ olduğu için, önceden belirlenmiş $H_1: \\beta > 0$ yönlü testin $p$-değeri de F "
            "testinin $p$-değerine eşittir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "F = t² eşitliği iki taraflı t testi içindir: F, t'nin karesini aldığı için işareti kaybeder ve her iki "
            "yöndeki büyük |t| değerlerini sayar. Yönlü testte tek taraflı p, t alternatif yöndeyse F'nin p-değerinin "
            "yarısıdır. WAGE1 deneyim katsayısında F testi ve iki taraflı t testi p = 0,064, tek taraflı test p = 0,032 "
            "verir (§8.8)."
        ),
    ),
    Question(
        key="d06", concept="buyuk-n-dar-aralik-yanlis-merkez", note=_note("8.11"),
        prompt=(
            "Eksik değişken yanlılığı olan bir modelde örneklem çok büyüdükçe güven aralıkları daralır ve yüzde 95 "
            "aralıkların gerçek katsayıyı kapsama oranı %95'e yaklaşır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Aralıklar daralır ama yanlış merkezin çevresinde: $\\mathbb{E}(u \\mid X) \\neq 0$ ise büyük n yanlış "
            "değerin daha kesin tahminini verir. Konu 8 Sezgi Deney 3'ün varsayılan ayarlarında kısa modelin kapsama "
            "oranı n = 50'de %66,4, n = 800 ve n = 2.000'de %0'dır; Z modeldeyken %95 çevresinde kalır (§8.11)."
        ),
    ),
    Question(
        key="d07", concept="kisitli-ve-kisitsiz-model-ayni-gozlemler", note=_note("8.13"),
        prompt=(
            "Bir araştırmacı kısıtsız modele eksik değerleri olan bir değişken ekliyor; kısıtsız model 510, bu "
            "değişkeni içermeyen kısıtlı model 526 gözlemle tahmin ediliyor. İki modelin SSR değerleriyle hesaplanan F "
            "istatistiği bu durumda da geçerlidir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "SSR ve R² karşılaştırması iki modelin aynı gözlemlerle tahmin edilmesini gerektirir; aksi hâlde SSR farkı "
            "kısıtların yarattığı uyum kaybını değil, farklı örneklemleri de yansıtır (Sık Yapılan Hatalar, madde 3). "
            "Kısıtlı model de kısıtsız modelin kullandığı 510 gözlemle yeniden tahmin edilmelidir; `statsmodels` "
            "içindeki `f_test` bunu kendiliğinden sağlar, çünkü kısıtları aynı modelin tahmininde sınar (§8.13)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="f-ssr-bicimiyle-hesap", note=_note("8.4"),
        prompt=(
            "n = 206 gözlemli, k = 5 açıklayıcı değişkenli bir kısıtsız modelde $SSR_{UR} = 1.100$'dür; üç kısıt "
            "uygulanınca $SSR_R = 1.250$ bulunuyor. Payda serbestlik derecesi **(1)**, F istatistiği **(2)** olur. "
            "(F için virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(200.0, 0.5, "200"), NumberBlank(9.09, 0.005, "9,09"))),
        explanation=(
            "Payda serbestlik derecesi kısıtsız modelinkidir: n − k − 1 = 206 − 5 − 1 = 200. "
            "F = [(1.250 − 1.100)/3]/(1.100/200) = 50/5,5 ≈ 9,09. Pay kısıt başına uyum kaybı, payda kısıtsız modelin "
            "artık varyansıdır (§8.4)."
        ),
    ),
    Question(
        key="b02", concept="genel-test-ssr-bicimiyle", note=_note("8.7", "Tablo 8.1", "Kod 8.2"),
        prompt=(
            "WAGE1 ücret modelinde kısıtsız modelin artık kareleri toplamı $SSR_{UR} = 4966{,}303$ (Bölüm 4'teki HKT), "
            "§8.6'daki beş basamaklı değerle $R^2_{UR} = 0{,}30642$'dir. $R^2 = 1 - SSR/\\text{TKT}$ olduğundan toplam "
            "kareler toplamı TKT **(1)** olur. Genel testte kısıtlı model yalnız sabit terimi içerdiği için "
            "$SSR_R = \\text{TKT}$'dir; bu testte kısıtların yarattığı uyum kaybı $SSR_R - SSR_{UR}$ **(2)** olur. "
            "(Virgülden sonra bir basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(7160.4, 0.05, "7160,4"), NumberBlank(2194.1, 0.05, "2194,1"))),
        explanation=(
            "TKT = SSR_UR/(1 − R²_UR) = 4966,303/0,69358 ≈ 7160,4 (Bölüm 4'te aynı veri için TKT = 7160,414). Yalnız "
            "sabitli modelin tahmini ortalamadır ve artıkları $Y_i - \\bar Y$'dir; bu yüzden onun SSR'si TKT'dir. Uyum "
            "kaybı TKT − SSR_UR ≈ 2194,1, yani kısıtsız modelin açıklanan kareler toplamıdır (MKT). SSR biçimiyle "
            "F = (2194,1/3)/(4966,303/522) ≈ 76,87: Kod 8.2'deki `F-statistic`. Genel test, kısıtlı modeli yalnız "
            "sabitli model olan özel bir ortak testtir (§8.7)."
        ),
    ),
    Question(
        key="b03", concept="kritik-f-kritik-t-nin-karesi", note=_note("8.8"),
        prompt=(
            "WAGE1 ücret modelinde (n − k − 1 = 522) iki taraflı $t$ testinin kritik değerleri yüzde 5 için "
            "$t_{0{,}025;\\,522} = 1{,}9645$, yüzde 1 için $t_{0{,}005;\\,522} = 2{,}5853$'tür. Tek kısıtlı $F(1, 522)$ "
            "testinin yüzde 5 kritik değeri **(1)**, yüzde 1 kritik değeri **(2)** olur. (Virgülden sonra iki basamak "
            "yazın.)"
        ),
        answer=FillBlanks((NumberBlank(3.86, 0.005, "3,86"), NumberBlank(6.68, 0.005, "6,68"))),
        explanation=(
            "Tek kısıtta F = t² olduğu için kritik değerler de aynı ilişkiyi taşır: $1{,}9645^2 \\approx 3{,}86$ ve "
            "$2{,}5853^2 \\approx 6{,}68$. |t| > 1,9645 ile F > 3,86 aynı kararı verir. Eğitim için t = 11,6795 ve "
            "F = 136,41 iki ölçekte de kritik değerlerin çok üstündedir (§8.8)."
        ),
    ),
    Question(
        key="b04", concept="reddettiren-en-kucuk-uyum-kaybi", note=_note("8.9", "Tablo 8.2"),
        prompt=(
            "Tablo 8.2'deki HPRICE1 ortak testinde $SSR_{UR} = 300723{,}805$, q = 2 ve n − k − 1 = 84'tür; "
            "$F(2, 84)$ için yüzde 5 kritik değer yaklaşık 3,105'tir. $H_0$'ı yüzde 5 düzeyinde reddettiren en küçük "
            "$SSR_R$ **(1)** olur (virgülden sonra bir basamak); bu, kısıtsız modelin SSR'sine göre yüzde **(2)** "
            "uyum kaybıdır (virgülden sonra iki basamak)."
        ),
        answer=FillBlanks((NumberBlank(322955.9, 15.0, "322955,9"), NumberBlank(7.39, 0.005, "7,39"))),
        explanation=(
            "F = c koşulundan $SSR_R^{*} = SSR_{UR}\\,[1 + q\\,c/(n - k - 1)] = 300723{,}805 \\times (1 + 2 \\times "
            "3{,}105/84) \\approx 322955{,}9$; uyum kaybı $100\\,q\\,c/(n - k - 1) \\approx$ %7,39. Tablo 8.2'deki kısıtlı "
            "model SSR'si 348053,432 bu eşiği aşar (%15,7 kayıp); bu yüzden F = 6,61 kritik değerden büyüktür ve H₀ "
            "reddedilir (§8.9)."
        ),
    ),
    Question(
        key="b05", concept="tutarlilik-ve-standartlastirma", note=_note("8.10", "Tablo 8.3"),
        prompt=(
            "Tutarlı bir tahmin edicinin standart sapması yaklaşık $1/\\sqrt{n}$ ile orantılı küçülüyor ve n = 100'de "
            "0,20'dir. n = 400'de tahminlerin standart sapması yaklaşık **(1)** olur. Aynı tahmin edicinin "
            "standartlaştırılmış biçimi (tahmin eksi gerçek değer, bölü tahmin edilen standart hata; Tablo 8.3'teki "
            "standartlaştırılmış eğim, yani t) n = 400'de yaklaşık **(2)** standart sapmaya sahiptir. (Virgülden sonra "
            "iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.10, 0.005, "0,10"), NumberBlank(1.0, 0.005, "1,00"))),
        explanation=(
            "Tutarlılık: yayılım örneklemle küçülür, 0,20/2 = 0,10. Standartlaştırma ise tahmini standart hataya "
            "bölerek yayılımı her n'de yaklaşık 1'e sabitler; asimptotik normallik bu standartlaştırılmış istatistiğin "
            "biçiminin "
            "standart normale yaklaşmasıdır. Tablo 8.3'te standartlaştırılmış eğimin standart sapması n = 25, 100 ve "
            "500'de 1,038; 0,985 ve 0,998'dir (§8.10)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="genel-test-icin-gereken-r2", note=_note("8.7"),
        prompt=(
            "$k$ açıklayıcı değişkenli ve $n$ gözlemli bir modelde genel F testinin kritik değeri $c$'dir. "
            "$F = (R^2/k)/[(1 - R^2)/(n - k - 1)]$ bağıntısını kullanarak testin tam sınırda kaldığı ($F = c$) $R^2$ "
            "değerini $c$, $k$ ve $n$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="R^2_{*}",
            symbols=(
                Symbol("c", "c", "genel testin kritik değeri", 1.5, 4),
                Symbol("k", "k", "açıklayıcı değişken sayısı", 1, 6),
                Symbol("n", "n", "gözlem sayısı", 60, 500),
            ),
            answer="c*k/(c*k + n - k - 1)",
            shown="\\frac{c\\,k}{c\\,k + n - k - 1}",
        ),
        explanation=(
            "F = c eşitliğinden $R^2 (n - k - 1) = c\\,k\\,(1 - R^2)$, yani $R^2 = c\\,k/(c\\,k + n - k - 1)$. Bu "
            "eşiğin üstündeki R² genel testi reddettirir. Eşik n ve k'ye bağlıdır: n = 60 iken k = 2 için yaklaşık 0,100, "
            "k = 8 için yaklaşık 0,250; n = 200 ve k = 2 için yaklaşık 0,030. Aynı R², küçük örneklemde ve çok sayıda "
            "açıklayıcı değişkenle daha zayıf kanıttır (§8.7)."
        ),
    ),
    Question(
        key="e02", concept="ssr-bicimden-r2-bicime", note=_note("8.4"),
        prompt=(
            "Kısıtlı ve kısıtsız model aynı bağımlı değişkeni ve aynı gözlemleri kullanıyor; her ikisinde "
            "$SSR = (1 - R^2)\\,\\text{TKT}$ (SSR, Bölüm 4'teki HKT'dir) ve TKT ortaktır. SSR biçimindeki F'yi "
            "$R^2_{UR}$, $R^2_R$, $q$ ve "
            "$d = n - k - 1$ cinsinden yazın ($R^2_{UR}$ için u, $R^2_R$ için r yazabilirsiniz)."
        ),
        answer=Equation(
            lhs="F",
            symbols=(
                Symbol("u", "R^2_{UR}", "kısıtsız modelin R²'si", 0.3, 0.6, aliases=_R2_UR),
                Symbol("r", "R^2_{R}", "kısıtlı modelin R²'si", 0.05, 0.25, aliases=_R2_R),
                Symbol("q", "q", "kısıt sayısı", 1, 4),
                Symbol("d", "d", "kısıtsız modelin artık serbestlik derecesi", 50, 500),
            ),
            answer="((u - r)/q)/((1 - u)/d)",
            shown="\\frac{(R^2_{UR} - R^2_{R})/q}{(1 - R^2_{UR})/d}",
        ),
        explanation=(
            "$SSR_R - SSR_{UR} = (1 - R^2_R)\\text{TKT} - (1 - R^2_{UR})\\text{TKT} = (R^2_{UR} - R^2_R)\\text{TKT}$ ve "
            "$SSR_{UR} = (1 - R^2_{UR})\\text{TKT}$. Pay ile paydadaki TKT sadeleşir. Sadeleşme ortak TKT'ye dayandığı "
            "için R² biçimi yalnız aynı bağımlı değişken ve aynı gözlemlerle kullanılabilir (§8.4)."
        ),
    ),
    Question(
        key="e03", concept="tek-kisitta-f-katsayidan", note=_note("8.8"),
        prompt=(
            "Tek kısıtlı $H_0: \\beta = a$ hipotezi için F istatistiğini katsayı tahmini $\\widehat\\beta$, $a$ ve "
            "standart hata $s$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="F",
            symbols=(
                Symbol("b", "\\widehat\\beta", "katsayı tahmini", -1, 2, aliases=_BETA_HAT),
                Symbol("a", "a", "sıfır hipotezindeki değer", -1, 1),
                Symbol("s", "s", "standart hata", 0.05, 0.5, aliases=_SE),
            ),
            answer="((b - a)/s)^2",
            shown="\\left(\\frac{\\widehat\\beta - a}{s}\\right)^2",
        ),
        explanation=(
            "Tek kısıtta F = t² ve $t = (\\widehat\\beta - a)/s$ olduğundan $F = [(\\widehat\\beta - a)/s]^2$. F işaret "
            "taşımaz: $\\widehat\\beta - a$'nın işareti karede kaybolur. WAGE1'de eğitim için (0,5990/0,05128)² ≈ "
            "136,4; notlarda t = 11,6795 ile F(1, 522) = 136,41 (§8.8)."
        ),
    ),
    Question(
        key="e04", concept="ayri-testlerde-en-az-bir-yanlis-ret", note=_note("8.1"),
        prompt=(
            "Birbirinden bağımsız $m$ ayrı testin her biri α düzeyinde yapılıyor ve bütün sıfır hipotezleri doğru. En az "
            "bir testin yanlışlıkla reddetme olasılığını α ve $m$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="P(\\text{en az bir yanlış ret})",
            symbols=(
                Symbol("alpha", "\\alpha", "her testin anlamlılık düzeyi", 0.01, 0.1,
                       aliases=("\\alpha", "α", "alfa")),
                Symbol("m", "m", "test sayısı", 2, 20),
            ),
            answer="1 - (1 - alpha)^m",
            shown="1 - (1 - \\alpha)^m",
        ),
        explanation=(
            "Bağımsız testlerde hiçbirinin reddetmeme olasılığı $(1 - \\alpha)^m$'dir; en az bir yanlış ret olasılığı "
            "bunun tümleyenidir. α = 0,05 ve m = 2 için 1 − 0,95² = 0,0975: bağımsız iki ayrı yüzde 5 testi ortak yanlış "
            "karar olasılığını %5'te tutmaz. Test istatistikleri birlikte hareket ediyorsa oran daha düşüktür; Konu 8 "
            "Sezgi Deney 2'de ρ = 0 iken “en az bir t testi reddeder” oranı bu değere yakındır; ρ yükseldikçe %5'e "
            "doğru iner (varsayılan ρ = 0,9'da %7,2). "
            "Ortak F testi her durumda %5 hedefindedir (§8.1)."
        ),
    ),
    Question(
        key="e05", concept="buyuk-n-kisa-modelin-merkezi", note=_note("8.11"),
        prompt=(
            "Konu 8 Sezgi Deney 3'ün veri üretim sürecinde $Y = 1 + \\beta X + \\gamma Z + u$, $Z = \\rho X + "
            "\\sqrt{1 - \\rho^2}\\,e$ ve $X$, $e$, $u$ standart normaldir. Z dışarıda bırakılıp Y yalnız X'e regres "
            "edilirse örneklem büyüdükçe X katsayısının tahminleri hangi değerin çevresinde yoğunlaşır? $\\beta$, "
            "$\\gamma$ ve $\\rho$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widetilde\\beta_X",
            symbols=(
                Symbol("b", "\\beta", "X'in gerçek katsayısı", 0, 1,
                       aliases=("\\beta_1", "\\beta_X", "β_1", "β₁", "β_X", "beta_1", "beta1", "\\beta", "β", "beta")),
                Symbol("g", "\\gamma", "Z'nin Y üzerindeki etkisi", -1, 1, aliases=("\\gamma", "γ", "gamma")),
                Symbol("r", "\\rho", "X ile Z arasındaki korelasyon", -0.9, 0.9, aliases=("\\rho", "ρ", "rho")),
            ),
            answer="b + g*r",
            shown="\\beta + \\gamma\\,\\rho",
        ),
        explanation=(
            "Konu 6'daki eksik değişken formülü: kısa model eğiminin merkezi (beklenen değeri) β + γδ'dır; δ, Z'nin "
            "X'e yardımcı regresyonundaki eğimdir: Cov(X, Z)/Var(X) = ρ (Var(X) = 1). Örneklem büyüdükçe tahminler bu "
            "değerin çevresinde daralır: β = γ = ρ = 0,5 iken 0,75. Büyük örneklem belirsizliği azaltır ama merkezi "
            "düzeltmez (§8.11)."
        ),
    ),
)


KONU08_QUIZ = QuestionSet(
    topic_key="konu08",
    title="Konu 8: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"8.{number}" for number in range(1, 14)),  # §8.14 bölüm özetidir
)
