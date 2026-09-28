"""Konu 3 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 3.1–3.13 ve Mini Quiz
3.1–3.13 maddelerini tekrar etmez: genel konuları regresyon sorusuna dönüştürme; konut, talep, sınav ve işsizlik
sorularında Y ile X'i belirleme; E(ücret | eğitim = 16) = 8 gibi koşullu ortalamaların sözlü yorumu; dört modelin
parametrelere göre doğrusallığı; hata teriminde kalan faktörleri sayma; E(u | X) = 0 ile ilgili dört ifade;
notasyonun anakütle–örneklem sınıflandırması; EKK ölçütüne ilişkin dört soru; (1, 2), (2, 4), (3, 5) verisiyle EKK
hesabı; Python çıktısından 12 ve 16 yıllık tahminler; dört yanlış katsayı yorumunun düzeltilmesi; 10 ve 14 yıllık
tahmin–artık hesabı; satış = 15 + 2,4 reklam uygulaması ve mini quizlerdeki sorular. Aynı becerileri yeni
bağlamlarla ve yeni sayılarla sınar. R² ve kareler toplamları Konu 4'ün; standart hata, t, p-değeri ve güven
aralığı Konu 7'nin konusudur; sorular bunlara başvurmaz.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol, bar_aliases, beta_aliases, beta_hat_aliases
from core.quiz.model import (
    Equation,
    FillBlanks,
    MultipleChoice,
    NumberBlank,
    Question,
    QuestionSet,
    TextBlank,
    TrueFalse,
)


def _note(section: str, *objects: str) -> NoteRef:
    return NoteRef(section, 0, tuple(objects))


_SXY = ("S_XY", "S_xy", "s_XY", "s_xy", "S_YX", "S_yx")
_SXX = ("S_XX", "S_xx", "s_XX", "s_xx")
_SYY = ("S_YY", "S_yy", "s_YY", "s_yy")


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="egitim-duzeyi-ortalamalari", note=_note("3.1", "Şekil 3.1"),
        prompt="Şekil 3.1'deki daha belirgin noktalar neyi gösterir?",
        answer=MultipleChoice(
            (
                "Regresyon doğrusunun o eğitim düzeyi için verdiği tahmini ücreti",
                "Aynı eğitim düzeyindeki çalışanların örneklemdeki ortalama ücretini",
                "Anakütledeki koşullu ortalamayı, $\\mathbb{E}(\\text{ücret} \\mid \\text{eğitim})$ değerini",
                "O eğitim düzeyinde en yüksek saatlik ücreti alan çalışanı",
            ),
            correct=1,
        ),
        explanation=(
            "Şekil 3.1'de soluk noktalar bireysel çalışanları, belirgin noktalar aynı eğitim düzeyindeki çalışanların "
            "ortalama ücretini, doğru ise bütün örneklemi tek bir doğrusal ilişkiyle özetleyen tahmini gösterir. "
            "Belirgin noktalar örneklemden hesaplanır; anakütledeki $\\mathbb{E}(\\text{ücret} \\mid \\text{eğitim})$ "
            "bilinmez, bu ortalamalar onun örneklemdeki karşılığıdır. Aynı düzeyde ortalamanın çevresine dağılan "
            "ücretler, modelin temsil etmesi gereken ikinci özelliktir (§3.1, Şekil 3.1)."
        ),
    ),
    Question(
        key="k02", concept="parametrelere-gore-dogrusal-olmayan", note=_note("3.4"),
        prompt="Aşağıdaki modellerden hangisi parametrelere göre doğrusal **değildir**?",
        answer=MultipleChoice(
            (
                "$Y = \\beta_0 + \\beta_1 \\sqrt{X} + u$",
                "$Y = \\beta_0 + \\beta_1 (1/X) + u$",
                "$Y = \\beta_0 + \\beta_1 X + \\beta_2 X^2 + u$",
                "$Y = \\beta_0 + X^{\\beta_1} + u$",
            ),
            correct=3,
        ),
        explanation=(
            "Ekonometride doğrusallık esas olarak parametrelere göredir. X'in karekökü, tersi ya da karesi modele yeni "
            "birer değişken gibi girer; her parametre yalnız bir sayıyla ya da bir değişkenle çarpılır. $X^{\\beta_1}$ "
            "teriminde ise parametre üsse girmiştir; model bu parametre bakımından doğrusal değildir (§3.4)."
        ),
    ),
    Question(
        key="k03", concept="dislanan-faktor-ve-kosullu-ortalama", note=_note("3.6", "(3.5)"),
        prompt=(
            "Bir konut modeli $\\text{fiyat} = \\beta_0 + \\beta_1\\,\\text{büyüklük} + u$ biçiminde kuruluyor. "
            "Aşağıdaki durumlardan hangisi $\\mathbb{E}(u \\mid \\text{büyüklük}) = 0$ koşulunu en çok şüpheli kılar?"
        ),
        answer=MultipleChoice(
            (
                "Aynı büyüklükteki konutların fiyatları birbirinden farklıdır",
                "Bazı konutların fiyatı tahmin edilen regresyon doğrusunun üzerinde, bazılarınınki altındadır",
                "Büyük konutlar sistematik olarak daha iyi semtlerdedir ve semt kalitesi modelde yoktur",
                "Fiyatı etkileyen ama büyüklükle ilişkisiz bir faktör (ör. satış günü) modelde yoktur",
            ),
            correct=2,
        ),
        explanation=(
            "Koşul, hata teriminin her büyüklük düzeyindeki ortalamasının sıfır olmasını ister. Fiyatı artıran semt "
            "kalitesi hata teriminde kalır ve büyüklükle sistematik biçimde ilişkiliyse u'nun ortalaması büyüklükle "
            "birlikte artar; eğim yalnız büyüklüğü yansıtmaz. Büyüklükle ilişkisiz bir dışlanmış faktör (satış günü) "
            "u'yu değiştirir ama her büyüklük düzeyindeki ortalamasını değiştirmez. Aynı büyüklükte farklı fiyatlar ve "
            "regresyon doğrusunun iki yanına düşen gözlemler de koşulla çelişmez (§3.6)."
        ),
    ),
    Question(
        key="k04", concept="egim-payinda-negatif-katki", note=_note("3.8", "(3.8)"),
        prompt=(
            "Eğim formülünün (3.8) payında bir gözlemin katkısı $(X_i - \\bar{X})(Y_i - \\bar{Y})$ negatiftir. Bu "
            "gözlem için hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Gözlem X'te ortalamanın üzerinde, Y'de ortalamanın altındadır ya da tam tersi",
                "X ve Y değerlerinin ikisi de kendi örneklem ortalamalarının altındadır",
                "Bu gözlem yüzünden tahmin edilen eğim, öteki gözlemler ne olursa olsun negatif çıkacaktır",
                "Bu gözlemin artığı negatiftir; yani gözlem tahmin edilen doğrunun altındadır",
            ),
            correct=0,
        ),
        explanation=(
            "Çarpım, iki sapmanın işaretleri farklıyken negatiftir: gözlem X'te ortalamanın bir yanında, Y'de öbür "
            "yanındadır. İkisi de ortalamanın altındaysa çarpım pozitiftir. Böyle gözlemler negatif eğimi destekler; "
            "eğimin işaretini ise bütün gözlemlerin katkılarının toplamı belirler. Çarpımın işareti artığın işaretini "
            "söylemez; artık tahmin edilen doğruya göre tanımlanır (§3.8)."
        ),
    ),
    Question(
        key="k05", concept="veri-araligi-disinda-sabit", note=_note("3.11"),
        prompt=(
            "Alanı 60 ile 180 m² arasındaki dairelerden oluşan bir örneklemde $\\widehat{\\text{kira}} = 3{.}200 + "
            "45\\,\\text{alan}$ tahmin ediliyor (kira TL/ay, alan m²). Sabit terim için en uygun yorum hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Alanı sıfır olan bir dairenin aylık kirası gerçekte 3.200 TL'dir",
                "Regresyon doğrusunun konumunu belirler; 0 m² veri "
                "aralığı dışında kaldığından ekonomik yorumu zayıftır",
                "Anlamsız göründüğü için modelden çıkarılmalı ve regresyon doğrusu orijinden geçirilmelidir",
                "Alanı bir metrekare büyük dairelerin tahmini aylık "
                "kirasının ortalama ne kadar yüksek olduğunu gösterir",
            ),
            correct=1,
        ),
        explanation=(
            "Sabit, alan sıfırken modelin tahmin ettiği kiradır. Örneklemdeki en küçük daire 60 m² olduğundan bu nokta "
            "veri aralığının dışındadır; doğrusal ilişkiyi oraya taşımak ekonomik anlam taşımayabilir. Yine de sabit, "
            "regresyon doğrusunun örneklemdeki konumunu belirleyen gerekli bir bileşendir; anlamsız görünmesi eğimi ya "
            "da modeli geçersiz kılmaz. Metrekare başına tahmini kira farkı eğimdir: aylık 45 TL (§3.11)."
        ),
    ),
    Question(
        key="k06", concept="ayni-x-ayni-tahmin-farkli-artik", note=_note("3.12", "(3.11)", "Tablo 3.4"),
        prompt=(
            "WAGE1'de eğitimi 16 yıl olan iki çalışanın saatlik ücretleri "
            "8,75 ve 6,00 dolardır. Denklem (3.11)'e göre hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "İki çalışanın artıkları aynıdır; farklı olan tahmin edilen ücretleridir",
                "Ücreti düşük olan çalışanın tahmin edilen ücreti de daha düşüktür",
                "Tahmin edilen ücret, iki çalışanın gözlenen ücretlerinin ortalaması olan 7,375 dolardır",
                "Tahmini ücretleri aynıdır (≈ 7,76 dolar); artıkları 0,99 ve −1,76 dolardır",
            ),
            correct=3,
        ),
        explanation=(
            "Tahmin edilen değer yalnız açıklayıcı değişkene bağlıdır: 16 yıl için $\\widehat{Y} = -0{,}9049 + "
            "0{,}5414 \\times 16 \\approx 7{,}76$ dolar. Artık, gözlenen ücretten tahmin çıkarılarak bulunur: 8,75 − "
            "7,76 = 0,99 (gözlem doğrunun üzerinde) ve 6,00 − 7,76 = −1,76 (altında). Aynı eğitim düzeyindeki "
            "farklılık artıklarda kalır; regresyon doğrusu bu iki çalışanın "
            "ortalamasından geçmek zorunda değildir (§3.12, Tablo 3.4)."
        ),
    ),
    Question(
        key="k07", concept="nedensel-iddiayi-sorgulayan-madde", note=_note("3.13"),
        prompt=(
            "81 ilin yatay kesit verisiyle tahmin edilen basit bir regresyona dayanan bir raporda “Kişi başına "
            "kütüphane sayısını artırmak kitap okuma oranını yükseltir.” yazıyor. Okuma kontrol listesinin hangi "
            "sorusu bu cümleyi doğrudan sorgular?"
        ),
        answer=MultipleChoice(
            (
                "Gözlem birimi ve veri yapısı nedir?",
                "Eğim katsayısının işareti ve büyüklüğü nedir?",
                "Yorum ilişki düzeyinde mi, nedensel düzeyde mi?",
                "Sabit terimin ekonomik yorumu veri aralığında anlamlı mı?",
            ),
            correct=2,
        ),
        explanation=(
            "Cümle bir müdahalenin sonucunu anlatır (“artırmak … yükseltir”); bu nedensel bir iddiadır. İl verisinde "
            "kütüphane sayısı rastgele belirlenmez; gelir ya da eğitim düzeyi gibi faktörler hem kütüphane sayısıyla "
            "hem okuma oranıyla ilişkili olabilir (§3.11). Kontrol listesinin dokuzuncu sorusu yorumun hangi düzeyde "
            "yapıldığını denetler; gözlem birimi, katsayının işareti ve sabit terim soruları katsayıyı doğru okumayı "
            "sağlar ama nedensel iddiayı sınamaz (§3.13)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="yonu-soru-belirler", note=_note("3.2"),
        prompt=(
            "Bir regresyonda hangi değişkenin bağımlı, hangisinin açıklayıcı olacağına iki değişken arasındaki "
            "korelasyonun işaretine bakılarak karar verilir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Yönü ekonomik soru ve teori belirler. Korelasyon simetriktir ($r_{XY} = r_{YX}$); hangi değişkenin sonuç "
            "olduğunu söylemez. “Ücret eğitimle nasıl değişir?” sorusunda ücret Y, eğitim X'tir; değişkenleri ters "
            "çevirmek başka bir araştırma sorusu kurar (§3.2)."
        ),
    ),
    Question(
        key="d02", concept="duz-dogruda-sabit-egim", note=_note("3.3", "(3.2)"),
        prompt=(
            "$\\mathbb{E}(Y \\mid X) = \\beta_0 + \\beta_1 X$ biçimindeki (düz doğru) anakütle regresyon fonksiyonunda "
            "$\\mathbb{E}(Y \\mid X = 5) - \\mathbb{E}(Y \\mid X = 4)$ farkı, $\\mathbb{E}(Y \\mid X = 15) - "
            "\\mathbb{E}(Y \\mid X = 14)$ farkına eşittir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "(3.1)'deki düz doğruda, (3.2)'ye göre $\\mathbb{E}(Y \\mid X = x + 1) - \\mathbb{E}(Y \\mid X = x) = "
            "\\beta_1$ bütün x değerlerinde geçerlidir; iki fark da $\\beta_1$ kadardır. Karesel terimli model de "
            "parametrelere göre doğrusaldır (§3.4), ama onda eğim X'e göre "
            "değişir; bu biçimler ilerleyen bölümlerde ele alınır (§3.3)."
        ),
    ),
    Question(
        key="d03", concept="hata-teriminde-yaklasik-bicim", note=_note("3.5", "(3.3)"),
        prompt=(
            "Basit regresyon modelinde hata terimi $u_i$ yalnız modelde yer almayan faktörleri kapsar; doğrusal "
            "biçimin gerçek ilişkiye ancak yaklaşık uymasından doğan sapmalar $u_i$ içinde yer almaz."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Notlara göre $u_i$ modelde yer almayan faktörleri, bireysel farklılıkları, tam ölçülemeyen özellikleri, "
            "beklenmeyen şokları ve kullanılan doğrusal biçimin yaklaşık olmasını temsil edebilir. Model (3.3)'e göre "
            "$u_i = Y_i - \\beta_0 - \\beta_1 X_i$ olduğundan, doğrusal biçim gerçek ilişkiye ancak yaklaşık uyuyorsa "
            "aradaki fark da $u_i$'ye girer (§3.5)."
        ),
    ),
    Question(
        key="d04", concept="artik-toplami-kosulu-kanitlamaz", note=_note("3.6"),
        prompt=(
            "WAGE1 regresyonunda EKK artıklarının toplamının sıfır çıkması, $\\mathbb{E}(u \\mid \\text{eğitim}) = 0$ "
            "koşulunun bu veride sağlandığını gösterir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Sabit terimli EKK doğrusunda artıkların toplamı her veri setinde sıfırdır; bu, EKK'nin hesaplama "
            "yapısından doğan bir özelliktir (§3.9). Koşul ise gözlenemeyen hata terimi u hakkındadır ve her eğitim "
            "düzeyinde u'nun ortalamasının sıfır olmasını ister. Eğitimle ilişkili yetenek u içinde kalıyorsa "
            "artıkların toplamı yine sıfır çıkar ama koşul bozulur (§3.6)."
        ),
    ),
    Question(
        key="d05", concept="farkli-orneklem-farkli-tahmin", note=_note("3.7", "Tablo 3.1"),
        prompt=(
            "İki araştırmacı aynı anakütleden çekilmiş iki ayrı rastgele örneklemle aynı basit regresyonu tahmin "
            "ediyor; eğimi biri 0,48, öteki 0,61 buluyor. İkisi de hesabını doğru yapmış olabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$\\widehat{\\beta}_1$ tek bir örneklemden hesaplanan sayıdır; başka bir örneklem başka bir değer verir. "
            "Bu yüzden iki farklı tahmin hesap hatasının değil, örneklemden örnekleme değişimin sonucu olabilir. "
            "Anakütle eğimi $\\beta_1$ ise sabittir ve bilinmez; tahminin örneklemden örnekleme ne kadar değiştiği "
            "ileride ölçülecektir (§3.7, Tablo 3.1)."
        ),
    ),
    Question(
        key="d06", concept="noktalardan-gecmek-ekk-degil", note=_note("3.8", "(3.7)", "Tablo 3.3"),
        prompt=(
            "Tablo 3.2 verisinde sabiti 48, eğimi 3 olan doğru ($Y = 48 + 3X$) beş gözlemin üçünden tam olarak geçer; "
            "bu yüzden bu doğrunun kareli artıklar toplamı EKK doğrusununkinden (1,60) küçüktür."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Bu doğrunun artıkları 1, 0, −1, 0 ve 0'dır; kareli artıklar toplamı 2,00 olur, EKK doğrusununki ise "
            "1,60'tır (Tablo 3.3). EKK tek tek noktalardan geçmeye değil, (3.7)'deki kareli artıklar toplamını en "
            "küçük yapmaya bakar: EKK doğrusu hiçbir noktadan tam geçmediği hâlde toplamı daha küçüktür (§3.8)."
        ),
    ),
    Question(
        key="d07", concept="rastgele-atama-ve-nedensel-okuma", note=_note("3.11", "(3.12)"),
        prompt=(
            "JTRAIN2'de programa katılım rastgele atandığı için 1,794 bin dolarlık eğim, programın ortalama kazanç "
            "etkisinin tahmini olarak okunabilir; WAGE1'deki 0,541 dolarlık "
            "eğitim eğimi için aynı okuma bu modelden tek başına yapılamaz."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Rastgele atamada katılım, kazancı etkileyen öteki faktörlerden bağımsız belirlenir; bu faktörleri taşıyan "
            "hata terimi için sıfır koşullu ortalama koşulu tasarım gereği savunulabilir. WAGE1'de eğitim rastgele "
            "atanmamıştır; deneyim, meslek ya da yetenek gibi faktörler eğitimle ilişkili olabilir. Atamanın gerçekten "
            "uygulanması ve eksik sonuç verisi gibi koşullar JTRAIN2 yorumu için de geçerlidir (§3.11)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="kosullu-ortalamayi-hesaplama", note=_note("3.3", "(3.1)", "(3.2)"),
        prompt=(
            "Bir hane anakütlesinde aylık tüketimin (bin TL) gelire (bin TL) göre koşullu ortalaması "
            "$\\mathbb{E}(\\text{tüketim} \\mid \\text{gelir}) = 1{,}2 + 0{,}75\\,\\text{gelir}$ olsun. Geliri 8 bin "
            "TL olan hanelerin ortalama tüketimi **(1)** bin TL olur; geliri 9 bin TL olan hanelerin koşullu ortalama "
            "tüketimi, geliri 8 bin TL olanlarınkinden **(2)** bin TL fazladır."
        ),
        answer=FillBlanks((NumberBlank(7.2, 0.005, "7,2"), NumberBlank(0.75, 0.005, "0,75"))),
        explanation=(
            "(3.1) biçimindeki ARF'de koşullu ortalama, gelir yerine 8 yazılarak bulunur: 1,2 + 0,75 × 8 = 7,2. "
            "Gelirdeki bir birimlik fark, koşullu ortalama tüketimde eğim kadar, yani 0,75 bin TL'lik farkla "
            "ilişkilidir; (3.2)'deki gibi bu fark her gelir düzeyinde aynıdır. 7,2 tek bir hanenin tüketimi değil, "
            "geliri 8 bin TL olan hanelerin anakütle ortalamasıdır (§3.3)."
        ),
    ),
    Question(
        key="b02", concept="hata-terimi-sayisal", note=_note("3.5", "(3.4)"),
        prompt=(
            "Yalnız bu soru için anakütle regresyon fonksiyonunun bilindiğini varsayın: $\\mathbb{E}(\\text{ücret} "
            "\\mid \\text{eğitim} = 14) = 7{,}20$ dolar. Eğitimi 14 yıl olan bir çalışanın saatlik ücreti 6,50 "
            "dolardır; bu çalışanın hata terimi $u_i =$ **(1)** dolar olur. Eğitimi yine 14 yıl olan başka bir "
            "çalışanın ücreti 8,10 dolarsa onun hata terimi **(2)** dolardır."
        ),
        answer=FillBlanks((NumberBlank(-0.7, 0.005, "−0,70"), NumberBlank(0.9, 0.005, "0,90"))),
        explanation=(
            "(3.4)'e göre hata terimi, gerçekleşen sonuç ile aynı X değerindeki anakütle ortalaması arasındaki "
            "farktır: 6,50 − 7,20 = −0,70 ve 8,10 − 7,20 = 0,90. Aynı eğitim düzeyindeki iki çalışanın koşullu "
            "ortalaması aynıdır; farklı olan hata terimleridir. Gerçekte anakütle regresyon fonksiyonu bilinmediği "
            "için $u_i$ gözlenemez; veriden yalnız artık hesaplanır (§3.5, §3.7)."
        ),
    ),
    Question(
        key="b03", concept="tek-gozlem-degisince-ekk", note=_note("3.9", "Tablo 3.2", "(3.8)", "(3.9)"),
        prompt=(
            "Tablo 3.2'deki beş öğrenciden beşincisinin notu 78 yerine 88 olsaydı (öteki değerler aynı kalsaydı) eğim "
            "tahmini $\\widehat{\\beta}_1 =$ **(1)**, sabit tahmini $\\widehat{\\beta}_0 =$ **(2)** olurdu."
        ),
        answer=FillBlanks((NumberBlank(3.9, 0.005, "3,9"), NumberBlank(44.6, 0.005, "44,6"))),
        explanation=(
            "Yeni ortalama $\\bar{Y} = 340/5 = 68$ olur. $Y_i - \\bar{Y}$ sapmaları −13, −8, −3, 4 ve 20; $X_i - "
            "\\bar{X}$ sapmaları değişmez (−4, −2, 0, 2, 4). Çarpımların toplamı 52 + 16 + 0 + 8 + 80 = 156, payda "
            "yine 40'tır: $\\widehat{\\beta}_1 = 156/40 = 3{,}9$. Sabit $\\widehat{\\beta}_0 = 68 - 3{,}9 \\times 6 = "
            "44{,}6$. X değeri ortalamadan en uzak gözlemlerden birindeki 10 puanlık değişiklik, eğimi saat başına 1 "
            "puan (2,9'dan 3,9'a) artırmıştır (§3.9, (3.8)–(3.9))."
        ),
    ),
    Question(
        key="b04", concept="formulde-bagimli-ve-aciklayici", note=_note("3.10", "Kod 3.1"),
        prompt=(
            "CEOSAL1 verisinde CEO'nun 1990 maaşı `salary` (bin dolar), firmanın 1988–1990 ortalama özkaynak kârlılığı "
            "`roe` (yüzde) değişkeniyle ölçülür. “Özkaynak kârlılığı bir yüzde puan yüksek olan firmaların CEO'ları "
            "ortalama olarak ne kadar farklı maaş alır?” sorusu için Kod 3.1'deki gibi `smf.ols(\"… ~ …\", "
            "data=ceosal1)` formülünde `~` işaretinin soluna **(1)**, sağına **(2)** yazılır."
        ),
        answer=FillBlanks((TextBlank(("salary",), "salary"), TextBlank(("roe",), "roe"))),
        explanation=(
            "Statsmodels formülünde `~` işaretinin solunda bağımlı, sağında açıklayıcı değişken bulunur. Soru maaşın "
            "özkaynak kârlılığına göre nasıl değiştiğini sorduğu için formül `salary ~ roe` olur. Çıktıda `Intercept` "
            "sabit tahminini, `roe` satırındaki katsayı eğim tahminini verir; ters yazılan formül (`roe ~ salary`) "
            "başka bir soruyu cevaplar (§3.10, Kod 3.1)."
        ),
    ),
    Question(
        key="b05", concept="sifir-bir-degiskende-grup-ortalamalari", note=_note("3.11", "(3.12)"),
        prompt=(
            "Bir çevrim içi mağaza müşterilerini kura ile iki gruba ayırıyor ve yalnız birine indirim kuponu "
            "gönderiyor. Kupon almayan grupta ortalama sepet tutarı 240 TL, kupon alan grupta 265 TL'dir. "
            "$\\text{sepet}_i = \\beta_0 + \\beta_1\\,\\text{kupon}_i + u_i$ modeli (kupon alanlar için 1, almayanlar "
            "için 0) EKK ile tahmin edilirse $\\widehat{\\beta}_0 =$ "
            "**(1)** TL ve $\\widehat{\\beta}_1 =$ **(2)** TL bulunur."
        ),
        answer=FillBlanks((NumberBlank(240, 0.05, "240"), NumberBlank(25, 0.05, "25"))),
        explanation=(
            "Sıfır–bir açıklayıcı değişkenli modelde sabit tahmini kontrol grubunun (kupon = 0) örneklem ortalamasına, "
            "eğim tahmini iki grubun ortalama farkına eşittir: 265 − 240 = 25 TL. Kupon kura ile dağıtıldığı için bu "
            "fark, kuponun ortalama sepet tutarına etkisinin tahmini olarak okunabilir. JTRAIN2'deki 1,794 bin "
            "dolarlık eğim de aynı yapıdadır (§3.11, (3.12))."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="egim-korelasyon-ve-standart-sapmalar", note=_note("3.8", "(3.8)"),
        prompt=(
            "Basit regresyonun eğim tahminini, X ile Y arasındaki örneklem korelasyonu $r$ ve örneklem standart "
            "sapmaları $s_X$, $s_Y$ cinsinden yazın. (İpucu: (3.8)'in payını ve paydasını $n - 1$'e bölün; "
            "korelasyonun tanımı §0.5'tedir.)"
        ),
        answer=Equation(
            lhs="\\widehat{\\beta}_1",
            symbols=(
                Symbol("r", "r", "örneklem korelasyonu", -0.9, 0.9, aliases=("r_XY", "r_xy", "r_YX", "r_yx")),
                Symbol("sx", "s_X", "X'in örneklem standart sapması", 0.5, 4, aliases=("s_X", "s_x", "S_X", "S_x")),
                Symbol("sy", "s_Y", "Y'nin örneklem standart sapması", 0.5, 4, aliases=("s_Y", "s_y", "S_Y", "S_y")),
            ),
            answer="r*sy/sx",
            shown="r\\,\\frac{s_Y}{s_X}",
        ),
        explanation=(
            "Pay ve payda $n - 1$'e bölününce eğim $s_{XY}/s_X^2$ olur. Korelasyon $r = s_{XY}/(s_X s_Y)$ olduğundan "
            "$s_{XY} = r\\,s_X s_Y$ ve $\\widehat{\\beta}_1 = r\\,s_Y/s_X$ bulunur. Eğimin işareti korelasyonun "
            "işaretiyle aynıdır; büyüklüğü ise iki değişkenin ölçü birimlerine bağlıdır. WAGE1'de $0{,}4059 \\times "
            "3{,}693/2{,}769 \\approx 0{,}541$ (§3.8, (3.8))."
        ),
    ),
    Question(
        key="e02", concept="sabit-tahmini-toplamlarla", note=_note("3.9", "(3.8)", "(3.9)"),
        prompt=(
            "Bir örneklemde $S_{XY} = \\sum (X_i - \\bar{X})(Y_i - \\bar{Y})$ ve $S_{XX} = \\sum (X_i - \\bar{X})^2$ "
            "olsun; örneklem ortalamaları $\\bar{X}$ ve $\\bar{Y}$ ile "
            "gösterilsin. Sabit tahminini bu dört büyüklük cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat{\\beta}_0",
            symbols=(
                Symbol("sxy", "S_{XY}", "çapraz çarpımlar toplamı", 5, 200, aliases=_SXY),
                Symbol("sxx", "S_{XX}", "X'in kareli sapmaları toplamı", 5, 100, aliases=_SXX),
                Symbol("xbar", "\\bar{X}", "X'in örneklem ortalaması", 1, 20, aliases=bar_aliases("X")),
                Symbol("ybar", "\\bar{Y}", "Y'nin örneklem ortalaması", 10, 90, aliases=bar_aliases("Y")),
            ),
            answer="ybar - sxy/sxx*xbar",
            shown="\\bar{Y} - \\frac{S_{XY}}{S_{XX}}\\,\\bar{X}",
        ),
        explanation=(
            "Önce (3.8)'e göre eğim $\\widehat{\\beta}_1 = S_{XY}/S_{XX}$, sonra (3.9)'a göre sabit "
            "$\\widehat{\\beta}_0 = \\bar{Y} - \\widehat{\\beta}_1 \\bar{X}$ bulunur. Tablo 3.2'de $S_{XY} = 116$, "
            "$S_{XX} = 40$, $\\bar{X} = 6$ ve $\\bar{Y} = 66$: $66 - 2{,}9 \\times 6 = 48{,}6$. Sabit formülü, tahmin "
            "edilen doğrunun $(\\bar{X}, \\bar{Y})$ noktasından geçmesini sağlar (§3.9, (3.8)–(3.9))."
        ),
    ),
    Question(
        key="e03", concept="ters-regresyonun-egimi", note=_note("3.2", "(3.8)"),
        prompt=(
            "WAGE1'de değişkenlerin yeri değiştirilip $\\text{eğitim}_i = \\gamma_0 + \\gamma_1\\,\\text{ücret}_i + "
            "v_i$ modeli (`educ ~ wage`) tahmin ediliyor. X eğitim, Y ücret olmak üzere $S_{XY} = \\sum (X_i - "
            "\\bar{X})(Y_i - \\bar{Y})$, $S_{XX} = \\sum (X_i - \\bar{X})^2$ ve $S_{YY} = \\sum (Y_i - \\bar{Y})^2$ "
            "olsun. Bu yeni regresyonun eğim tahminini yazın."
        ),
        answer=Equation(
            lhs="\\widehat{\\gamma}_1",
            symbols=(
                Symbol("sxy", "S_{XY}", "çapraz çarpımlar toplamı", 5, 200, aliases=_SXY),
                Symbol("sxx", "S_{XX}", "eğitimin kareli sapmaları toplamı", 5, 200, aliases=_SXX),
                Symbol("syy", "S_{YY}", "ücretin kareli sapmaları toplamı", 5, 200, aliases=_SYY),
            ),
            answer="sxy/syy",
            shown="\\frac{S_{XY}}{S_{YY}}",
        ),
        explanation=(
            "(3.8)'de payda, açıklayıcı değişkenin kareli sapmaları toplamıdır. Roller değişince açıklayıcı değişken "
            "ücret olur ve payda ücretin kareli sapmaları toplamı olur: $\\widehat{\\gamma}_1 = S_{XY}/S_{YY}$. Bu "
            "sayı $1/\\widehat{\\beta}_1 = S_{XX}/S_{XY}$ değerine yalnız bütün noktalar tek bir doğru üzerindeyse "
            "eşittir. WAGE1'de $\\widehat{\\gamma}_1 \\approx 0{,}304$: saatlik ücretteki bir dolarlık fark başına "
            "0,304 yıl; $1/\\widehat{\\beta}_1$ ise yaklaşık 1,847'dir. Ters regresyon “Saatlik ücreti bir dolar "
            "yüksek olanların ortalama eğitimi ne kadar farklı?” sorusunu cevaplar (§3.2, (3.8))."
        ),
    ),
    Question(
        key="e04", concept="hata-teriminin-kosullu-ortalamasi", note=_note("3.6", "(3.5)"),
        prompt=(
            "Y'nin anakütledeki koşullu ortalaması $X = x$ iken $m$ olsun: $m = \\mathbb{E}(Y \\mid X = x)$. Model $Y "
            "= \\beta_0 + \\beta_1 X + u$ biçiminde yazılıyor. Hata teriminin $X = x$'teki koşullu ortalamasını "
            "($\\mathbb{E}(u \\mid X = x)$) $m$, $\\beta_0$, $\\beta_1$ ve $x$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\mathbb{E}(u \\mid X = x)",
            symbols=(
                Symbol("m", "m", "Y'nin X = x'teki koşullu ortalaması", 1, 20),
                Symbol("b0", "\\beta_0", "sabit parametre", -5, 5, aliases=beta_aliases(0)),
                Symbol("b1", "\\beta_1", "eğim parametresi", 0.1, 2, aliases=beta_aliases(1)),
                Symbol("x", "x", "açıklayıcı değişkenin değeri", 0, 20),
            ),
            answer="m - b0 - b1*x",
            shown="m - \\beta_0 - \\beta_1\\,x",
        ),
        explanation=(
            "Modelden $u = Y - \\beta_0 - \\beta_1 X$; koşullu beklenen değer alınınca $\\mathbb{E}(u \\mid X = x) = "
            "\\mathbb{E}(Y \\mid X = x) - \\beta_0 - \\beta_1 x = m - \\beta_0 - \\beta_1 x$. (3.5)'teki koşul her "
            "x'te bu farkın sıfır olmasını, yani doğrunun gerçekten koşullu ortalamayı temsil etmesini ister. "
            "$\\beta_0 + \\beta_1 x$ doğrusu bazı x değerlerinde koşullu ortalamadan ayrılıyorsa (ör. eğitimle "
            "ilişkili yetenek u içinde kalıyorsa) orada $\\mathbb{E}(u \\mid X = x) \\neq 0$ olur (§3.6)."
        ),
    ),
    Question(
        key="e05", concept="artik-ile-hata-teriminin-farki", note=_note("3.7", "Tablo 3.1"),
        prompt=(
            "Anakütle modeli $Y_i = \\beta_0 + \\beta_1 X_i + u_i$, tahmin edilen doğru $\\widehat{Y}_i = "
            "\\widehat{\\beta}_0 + \\widehat{\\beta}_1 X_i$ olsun. Artık ile hata terimi arasındaki farkı "
            "($\\widehat{u}_i - u_i$) parametreler, tahminler ve $X_i$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat{u}_i - u_i",
            symbols=(
                Symbol("b0", "\\beta_0", "anakütle sabiti", -5, 5, aliases=beta_aliases(0)),
                Symbol("b1", "\\beta_1", "anakütle eğimi", 0.1, 2, aliases=beta_aliases(1)),
                Symbol("bh0", "\\widehat{\\beta}_0", "sabit tahmini", -5, 5, aliases=beta_hat_aliases(0)),
                Symbol("bh1", "\\widehat{\\beta}_1", "eğim tahmini", 0.1, 2, aliases=beta_hat_aliases(1)),
                Symbol("x", "X_i", "gözlemin X değeri", 0, 20, aliases=("X_i", "x_i", "Xi", "xi")),
            ),
            answer="(b0 - bh0) + (b1 - bh1)*x",
            shown="(\\beta_0 - \\widehat{\\beta}_0) + (\\beta_1 - \\widehat{\\beta}_1)\\,X_i",
        ),
        explanation=(
            "$\\widehat{u}_i = Y_i - \\widehat{\\beta}_0 - \\widehat{\\beta}_1 X_i$ ve $u_i = Y_i - \\beta_0 - "
            "\\beta_1 X_i$ olduğundan $Y_i$ farkta birbirini götürür. Artık, hata teriminden tahmin edilen doğrunun "
            "anakütle doğrusundan sapması kadar ayrılır; tahminler parametrelere eşit olsaydı ikisi aynı olurdu. "
            "Tahminler örneklemden örnekleme değiştiği için bu fark da "
            "değişir; Tablo 3.1'deki ayrım bu yüzden önemlidir (§3.7)."
        ),
    ),
)


KONU03_QUIZ = QuestionSet(
    topic_key="konu03",
    title="Konu 3: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"3.{number}" for number in range(1, 14)),  # §3.14 bölüm özetidir
)
