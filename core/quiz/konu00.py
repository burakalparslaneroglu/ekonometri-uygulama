"""Konu 0 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 0.1–0.9 ve Mini Quiz
0.1–0.9 maddelerini tekrar etmez: 120 firmalık veri setinin gözlem birimi ve kodları; fakültedeki 300 öğrencinin
anakütlesi; beş aylık satışların toplamı, ortalaması ve altıncı ay; iki sınıfın ortalaması ve yayılımı; fiyat–miktar,
eğitim–ücret ve sıcaklık–ısınma harcaması kovaryans işaretleri; 200'den 230'a yüzde değişim, işsizlikte %8'den %10'a
yüzde puan, satışların logaritması; koşullu ve koşulsuz ortalama sınıflandırması; parametre, tahmin edici ve tahmin
sınıflandırması; WAGE1 çıktısının anatomisi ve mini quizlerdeki kavram soruları (sütun, kimlik numarası, bölge kodu,
anakütle–örneklem farkı, büyük örneklem, açık toplam, ortalama formülü, sapmaların toplamı, varyans–standart sapma
ilişkisi, standart sapmanın birimi, pozitif kovaryans, korelasyonun üstünlüğü, sıfır korelasyon, %40'tan %45'e, ln(0),
log farkının iyi yaklaştığı durum, ȳ ile E(Y), E(Y | X = x), β₁ ile β̂₁, tahmin edici ile tahmin, H₀'ın sözlü
yorumu, coef sütunu, gözlem sayısı satırı, R² ve nedensellik). Aynı becerileri yeni bağlamlarla ve yeni sayılarla
sınar; notların metin içi örnekleri (ör. 14,4 yıllık ortalama, 415,09'luk kovaryans) soru olarak tekrarlanmaz, yalnız
açıklamalarda anılır. Standart hata, t ve p-değeri Bölüm 7'nin konusudur; sorular yalnız notların Bölüm 0'da verdiği
düzeyde onlara değinir.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol
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


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="gosterge-ortalamasi-pay", note=_note("0.1", "Tablo 0.1"),
        prompt=(
            "1.500 hanelik bir ankette `otomobil` değişkeni hanenin otomobili varsa 1, yoksa 0 değerini alıyor; "
            "değişkenin ortalaması 0,62'dir. Bu sayı neyi gösterir?"
        ),
        answer=MultipleChoice(
            (
                "Örneklemdeki hanelerin %62'sinin otomobili vardır.",
                "Hanelerin ortalama 0,62 otomobili vardır; otomobil sayısı bu değişkenden okunur.",
                "Değişken kategorik olduğu için ortalamanın hiçbir yorumu yoktur.",
                "Otomobili olan hanelerin %62'si ankete katılmıştır.",
            ),
            correct=0,
        ),
        explanation=(
            "0/1 gösterge değişkeninin ortalaması, 1 ile kodlanan grubun örneklemdeki payıdır: 1.500 hanenin 930'unun "
            "otomobili vardır. Değişken otomobil sayısını değil, sahip olup olmamayı ölçer; kodlama tersine çevrilse "
            "(otomobili yok = 1) ortalama 0,38 olurdu. Bölge kodu gibi keyfî numaraların ortalaması bir miktar "
            "bildirmez, ama 0/1 göstergenin ortalaması bir paydır (§0.1, Tablo 0.1)."
        ),
    ),
    Question(
        key="k02", concept="anakutle-tanimi-sinirlari", note=_note("0.2"),
        prompt="Bir araştırma için aşağıdaki anakütle tanımlarından hangisi yeterince açıktır?",
        answer=MultipleChoice(
            (
                "Bu dönemde Türkiye'deki üniversitelerde okuyan ve derslerine düzenli devam eden öğrencilerin çoğu",
                "İzmir'deki üniversitelerde son birkaç yılda herhangi bir programda eğitim görmüş gençlerin bir bölümü",
                "2026–2027 güz döneminde İzmir'deki devlet üniversitelerinin lisans programlarına kayıtlı öğrenciler",
                "Türkiye'de yaşayan ve ileride bir yükseköğretim kurumuna kaydolmayı düşünen bütün gençler",
            ),
            correct=2,
        ),
        explanation=(
            "Anakütlenin sınırları açık olmalıdır: hangi birimler, hangi yer ve hangi dönem. Yalnız üçüncü tanım "
            "kurumu, programı, öğrenci statüsünü, yeri ve dönemi belirtir; diğerlerinde “çoğu”, “bir bölümü”, “son "
            "birkaç yıl” ya da “düşünen” gibi belirsiz sınırlar vardır. Anakütle belirsizse teknik olarak kusursuz bir "
            "analiz bile araştırma sorusunu cevaplamaz (§0.2)."
        ),
    ),
    Question(
        key="k03", concept="toplamin-dogrusal-ozelligi", note=_note("0.3"),
        prompt=(
            "Toplam sembolüyle ilgili aşağıdaki eşitliklerden hangisi her veri seti için doğrudur? (Bütün toplamlar "
            "$i = 1, \\dots, n$ üzerindendir.)"
        ),
        answer=MultipleChoice(
            (
                "Σ xᵢyᵢ = (Σ xᵢ)·(Σ yᵢ)",
                "Σ (xᵢ + yᵢ) = Σ xᵢ + Σ yᵢ",
                "Σ xᵢ² = (Σ xᵢ)²",
                "Σ (xᵢ / yᵢ) = (Σ xᵢ) / (Σ yᵢ)",
            ),
            correct=1,
        ),
        explanation=(
            "Toplam sembolü terimleri tek tek toplamanın kısa yazımıdır: Σ(xᵢ + yᵢ) = (x₁ + y₁) + … + (xₙ + yₙ) = "
            "(x₁ + … + xₙ) + (y₁ + … + yₙ) = Σxᵢ + Σyᵢ. Çarpım, kare ve bölümde bu ayrıştırma genel olarak "
            "yapılamaz: ör. x = 1, 2 ve y = 3, 4 için Σxᵢyᵢ = 11, (Σxᵢ)(Σyᵢ) = 21 (§0.3)."
        ),
    ),
    Question(
        key="k04", concept="olcek-degisiminde-varyans", note=_note("0.4"),
        prompt=(
            "TL olarak ölçülen saatlik ücretler dolara çevriliyor: her değer 40'a bölünüyor (1 dolar = 40 TL). "
            "Ücretin örneklem varyansı nasıl değişir?"
        ),
        answer=MultipleChoice(
            ("Değişmez.", "40'a bölünür.", "40 katına çıkar.", "1.600'e bölünür."),
            correct=3,
        ),
        explanation=(
            "yᵢ = a·xᵢ ise s_y = |a|·s_x; varyans standart sapmanın karesi olduğu için a² ile çarpılır. a = 1/40 "
            "olduğundan varyans 40² = 1.600'e bölünür, standart sapma 40'a bölünür. Varyansın birimi özgün birimin "
            "karesidir (§0.4)."
        ),
    ),
    Question(
        key="k05", concept="korelasyonda-ortak-etken", note=_note("0.5"),
        prompt=(
            "İllerdeki itfaiye aracı sayısı ile yıllık yangın hasarı arasında güçlü pozitif korelasyon bulunuyor. En "
            "makul açıklama hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "İtfaiye araçları yangınlardaki hasarı artırmaktadır; araç sayısı azaltılırsa hasar da azalır.",
                "Büyük ve kalabalık illerde hem daha çok itfaiye aracı bulunur hem de yangın hasarı daha büyüktür.",
                "Korelasyon güçlü ve pozitif olduğuna göre iki değişken arasındaki nedensellik kanıtlanmıştır.",
                "Korelasyon ölçü birimine bağlı olduğu için hasarın TL ile ölçüldüğü bu sonuç anlamsızdır.",
            ),
            correct=1,
        ),
        explanation=(
            "Korelasyon nedensellik değildir. İlin büyüklüğü (nüfus, bina sayısı) iki değişkeni de artıran ortak bir "
            "etken olabilir; dondurma satışı ile boğulma vakalarını birlikte etkileyebilen sıcaklık gibi. "
            "Korelasyonun büyüklüğü ölçü biriminden bağımsızdır (§0.5)."
        ),
    ),
    Question(
        key="k06", concept="rassal-degiskenin-anlami", note=_note("0.7"),
        prompt="Bir değişkenin “rassal değişken” olarak ele alınması neyi ifade eder?",
        answer=MultipleChoice(
            (
                "Değişkenin değerinin hiçbir nedene bağlı olmadan, tamamen kendiliğinden oluştuğunu",
                "Gözlem gerçekleşmeden önce hangi değeri alacağının kesin olarak bilinmediğini",
                "Değişkenin bütün gözlemlerde aynı değeri aldığını ve bu değerin önceden bilindiğini",
                "Değişkenin doğrudan ölçülemeyen, yalnız tahmin edilebilen bir özellik olduğunu",
            ),
            correct=1,
        ),
        explanation=(
            "“Rassal”, değişkenin nedensiz olduğu anlamına gelmez; gözlem gerçekleşmeden önce değerin belirsiz "
            "olduğunu anlatır. Rassal değişkenin olasılık ağırlıklı ortalaması beklenen değer E(Y)'dir (§0.7)."
        ),
    ),
    Question(
        key="k07", concept="hipotez-testinin-niteligi", note=_note("0.8"),
        prompt="$H_0: \\beta_1 = 0$ hipotezinin sınanmasıyla ilgili hangisi doğrudur?",
        answer=MultipleChoice(
            (
                "Test, sıfır hipotezinin doğru ya da yanlış olduğunu matematiksel kesinlikle kanıtlar.",
                "Örneklemden hesaplanan β̂₁ sıfırdan farklıysa H₀ kesinlikle yanlıştır ve reddedilir.",
                "Test, örneklem kanıtının H₀ ile uyumunu değerlendirir; karar hata olasılığı taşır.",
                "Hipotezler tahminler hakkında kurulur: H₀'daki β₁, örneklemden hesaplanan sayının kendisidir.",
            ),
            correct=2,
        ),
        explanation=(
            "Hipotezler bilinmeyen parametreler (β₁) hakkındadır, örneklemden hesaplanan tahmin (β̂₁) hakkında değil. "
            "Tahmin örneklemden örnekleme değiştiği için β̂₁ ≠ 0 olması tek başına H₀'ı çürütmez. Test kanıtın H₀ ile "
            "uyumunu değerlendirir ve her karar hata olasılığı taşır; ayrıntılar Bölüm 7'dedir (§0.8)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="gozlem-birimi-donem", note=_note("0.1"),
        prompt=(
            "Bir makroekonomik çalışmanın veri tablosunda her satır bir yılı ya da bir çeyreği temsil edebilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Gözlemin neyi temsil ettiği araştırmanın birimine bağlıdır: ücret araştırmasında bir çalışan, firma "
            "araştırmasında bir şirket, makroekonomik çalışmada bir yıl ya da bir çeyrek (§0.1)."
        ),
    ),
    Question(
        key="d02", concept="ortalamanin-gozlenmesi-gerekmez", note=_note("0.3"),
        prompt="Dört günün ortalama sıcaklığı 17,5 derece ise günlerden en az birinde sıcaklık 17,5 derece olmalıdır.",
        answer=TrueFalse(False),
        explanation=(
            "Ortalama veride gözlenen bir değer olmak zorunda değildir: 12, 16, 19 ve 23 derecenin ortalaması "
            "70/4 = 17,5 derecedir, ama hiçbir günün sıcaklığı 17,5 derece değildir (§0.3)."
        ),
    ),
    Question(
        key="d03", concept="standart-sapma-negatif-olmaz", note=_note("0.4"),
        prompt=(
            "Gözlemlerin çoğu ortalamanın altındaysa standart sapma negatif çıkabilir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Varyans kareli sapmaların toplamından hesaplanır; kareler negatif olmadığı için varyans da negatif "
            "olamaz. "
            "Standart sapma varyansın pozitif kareköküdür. Sapmaların işareti, gözlemin ortalamanın üstünde mi altında "
            "mı olduğunu gösterir; yayılım ölçüsünün işaretini değil (§0.4)."
        ),
    ),
    Question(
        key="d04", concept="korelasyonun-isareti-ve-gucu", note=_note("0.5", "Şekil 0.2"),
        prompt="r = −0,9 olan bir ilişki, r = 0,5 olan bir ilişkiden daha zayıf bir doğrusal ilişkidir.",
        answer=TrueFalse(False),
        explanation=(
            "Korelasyonun işareti yönü, mutlak değeri doğrusal ilişkinin gücünü gösterir. |−0,9| = 0,9 > 0,5 olduğu "
            "için r = −0,9 daha güçlü bir doğrusal ilişkidir; yalnız yönü negatiftir (§0.5, Şekil 0.2)."
        ),
    ),
    Question(
        key="d05", concept="yuzde-degisimin-asimetrisi", note=_note("0.6"),
        prompt="Bir ürünün fiyatı önce %25 artıp ardından %20 düşerse başlangıç düzeyine döner.",
        answer=TrueFalse(True),
        explanation=(
            "100 TL %25 artışla 125 TL olur; 125 TL'nin %20'si 25 TL'dir, fiyat yeniden 100 TL'ye iner. Yüzde değişim "
            "başlangıç değerine göre hesaplandığı için aynı büyüklükteki artış ve azalış simetrik değildir (§0.6)."
        ),
    ),
    Question(
        key="d06", concept="kosullu-ortalama-bireysel-deger-degil", note=_note("0.7"),
        prompt=(
            "E(ücret | eğitim = 16) = 8 dolar ise anakütlede 16 yıl eğitimli her çalışanın saatlik ücreti 8 dolardır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Koşullu ortalama bir ortalamadır: 16 yıl eğitimli çalışanların saatlik ücretleri 8 dolar çevresinde "
            "dağılır; tek tek çalışanlar bundan farklı ücret alabilir. WAGE1'deki 16 yıl eğitimli 68 çalışanın "
            "ortalama saatlik ücreti 8,04 dolardır, ama tek tek ücretler 3,00 ile 22,86 dolar arasında değişir "
            "(§0.7)."
        ),
    ),
    Question(
        key="d07", concept="standart-hatanin-anlami", note=_note("0.9", "Kod 0.2"),
        prompt=(
            "Regresyon çıktısındaki `std err` sütunu, katsayı tahmininin örnekleme belirsizliğini özetler; bu "
            "belirsizliğin kaynağı, tahminin örneklemden örnekleme değişmesidir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Standart hata, katsayı tahmininin örnekleme belirsizliğini özetler: farklı örneklemler farklı β̂₁ "
            "değerleri verebilir (§0.8). Standart hatanın hesabı ve test ile güven aralığındaki kullanımı Bölüm 7'de "
            "açıklanır (§0.9, Kod 0.2)."
        ),
    ),
    # --- Boşluk doldurma -----------------------------------------------------------
    Question(
        key="b01", concept="toplam-kurallari-sayisal", note=_note("0.3"),
        prompt=(
            "$n = 8$ gözlemde $\\sum_{i=1}^{8} x_i = 40$'tır. Buna göre $\\sum_{i=1}^{8} 3x_i =$ **(1)** ve "
            "$\\sum_{i=1}^{8} (3x_i + 2) =$ **(2)**."
        ),
        answer=FillBlanks((NumberBlank(120, 0.5, "120"), NumberBlank(136, 0.5, "136"))),
        explanation=(
            "Σ3xᵢ = 3x₁ + … + 3x₈ = 3(x₁ + … + x₈) = 3·40 = 120: sabit çarpan toplamın dışına alınır. Sabit 2 sekiz "
            "gözlemin her birinde bir kez eklenir: Σ2 = 8·2 = 16; bu yüzden Σ(3xᵢ + 2) = 120 + 16 = 136 (§0.3)."
        ),
    ),
    Question(
        key="b02", concept="varyans-ve-standart-sapma-hesabi", note=_note("0.4"),
        prompt=(
            "Üç gözlem 4, 6 ve 8'dir. Örneklem varyansı $s^2 =$ **(1)**, örneklem standart sapması $s =$ **(2)**."
        ),
        answer=FillBlanks((NumberBlank(4, 0.005, "4"), NumberBlank(2, 0.005, "2"))),
        explanation=(
            "Ortalama 6; sapmalar −2, 0 ve 2, kareli sapmaların toplamı 8. Örneklem varyansı 8/(3 − 1) = 4, standart "
            "sapma √4 = 2. Paydada n değil n − 1 vardır (§0.4)."
        ),
    ),
    Question(
        key="b03", concept="log-farkinin-buyuk-degisimde-sapmasi", note=_note("0.6", "Tablo 0.3"),
        prompt=(
            "Bir değişken 80'den 100'e çıkıyor. Tam yüzde değişim **(1)**, log farkına dayalı yaklaşım "
            "$100\\,[\\ln(100) - \\ln(80)]$ ise **(2)** olur. (Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(25, 0.005, "25,00"), NumberBlank(22.31, 0.01, "22,31"))),
        explanation=(
            "Tam değişim 100·(100 − 80)/80 = 25,00; log farkı 100·ln(1,25) ≈ 22,31. %25 büyük bir değişimdir: "
            "yaklaşım tam değerden 2,69 yüzde puan küçüktür. Tablo 0.3'te olduğu gibi değişim büyüdükçe fark artar "
            "(§0.6)."
        ),
    ),
    Question(
        key="b04", concept="kovaryansin-birime-bagliligi", note=_note("0.5"),
        prompt=(
            "Bir örneklemde saatlik ücret (dolar) ile deneyim (yıl) arasındaki kovaryans 2,5'tir. Ücret sent olarak "
            "ölçülürse kovaryans **(1)** olur; ücret sent, deneyim ay olarak ölçülürse **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(250, 0.5, "250"), NumberBlank(3000, 0.5, "3.000"))),
        explanation=(
            "Kovaryans iki değişkenin ölçü birimlerine bağlıdır: ücret 100 ile çarpılınca kovaryans 100 katına çıkar "
            "(250); deneyim de 12 ile çarpılınca 250 × 12 = 3.000 olur. Korelasyon ise bu dönüşümlerde değişmez; "
            "WAGE1'de de ücret sente çevrilince kovaryans 4,1509'dan 415,09'a çıkar, korelasyon 0,406 kalır (§0.5)."
        ),
    ),
    Question(
        key="b05", concept="r-kare-korelasyonun-karesi", note=_note("0.9"),
        prompt=(
            "Sabitli basit regresyonda $R^2$, iki değişkenin korelasyonunun karesidir. Korelasyon 0,5 ise "
            "$R^2 =$ **(1)**; $R^2 = 0{,}36$ ve eğim pozitifse korelasyon **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(0.25, 0.005, "0,25"), NumberBlank(0.6, 0.005, "0,6"))),
        explanation=(
            "R² = r² = 0,5² = 0,25. Tersinden r² = 0,36 ise |r| = 0,6. Eğim (s_xy/s_x²) ile korelasyon "
            "(s_xy/(s_x·s_y)) aynı işareti, yani s_xy'nin işaretini taşır; eğim pozitifse r = 0,6'dır. WAGE1'de "
            "r = 0,406 ve R² = 0,165 (§0.9)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="dogrusal-donusumde-ortalama", note=_note("0.4"),
        prompt=(
            "Bir değişkenin örneklem ortalaması $\\bar{x}$ ile gösteriliyor. Her gözlem $y_i = a\\,x_i + c$ ile "
            "dönüştürülüyor. Yeni değişkenin ortalamasını $\\bar{x}$, $a$ ve $c$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\bar{y}",
            symbols=(
                Symbol("xbar", "\\bar{x}", "x'in ortalaması", 1, 20,
                       aliases=("x̄", "x̅", "X̄", "X̅", "\\barx", "\\bar x", "\\barX", "\\bar X", "\\overlinex",
                                "\\overline x", "\\overlineX", "\\overline X", "x_bar", "xort")),
                Symbol("a", "a", "ölçek", -5, 5),
                Symbol("c", "c", "kayma", -10, 10),
            ),
            answer="a*xbar + c",
            shown="a\\,\\bar{x} + c",
        ),
        explanation=(
            "Ortalama doğrusal dönüşümü izler: her değer a ile çarpılıp c eklenirse ortalama da a ile çarpılıp c "
            "kadar kayar. Ör. puanlar 2 ile çarpılıp 10 eklenirse ortalaması 70 olan bir sınıfın yeni ortalaması "
            "150 olur. Standart sapma ise yalnız |a| ile çarpılır (§0.4)."
        ),
    ),
    Question(
        key="e02", concept="korelasyon-formulu", note=_note("0.5"),
        prompt=(
            "İki değişkenin örneklem kovaryansı $s_{xy}$, standart sapmaları $s_x$ ve $s_y$'dir. Örneklem "
            "korelasyonunu "
            "yazın."
        ),
        answer=Equation(
            lhs="r_{xy}",
            symbols=(
                Symbol("sxy", "s_{xy}", "örneklem kovaryansı", -5, 5,
                       aliases=("s_xy", "S_xy", "s_yx", "S_yx", "syx", "Syx")),
                Symbol("sx", "s_x", "x'in standart sapması", 0.5, 4, aliases=("s_x", "S_x")),
                Symbol("sy", "s_y", "y'nin standart sapması", 0.5, 4, aliases=("s_y", "S_y")),
            ),
            answer="sxy/(sx*sy)",
            shown="\\frac{s_{xy}}{s_x\\,s_y}",
        ),
        explanation=(
            "Korelasyon kovaryansı iki standart sapmanın çarpımına bölerek standartlaştırır; sonuç ölçü biriminden "
            "bağımsızdır ve −1 ile 1 arasındadır. WAGE1'de 4,1509/(3,6931 × 2,7690) = 0,406 (§0.5)."
        ),
    ),
    Question(
        key="e03", concept="yuzde-degisim-formulu", note=_note("0.6"),
        prompt="Bir değişken $x_0$ değerinden $x_1$ değerine çıkıyor. Tam yüzde değişimi yazın.",
        answer=Equation(
            lhs="\\%\\Delta x",
            symbols=(
                Symbol("x0", "x_0", "başlangıç değeri", 1, 10, aliases=("x_0",)),
                Symbol("x1", "x_1", "yeni değer", 1, 10, aliases=("x_1",)),
            ),
            answer="100*(x1 - x0)/x0",
            shown="100\\,\\frac{x_1 - x_0}{x_0}",
        ),
        explanation=(
            "Yüzde değişim, değişimin başlangıç değerine oranının 100 katıdır. Payda başlangıç değeri olduğu için "
            "100'den 120'ye yüzde değişim %20, 120'den 100'e ise %−16,67'dir (§0.6)."
        ),
    ),
    Question(
        key="e04", concept="log-farki-yaklasimi", note=_note("0.6"),
        prompt=(
            "Bir değişken $x_0$ değerinden $x_1$ değerine çıkıyor. Doğal logaritma farkına dayalı yaklaşık yüzde "
            "değişimi yazın (doğal logaritma için `ln` ya da `log` yazabilirsiniz)."
        ),
        answer=Equation(
            lhs="\\%\\Delta x \\approx",
            symbols=(
                Symbol("x0", "x_0", "başlangıç değeri", 1, 10, aliases=("x_0",)),
                Symbol("x1", "x_1", "yeni değer", 1, 10, aliases=("x_1",)),
            ),
            answer="100*(log(x1) - log(x0))",
            shown="100\\,[\\ln(x_1) - \\ln(x_0)]",
        ),
        explanation=(
            "Küçük değişimlerde 100·[ln(x₁) − ln(x₀)] yaklaşık yüzde değişimi verir: 100'den 105'e 4,88 (tam %5). "
            "Logaritmanın özelliğiyle ln(x₁) − ln(x₀) = ln(x₁/x₀) olduğu için 100·ln(x₁/x₀) de aynı ifadedir (§0.6)."
        ),
    ),
    Question(
        key="e05", concept="egim-kovaryans-bolu-varyans", note=_note("0.9"),
        prompt=(
            "Sabitli basit regresyonda eğim tahmini, bağımlı değişken ile açıklayıcı değişkenin kovaryansının "
            "açıklayıcı değişkenin varyansına oranıdır. Kovaryans $s_{xy}$, açıklayıcı değişkenin standart sapması "
            "$s_x$ ise eğim tahminini yazın."
        ),
        answer=Equation(
            lhs="\\hat{\\beta}_1",
            symbols=(
                Symbol("sxy", "s_{xy}", "kovaryans", -5, 5, aliases=("s_xy", "S_xy", "s_yx", "S_yx", "syx", "Syx")),
                Symbol("sx", "s_x", "açıklayıcı değişkenin standart sapması", 0.5, 4, aliases=("s_x", "S_x")),
            ),
            answer="sxy/sx^2",
            shown="\\frac{s_{xy}}{s_x^2}",
        ),
        explanation=(
            "Varyans standart sapmanın karesidir; eğim s_xy/s_x². WAGE1'de eğitimin varyansı 7,6675'tir (standart "
            "sapma 2,7690'ın karesi; son basamaktaki fark yuvarlamadandır): 4,1509/7,6675 = 0,5414, yazılım "
            "çıktısındaki eğitim katsayısıyla aynıdır. Eğim formülü Bölüm 3'te türetilir (§0.9)."
        ),
    ),
)


KONU00_QUIZ = QuestionSet(
    topic_key="konu00",
    title="Konu 0: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir "
        "bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki "
        "egzersiz ve mini quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"0.{number}" for number in range(1, 10)),  # §0.10 bölüm özetidir
)
