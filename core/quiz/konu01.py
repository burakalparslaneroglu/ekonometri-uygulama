"""Konu 1 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 1.1–1.7 ve Mini Quiz
1.1–1.7 maddelerini tekrar etmez: işsizlik grafiği, ücret denklemi ve konut istatistiklerinin sınıflandırılması;
reklam, sınıf büyüklüğü, faiz ve iş başında eğitim sorularının yeniden yazımı; konut fiyatı modelinin bileşenleri;
çalışma süresi ve not araştırmasının aşamaları; kira, metro hattı ve devamsızlık sorularının türü; WAGE1 çıktısının
yedi maddelik yorumu; beş yanlış yorumun düzeltilmesi ve mini quizlerdeki üç bileşen, hata teriminin gözlenebilirliği,
coef sütunu, standart hataların sunumu, R²'nin nedensellik ölçüsü olmaması, istatistiksel ve ekonomik anlamlılık
ayrımı. Aynı becerileri yeni bağlamlarla ve yeni sayılarla sınar. Standart hata, t ve p-değeri Konu 7'nin konusudur;
sorular yalnız notların Bölüm 1'de verdiği düzeyde onlara değinir.
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
    TextBlank,
    TrueFalse,
)


def _note(section: str, *objects: str) -> NoteRef:
    return NoteRef(section, 0, tuple(objects))


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="matematiksel-iktisadin-ciktisi", note=_note("1.1", "Tablo 1.1"),
        prompt=(
            "Bir araştırmacı, emek talebinin reel ücret yükseldikçe azaldığına ilişkin bilinen teorik fikri firmanın "
            "kâr maksimizasyonu problemi olarak yazıyor ve emek talebi denklemini türetiyor; yeni bir mekanizma "
            "önermiyor, veri de kullanmıyor. Tablo 1.1'e göre bu çalışma en çok hangi alanın tipik çıktısıdır?"
        ),
        answer=MultipleChoice(("İktisat teorisi", "Matematiksel iktisat", "Ekonomik istatistik", "Ekonometri"),
                              correct=1),
        explanation=(
            "İktisat teorisi ilişkinin neden ve hangi yönde olabileceğini açıklar; burada fikir zaten verilidir. "
            "Çalışma onu optimizasyon modeli ve denklemle ifade eder: bu matematiksel iktisadın tipik çıktısıdır. "
            "Veriye bağlanmadığı için ekonometri, ölçüm ve özet üretmediği için ekonomik istatistik değildir "
            "(§1.1, Tablo 1.1)."
        ),
    ),
    Question(
        key="k02", concept="olculebilir-sorunun-bilesenleri", note=_note("1.2"),
        prompt="Aşağıdakilerden hangisi ölçülebilir bir araştırma sorusunun belirtmesi gereken unsurlardan biri değildir?",
        answer=MultipleChoice(
            (
                "İncelenen birim",
                "Temel açıklayıcı değişken",
                "Analizde kullanılacak yazılım",
                "Amaç: betimleme, ilişki, tahmin ya da nedensel etki",
            ),
            correct=2,
        ),
        explanation=(
            "Ölçülebilir soru; birimi, sonuç değişkenini, temel açıklayıcı değişkeni, anakütle–yer–dönemi ve amacı "
            "belirtir. Yazılım bir araçtır; sorunun içeriğini ve yorumun kapsamını belirlemez (§1.1–1.2)."
        ),
    ),
    Question(
        key="k03", concept="i-alt-indisinin-anlami", note=_note("1.3", "(1.2)"),
        prompt="ücretᵢ = β₀ + β₁eğitimᵢ + uᵢ modelinde i alt indisi neyi gösterir?",
        answer=MultipleChoice(
            (
                "Örneklemdeki i. çalışanı",
                "Modeldeki i. parametreyi",
                "i. yılı",
                "Eğitimin i. düzeyini",
            ),
            correct=0,
        ),
        explanation=(
            "i örneklemdeki gözlemleri (burada çalışanları) numaralandırır; her çalışanın kendi ücreti, eğitimi ve "
            "hata terimi uᵢ vardır. Parametreler β₀ ve β₁ bütün çalışanlar için aynıdır (§1.3, (1.2))."
        ),
    ),
    Question(
        key="k04", concept="modeli-ve-sonucu-sorgulama-asamasi", note=_note("1.4", "Şekil 1.1"),
        prompt=(
            "Bir araştırmacı tahminden sonra aykırı gözlemleri çıkardığında ve farklı fonksiyonel biçimler "
            "denediğinde katsayının nasıl değiştiğine bakıyor. Bu iş sekiz aşamadan hangisine aittir?"
        ),
        answer=MultipleChoice(
            (
                "Ekonomik mekanizmayı kurma",
                "Anakütleyi, veriyi ve değişkenleri tanımlama",
                "Modeli tahmin etme",
                "Modeli ve sonucu sorgulama",
            ),
            correct=3,
        ),
        explanation=(
            "Varsayımlar, aykırı gözlemler, fonksiyonel biçim, alternatif modeller ve dayanıklılık yedinci aşamada "
            "değerlendirilir. Bu aşama modelin yeniden yazılmasına geri dönebilir: süreç geri bildirimlidir (§1.4)."
        ),
    ),
    Question(
        key="k05", concept="tahmin-sorusunu-tanima", note=_note("1.5", "Tablo 1.2"),
        prompt=(
            "Bir e-ticaret firması, geçmiş alışveriş verisini kullanarak önümüzdeki kampanya döneminde hangi "
            "müşterilerin en çok harcama yapacağını sıralamak ve stok planını buna göre yapmak istiyor. Bu soru "
            "hangi türdedir?"
        ),
        answer=MultipleChoice(("Betimleme", "İlişki", "Tahmin", "Nedensel etki"), correct=2),
        explanation=(
            "Amaç bilinen özelliklerle yeni bir sonucu, her müşterinin gelecekteki harcamasını öngörmektir. Soru "
            "kampanyanın harcamayı ne kadar artırdığını sormaz; bu nedensel etki sorusu olurdu. İyi bir tahmin modeli "
            "yeni gözlemlerde düşük tahmin hatası üretmelidir; katsayılarının nedensel yorumlanması bu amaç için "
            "zorunlu değildir (§1.5)."
        ),
    ),
    Question(
        key="k06", concept="serpilme-diyagramindaki-dogru", note=_note("1.6", "Şekil 1.2"),
        prompt="Şekil 1.2'de (WAGE1, eğitim ve saatlik ücret) noktaların arasından geçen doğru neyi özetler?",
        answer=MultipleChoice(
            (
                "Eğitime göre tahmin edilen ortalama ücretin doğrusal özetini",
                "Her çalışanın gerçekleşen ücretini",
                "Ek bir eğitim yılının ücrete nedensel etkisini",
                "Her eğitim düzeyindeki ortalama ücretleri birleştiren çizgiyi",
            ),
            correct=0,
        ),
        explanation=(
            "Grafikteki doğru, eğitim düzeylerine göre tahmin edilen ortalama ücretin doğrusal özetidir; tek tek "
            "çalışanlar doğrunun çevresine dağılır. Düz bir doğru olduğu için eğitim düzeylerinin ortalamalarından "
            "tek tek geçmek zorunda değildir. Eğimi bir örneklem ilişkisidir; tek başına nedensel etki olarak "
            "okunamaz (§1.6, Şekil 1.2)."
        ),
    ),
    Question(
        key="k07", concept="modelde-baska-neler-var", note=_note("1.7"),
        prompt="Okuma kontrol listesindeki “Modelde başka neler vardır?” sorusu neyi belirlemek için sorulur?",
        answer=MultipleChoice(
            (
                "Sonucun hangi gözlem birimi ve örneklem için elde edildiğini",
                "Katsayının işaretini, ölçü birimini ve ekonomik anlamını",
                "Katsayının belirsizliğinin ne kadar olduğunu",
                "Katsayının basit ilişkiyi mi, diğer değişkenler sabitken ilişkiyi mi gösterdiğini",
            ),
            correct=3,
        ),
        explanation=(
            "Beşinci soru modelin içeriğini belirler: aynı değişkenin katsayısı, modele başka açıklayıcı değişkenler "
            "eklendiğinde farklı bir soruyu cevaplar. Diğer seçenekler listenin başka maddeleridir: örneklem (2), "
            "büyüklük (6) ve belirsizlik (7) (§1.7)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="birlikte-hareket-yeterli-degil", note=_note("1.1"),
        prompt=(
            "Reklam harcaması ile satışların birlikte arttığını gösteren bir serpilme diyagramı çizen araştırmacı "
            "ekonometrik analizi tamamlamış olur."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Birlikte hareket başlangıçtır. Araştırmacı hareketin hangi soru için önemli olduğunu, verinin nasıl "
            "üretildiğini, modelin varsayımlarını ve sonucun hangi anakütle için geçerli olabileceğini tartışır (§1.1)."
        ),
    ),
    Question(
        key="d02", concept="iliski-duzeyindeki-soru", note=_note("1.2"),
        prompt=(
            "“Örneklemde eğitim yılı bir yıl daha yüksek olan çalışanların saatlik ücreti ortalama olarak ne kadar "
            "farklıdır?” sorusu, eğitimin ücreti artırdığını iddia eden nedensel bir sorudur."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Soru örneklemdeki ortalama ilişkiyi sorar. Eğitimdeki farkın ücret farkını oluşturup oluşturmadığı daha "
            "güçlü bir sorudur ve ek varsayımlar gerektirir (§1.2)."
        ),
    ),
    Question(
        key="d03", concept="hata-terimi-yanlis-hesap-degil", note=_note("1.3"),
        prompt="Bir ekonometrik modelde hata terimi bulunması, araştırmacının modeli yanlış kurduğunu göstermez.",
        answer=TrueFalse(True),
        explanation=(
            "Ekonomik sonuçları çok sayıda faktör etkiler; hepsini ölçmek ve modele ayrı ayrı eklemek çoğu zaman mümkün "
            "değildir. Hata terimi bu etkilerin bileşik temsilidir. Kritik soru, hata terimindeki faktörlerin "
            "açıklayıcı değişkenlerle nasıl ilişkili olduğudur (§1.3)."
        ),
    ),
    Question(
        key="d04", concept="orneklemin-kapsami", note=_note("1.4"),
        prompt=(
            "Bir konut kredisi araştırmasının verisi yalnız onaylanan başvuruları içeriyorsa, bu veriden elde edilen "
            "sonuç reddedilen başvurulara doğrudan genellenemez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Anakütle, veri ve değişkenler tanımlanırken örneklemin kimleri kapsadığı belirlenir. Yalnız onaylanan "
            "başvurular gözlendiğinde sonuç onlar için yorumlanır; bu karar katsayının yorumunu değiştirir (§1.4)."
        ),
    ),
    Question(
        key="d05", concept="nedensel-ama-dusuk-tahmin-gucu", note=_note("1.5"),
        prompt=(
            "Hanelerin elektrik tüketimini tahmin etmede tek başına zayıf kalan bir değişkenin (ör. evin yalıtımı) "
            "tüketime nedensel etkisi yine de büyük olabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Tahmin başarısı ile nedensel geçerlilik aynı şey değildir. Tüketimi hane büyüklüğü, iklim ve cihazlar gibi "
            "birçok faktör de etkiliyorsa, gerçek etkisi olan bir değişken tek başına haneler arasındaki tüketim "
            "farklarının küçük bir kısmını öngörebilir (§1.5)."
        ),
    ),
    Question(
        key="d06", concept="cikti-ve-makale-ayni-bilgi", note=_note("1.6", "Kod 1.4", "Tablo 1.3"),
        prompt=(
            "Kod 1.4'teki Python çıktısı ile Tablo 1.3'teki makale tipi tablo farklı temel sonuçlar sunar; biri "
            "diğerinin yerine okunamaz."
        ),
        answer=TrueFalse(False),
        explanation=(
            "İkisi aynı temel sonuçları farklı biçimlerde sunar: `coef` katsayı sütununa, `std err` parantez içindeki "
            "standart hataya, `No. Observations` gözlem sayısına, `R-squared` R² satırına karşılık gelir. Makale "
            "tablosu daha kısadır (§1.6, Kod 1.4, Tablo 1.3)."
        ),
    ),
    Question(
        key="d07", concept="yildiz-yok-iliski-yok-degil", note=_note("1.7"),
        prompt=(
            "Bir makale tablosunda reklam harcaması katsayısının yanında yıldız yoksa, reklam ile satış arasında hiçbir "
            "ilişki olmadığı sonucuna varılır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "İstatistiksel belirsizlik, örneklem büyüklüğü, değişkenlik ve model kurma kararları sonucu etkiler. "
            "“Kanıt bulunamadı” ile “etki kesinlikle sıfırdır” aynı değildir (§1.7)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="r2-yi-yuzde-olarak-yorumlama", note=_note("1.6"),
        prompt=(
            "Bir ücret regresyonunda R² = 0,284 bulunmuştur. Ücretteki örneklem değişkenliğinin yüzde **(1)** kadarı "
            "bu doğrusal modelde açıklanır, yüzde **(2)** kadarı açıklanmaz."
        ),
        answer=FillBlanks((NumberBlank(28.4, 0.05, "28,4"), NumberBlank(71.6, 0.05, "71,6"))),
        explanation=(
            "R² × 100 = %28,4 açıklanan paydır; kalan %71,6 modelde açıklanmayan değişkenliktir. WAGE1'de R² = 0,165 "
            "yaklaşık %16,5 demektir. R² nedensellik ya da ekonomik önem ölçüsü değildir (§1.6)."
        ),
    ),
    Question(
        key="b02", concept="farkin-ortalama-ucrete-orani", note=_note("1.6", "(1.4)"),
        prompt=(
            "WAGE1'de eğitim katsayısı 0,5414, ortalama saatlik ücret 5,896'dır. Bir yıllık eğitim farkına karşılık "
            "gelen tahmini ücret farkı ortalama ücretin yüzde **(1)** kadarıdır; dört yıllık fark için bu oran yüzde "
            "**(2)** olur. (Ara adımlarda yuvarlamadan, virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(9.18, 0.01, "9,18"), NumberBlank(36.73, 0.01, "36,73"))),
        explanation=(
            "0,5414/5,896 × 100 ≈ %9,18; dört yıl için 4 × 0,5414 = 2,1656 ve 2,1656/5,896 × 100 ≈ %36,73. §1.7 "
            "katsayının büyüklüğünü ölçü birimi ve ekonomik anlamla birlikte okumayı ister; farkı ortalama ücrete "
            "oranlamak bunun bir yoludur (yüzde hesabı: §0.6) (§1.6, (1.4))."
        ),
    ),
    Question(
        key="b03", concept="sistematik-ve-rastlantisal-bolum", note=_note("1.3"),
        prompt=(
            "ücretᵢ = β₀ + β₁eğitimᵢ + uᵢ ekonometrik modelinde β₀ + β₁eğitimᵢ kısmına **(1)** bölüm, uᵢ kısmına "
            "**(2)** bölüm denir."
        ),
        answer=FillBlanks((
            TextBlank(("sistematik", "sistematik bölüm", "deterministik", "deterministik bölüm"), "sistematik"),
            TextBlank(("rastlantısal", "rastlantısal bölüm", "rassal", "rassal bölüm", "rastgele", "rasgele",
                       "tesadüfi", "stokastik", "gözlenmeyen", "gözlenmeyen bölüm", "gözlenmeyen bileşen"),
                      "rastlantısal"),
        )),
        explanation=(
            "Model iki parçalıdır: açıklayıcı değişkenlere bağlanan sistematik bölüm ve gözlenmeyen etkileri temsil "
            "eden rastlantısal bölüm. Regresyon, sistematik bölümdeki bilinmeyen parametreleri tahmin eder (§1.3)."
        ),
    ),
    Question(
        key="b04", concept="betimleme-ve-nedensel-soru", note=_note("1.5", "Tablo 1.2"),
        prompt=(
            "Bir belediye otobüs verisiyle iki soru soruyor: “Hat başına günlük ortalama yolcu sayısı nedir?” "
            "sorusu **(1)** sorusudur; “Bilet fiyatı 5 TL düşürülürse günlük yolcu sayısı ne kadar değişir?” "
            "sorusu **(2)** sorusudur."
        ),
        answer=FillBlanks((
            TextBlank(("betimleme", "betimleyici", "betimsel", "betimleme sorusu", "betimleyici soru", "betimsel soru",
                       "tanımlayıcı", "tanımlayıcı soru"), "betimleme"),
            TextBlank(("nedensel etki", "nedensel", "nedensellik", "nedensel etki sorusu", "nedensellik sorusu",
                       "nedensel soru"), "nedensel etki"),
        )),
        explanation=(
            "İlk soru mevcut durumu özetler: betimleme. İkincisi bir müdahalenin sonucu ne kadar değiştireceğini "
            "sorar: nedensel etki. Nedensel soru daha güçlü araştırma tasarımı ve varsayımlar gerektirir (§1.5)."
        ),
    ),
    Question(
        key="b05", concept="sonuc-ve-aciklayici-degisken", note=_note("1.2"),
        prompt=(
            "“2025 yılında Türkiye'deki çağrı merkezi çalışanlarından haftalık uzaktan çalışma günü bir gün daha "
            "fazla olanlar, saat başına ortalama kaç görüşme daha fazla ya da daha az tamamlar?” sorusunda görüşme "
            "sayısı **(1)** değişkeni, uzaktan çalışma günü **(2)** değişkenidir."
        ),
        answer=FillBlanks((
            TextBlank(("sonuç", "sonuç değişkeni", "bağımlı", "bağımlı değişken", "açıklanan", "açıklanan değişken"),
                      "sonuç"),
            TextBlank(("açıklayıcı", "açıklayıcı değişken", "temel açıklayıcı", "temel açıklayıcı değişken",
                       "bağımsız", "bağımsız değişken"), "açıklayıcı"),
        )),
        explanation=(
            "Sonuç değişkeni açıklamak ya da tahmin etmek istediğimiz değişkendir; temel açıklayıcı değişken sonuçla "
            "ilişkisini incelediğimiz faktördür. Soru birimi (çalışan), anakütleyi ve yeri (Türkiye'deki çağrı "
            "merkezi çalışanları), dönemi (2025) de belirtir (§1.2)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="tahmin-hatasi", note=_note("1.5", "Tablo 1.2"),
        prompt=(
            "Tahmin hatası, gerçekleşen değerden tahmin edilen değer çıkarılarak tanımlanır. İki yeni gözlemde "
            "gerçekleşen değerler $y_1$ ve $y_2$, modelin tahminleri $p_1$ ve $p_2$ ise iki tahmin hatasının "
            "ortalamasını yazın."
        ),
        answer=Equation(
            lhs="\\text{ortalama tahmin hatası}",
            symbols=(
                Symbol("y1", "y_1", "1. gözlemin gerçekleşen değeri", 1, 20, aliases=("y_1",)),
                Symbol("y2", "y_2", "2. gözlemin gerçekleşen değeri", 1, 20, aliases=("y_2",)),
                Symbol("p1", "p_1", "1. gözlemin tahmini", 1, 20, aliases=("p_1",)),
                Symbol("p2", "p_2", "2. gözlemin tahmini", 1, 20, aliases=("p_2",)),
            ),
            answer="((y1 - p1) + (y2 - p2))/2",
            shown="\\frac{(y_1 - p_1) + (y_2 - p_2)}{2}",
        ),
        explanation=(
            "Her gözlemin tahmin hatası gerçekleşen değer eksi tahmindir; iki hatanın ortalaması toplamlarının "
            "yarısıdır (ortalama: §0.3). Pozitif ve negatif hatalar birbirini götürebilir (§0.4): ortalama hata "
            "tahminlerin sistematik olarak yukarı ya da aşağı sapıp sapmadığını gösterir, tek tek hataların "
            "büyüklüğünü göstermez. İyi bir tahmin modeli yeni gözlemlerde küçük tahmin hataları üretir; bu, "
            "katsayıların nedensel olduğunu göstermez (§1.5, Tablo 1.2)."
        ),
    ),
    Question(
        key="e02", concept="tahmin-edilen-ucret", note=_note("1.6", "(1.4)"),
        prompt=(
            "Tahmin edilen doğrunun sabiti $c$, eğimi $m$ ise eğitimi $x$ yıl olan bir çalışanın tahmin edilen "
            "saatlik ücretini yazın."
        ),
        answer=Equation(
            lhs="\\widehat{\\text{wage}}",
            symbols=(
                Symbol("c", "c", "sabit terim", -2, 2),
                Symbol("m", "m", "eğim", 0.1, 1),
                Symbol("x", "x", "eğitim yılı", 0, 18),
            ),
            answer="c + m*x",
            shown="c + m\\,x",
        ),
        explanation=(
            "Tahmin edilen değer sabit terim ile eğim × eğitim toplamıdır. Denklem (1.4)'e göre eğitimi 12 yıl olan "
            "bir çalışanın tahmin edilen ücreti −0,9049 + 0,5414 × 12 ≈ 5,59 ücret birimidir (§1.6, (1.4))."
        ),
    ),
    Question(
        key="e03", concept="egitim-farkina-karsilik-tahmini-fark", note=_note("1.6", "(1.4)"),
        prompt=(
            "Tahmin edilen doğrunun eğimi $m$'dir. Eğitimleri arasında $d$ yıl fark olan iki çalışandan eğitimi fazla "
            "olanın tahmin edilen saatlik ücretinden diğerininkini çıkararak farkı yazın."
        ),
        answer=Equation(
            lhs="\\Delta\\widehat{\\text{wage}}",
            symbols=(Symbol("m", "m", "eğim", 0.1, 1), Symbol("d", "d", "eğitim farkı (yıl)", 1, 6)),
            answer="m*d",
            shown="m\\,d",
        ),
        explanation=(
            "Sabit terim c iki tahminde de aynı olduğu için farkta birbirini götürür: eğitimler x₁ ve x₂ = x₁ + d ise "
            "(c + m x₂) − (c + m x₁) = m d. WAGE1'de 3 yıllık fark 3 × 0,5414 ≈ 1,62 ücret birimidir (§1.6)."
        ),
    ),
    Question(
        key="e04", concept="saatlik-farktan-aylik-farka", note=_note("1.7"),
        prompt=(
            "Saatlik ücret regresyonunda eğitim katsayısı $m$'dir (eğitim yılı başına TL/saat); çalışanlar ayda $h$ "
            "saat çalışmaktadır. Bir yıllık eğitim farkına karşılık gelen tahmini aylık ücret farkını yazın."
        ),
        answer=Equation(
            lhs="\\text{aylık fark}",
            symbols=(Symbol("m", "m", "eğitim katsayısı (TL/saat)", 5, 50), Symbol("h", "h", "aylık çalışma saati", 100, 200)),
            answer="m*h",
            shown="m\\,h",
        ),
        explanation=(
            "Katsayı, bir yıllık eğitim farkına karşılık gelen saatlik ücret farkıdır (TL/saat); ayda h saat "
            "çalışıldığında aylık fark m·h olur. Ör. m = 20 TL/saat ve h = 160 saat için 3.200 TL. Katsayının "
            "büyüklüğü ölçü birimiyle ve ekonomik anlamıyla birlikte okunur (§1.7)."
        ),
    ),
    Question(
        key="e05", concept="hata-teriminin-yazimi", note=_note("1.3", "(1.2)"),
        prompt=(
            "Ekonometrik model $y = a + b\\,x + u$ ise hata terimi $u$'yu $y$, $x$, $a$ ve $b$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="u",
            symbols=(
                Symbol("y", "y", "sonuç değişkeni", 1, 20),
                Symbol("x", "x", "açıklayıcı değişken", 0, 18),
                Symbol("a", "a", "sabit parametre", -2, 2),
                Symbol("b", "b", "eğim parametresi", 0.1, 1),
            ),
            answer="y - a - b*x",
            shown="y - a - b\\,x",
        ),
        explanation=(
            "u sonuç ile sistematik bölüm arasındaki farktır. Parametreler bilinmediği için u doğrudan gözlenmez; "
            "modelin sağ tarafında ayrı değişken olarak yer almayan faktörlerin bileşik etkisidir (§1.3)."
        ),
    ),
)


KONU01_QUIZ = QuestionSet(
    topic_key="konu01",
    title="Konu 1: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir "
        "bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki "
        "egzersiz ve mini quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"1.{number}" for number in range(1, 8)),  # §1.8 bölüm özetidir
)
