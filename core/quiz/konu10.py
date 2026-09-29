"""Konu 10 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz ve Mini Quiz maddelerini
tekrar etmez: altı değişkenin nicel ya da kategorik olması; 68 + 7D sınav modeli ve ters kodlama; Tablo 10.2'nin ham
ve kontrollü fark maddeleri; 12,4 ve 4,0'lık kampanya kuklası; tarım referanslı 20 + 5D + 2D sektör modeli; okul türü
örneğinde kukla tuzağı; δ = 0,05, 0,20, −0,10 ve −0,40 için yüzde farklar; F(3, 240) = 4,10'luk bölge testi; Tablo
10.5'in yedi maddesi; 1,20 + 0,08X − 0,18D₁ + 0,12D₂ modeli ve mini quizlerdeki sorular. Aynı becerileri yeni
bağlamlarla ve yeni sayılarla sınar. Gruplar arasında eğim farkı (etkileşim) Konu 11'in, heteroskedastisiteye
dayanıklı çıkarım Konu 12'nin konusudur; sorular bunları kullanmaz.
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


_DELTA_HAT = ("\\hat\\delta", "\\widehat\\delta", "δ̂", "deltahat", "dhat", "\\delta", "δ", "delta")
"""Kukla katsayısı tahmini δ̂ yazımları (şapkasız δ de kabul edilir)."""
_SIGMA_HAT = ("\\hat\\sigma", "\\widehat\\sigma", "σ̂", "sigmahat", "\\sigma", "σ", "sigma")
_TKT = ("\\text{TKT}", "\\mathrm{TKT}", "\\operatorname{TKT}", "TKT", "SST")


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="sirali-kodda-esit-adim-varsayimi", note=_note("10.1"),
        prompt=(
            "Eğitim düzeyi ilkokul = 1, ortaokul = 2, lise = 3, üniversite = 4 diye kodlanıp tek bir sayısal açıklayıcı "
            "değişken olarak ücret modeline konuyor. Bu gösterim hangi varsayımı yapar?"
        ),
        answer=MultipleChoice(
            (
                "Eğitim düzeyleri arasında hiçbir sıralama olmadığını",
                "Her düzeyin ortalamasının ayrı ayrı ve serbestçe tahmin edildiğini",
                "Ardışık iki düzey arasındaki ücret farkının hep aynı olduğunu",
                "Kodların değerinin önemsiz olduğunu: 10, 20, 30, 40 kodları da aynı katsayıyı verir",
            ),
            correct=2,
        ),
        explanation=(
            "Tek sayısal değişken her kod artışına aynı katsayıyı verir: ilkokuldan ortaokula geçiş ile liseden "
            "üniversiteye geçiş aynı ücret farkı sayılır. Burada sıralama anlamlıdır, ama eşit uzaklık veriden değil "
            "kodlamadan gelir. Kodlar 10, 20, 30, 40 olsaydı katsayı onda birine inerdi; yorum yine eşit adıma dayanırdı. Üç "
            "kukla (bir düzey referans) her düzeyin farkını ayrı tahmin eder; Konu 10 Sezgi Deney 1 iki gösterimi "
            "karşılaştırır (§10.1)."
        ),
    ),
    Question(
        key="k02", concept="anlamlilik-ve-iktisadi-onem", note=_note("10.4"),
        prompt=(
            "İdari kayıtlardan alınan 200.000 kişilik bir veride $\\ln(\\text{ücret})$ modelinde “şehir merkezinde "
            "oturma” kuklasının katsayısı 0,004, standart hatası 0,001'dir (t = 4, p < 0,001). Hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "İstatistiksel olarak anlamlı, iktisadi olarak küçük bir fark: yaklaşık %0,4",
                "p < 0,001 olduğu için fark iktisadi bakımdan da büyük ve önemlidir",
                "Katsayı 0,004 gibi küçük bir sayı olduğu için fark istatistiksel olarak anlamsızdır",
                "t = 4 olduğundan şehir merkezinde oturmak ücreti yaklaşık %4 artırır",
            ),
            correct=0,
        ),
        explanation=(
            "Büyük örneklemde standart hata çok küçülür; küçük bir fark bile sıfırdan istatistiksel olarak ayrılabilir. "
            "0,004 log birim yaklaşık %0,4'lük ücret farkıdır: saatlik ücreti 100 TL olan biri için 40 kuruş. p-değeri "
            "farkın sıfır olup olmadığını sınar, büyüklüğünü ve önemini değil. t = 4, katsayının standart hatasının dört "
            "katı olduğunu söyler; yüzde etki değildir. Gözlemsel veride farkın nedeni de ayrıca değerlendirilir "
            "(§10.4)."
        ),
    ),
    Question(
        key="k03", concept="referans-disi-farkin-standart-hatasi", note=_note("10.5", "Tablo 10.3"),
        prompt=(
            "Kuzeydoğu referanslı bölge modelinde Batı katsayısının standart hatası 0,0598, Güney'inki 0,0505'tir. "
            "Batı ile Güney arasındaki farkın ($0{,}0331 - (-0{,}0829) = 0{,}1160$) standart hatası bu iki sayıdan "
            "bulunabilir mi?"
        ),
        answer=MultipleChoice(
            (
                "Evet: farkın standart hatası iki standart hatanın toplamıdır, 0,1103",
                "Evet: farkın standart hatası kareleri toplamının kareköküdür, yaklaşık 0,0782",
                "Hayır: iki tahmin birlikte hareket eder; referans Güney yapılıp Batı'nınki okunur: 0,0548",
                "Hayır: referans olmayan iki kategori arasındaki fark hiçbir modelde tahmin edilemez",
            ),
            correct=2,
        ),
        explanation=(
            "İki katsayı aynı referans grubuna (Kuzeydoğu) göre yazıldığı için tahminleri birlikte hareket eder "
            "(§8.1'deki ortak belirsizlik); bu modelde korelasyonları yaklaşık 0,52'dir. Farkın standart hatası bu ortak "
            "hareketi hesaba katmalıdır; kareler toplamının karekökü (0,0782) onu yok sayar ve belirsizliği abartır. En "
            "kolay yol referansı Güney yapmaktır: Batı'nın katsayısı 0,1160, standart hatası 0,0548 olur ve kareler "
            "toplamının karekökünden belirgin biçimde küçüktür. Aynı sonuç `f_test` ile eşitlik kısıtından da alınır "
            "(§10.5)."
        ),
    ),
    Question(
        key="k04", concept="iki-kuklanin-birlesik-tam-farki", note=_note("10.7"),
        prompt=(
            "Additif bir log ücret modelinde kadın kuklasının katsayısı −0,30, evli kuklasınınki 0,10'dur. Diğer "
            "değişkenler aynıyken evli bir kadının tahmin edilen ücreti bekâr bir erkeğinkinden tam hesapla yüzde kaç "
            "farklıdır?"
        ),
        answer=MultipleChoice(("%−25,92", "%−15,40", "%−20,00", "%−18,13"), correct=3),
        explanation=(
            "Log farkları toplanır: −0,30 + 0,10 = −0,20; tam fark $100(e^{-0{,}20} - 1) \\approx -18{,}13$. Tam "
            "yüzdeleri toplamak (−25,92 + 10,52 = −15,40) yanlıştır: yüzde değişimler çarpımsal birleşir, $0{,}7408 "
            "\\times 1{,}1052 \\approx 0{,}8187$. %−20,00 yaklaşık yorumdur; %−25,92 yalnız kadın farkıdır. Additif "
            "model evlilik farkını kadınlar ve erkekler için aynı kabul eder; bu farkın gruplara göre değişmesine izin "
            "veren etkileşim terimleri Konu 11'de ele alınır (§10.7)."
        ),
    ),
    Question(
        key="k05", concept="tek-tek-anlamsiz-birlikte-anlamli", note=_note("10.8"),
        prompt=(
            "Bir kategorik değişkenin üç kuklası için tek tek p-değerleri 0,08; 0,12 ve 0,15, üç kuklanın ortak F "
            "testinin p-değeri 0,01'dir. Hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Mümkün değildir: ortak test ancak bir kukla tek başına anlamlıysa reddeder",
                "Tutarlıdır: ortak test kanıtı birlikte değerlendirir; değişken bütünüyle anlamlıdır",
                "Ortak test hatalıdır; karar yalnız tek tek t testlerinin sonuçlarına göre verilmelidir",
                "Referans kategori değiştirilirse ortak test de kendiliğinden anlamsız hâle gelir",
            ),
            correct=1,
        ),
        explanation=(
            "Tek tek t testleri her kategorinin yalnız referansla farkını sorar; ortak F testi bütün kukla katsayılarının "
            "aynı anda sıfır olup olmadığını, yani bütün kategorilerin aynı düzeyde olup olmadığını sorar. Bu yüzden "
            "ortak test kategorilerin birbirleriyle farklarını da kullanır: bir kategori referansın biraz üstünde, öteki "
            "biraz altındaysa referansla farkların hiçbiri yüzde 5 eşiğini geçmeyebilir, ama iki kategori arasındaki fark "
            "büyüktür ve ortak test reddedebilir. Ortak test referans seçimine bağlı değildir (“Tek tek katsayılar ile "
            "ortak test farklı soruları yanıtlar” kutusu) (§10.8)."
        ),
    ),
    Question(
        key="k06", concept="ciktidan-referans-kategoriyi-bulma", note=_note("10.9", "Kod 10.5"),
        prompt=(
            "Bir öğrenci bölge modelini `C(region)` ile, referans belirtmeden tahmin ediyor. Çıktıda "
            "`C(region)[T.Güney]`, `C(region)[T.Kuzey Merkez]` ve `C(region)[T.Kuzeydoğu]` satırları görülüyor. "
            "Referans kategori hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Batı: çıktıda adı geçmeyen kategori",
                "Kuzeydoğu: listenin son satırındaki kategori",
                "Güney: listenin ilk satırındaki kategori",
                "Referans yoktur; bütün kategoriler çıktıda yer alır",
            ),
            correct=0,
        ),
        explanation=(
            "Sabitli modelde dört kategorinin üçü kuklayla girer; çıktıda görünmeyen kategori referanstır ve sabit "
            "terim onun düzeyini taşır. Referans belirtilmezse formül arayüzü kategorileri sıralayıp ilkini (burada "
            "alfabetik sırada Batı) referans alır; bu yüzden katsayılar notlardaki Kuzeydoğu referanslı Tablo 10.3'ten "
            "farklıdır. Referansı `Treatment(reference=...)` ile açıkça yazmak ve tabloda belirtmek gerekir (§10.9)."
        ),
    ),
    Question(
        key="k07", concept="ortak-destek-olmadan-kontrol", note=_note("10.3"),
        prompt=(
            "Bir firmada bütün kadın çalışanların kıdemi 0–5 yıl, bütün erkeklerinki 15–30 yıl arasındadır. "
            "`ücret ~ kadın + kıdem` modelindeki kadın katsayısı için hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Kıdem sabit tutulduğu için veri aynı kıdemdeki kadın–erkek karşılaştırmasını doğrudan sağlar",
                "Aynı kıdemde iki gruptan gözlem yoktur; karşılaştırma doğrusal kıdem teriminin uzatılmasına dayanır",
                "Kıdem kontrol edildiği için katsayı aynı kıdemdeki nedensel ücret farkıdır",
                "Kıdem ile kadın kuklası tam doğrusal bağlantılı olduğu için model tahmin edilemez",
            ),
            correct=1,
        ),
        explanation=(
            "Kontrollü katsayı “aynı kıdemdeki” kadın ve erkeği karşılaştırır; bu karşılaştırmanın veri içinde ortak "
            "desteği olmalıdır. Burada kıdem aralıkları örtüşmez: model kadınların ücretini 15–30 yıla, erkeklerinkini "
            "0–5 yıla doğrusal terimle uzatarak karşılaştırır ve sonuç bu biçim varsayımına dayanır. Kıdem kukladan tam "
            "olarak türetilemediği için tam bağlantı yoktur; model tahmin edilir ama yorumu zayıftır (§10.3)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="kukla-modeli-sayisal-kodu-kapsar", note=_note("10.1"),
        prompt=(
            "Üç kategorili bir değişken için (a) kodu 1, 2, 3 diye tek sayısal değişken olarak kullanan model ve (b) iki "
            "kuklalı sabitli model aynı veriyle tahmin ediliyor. (b)'nin $R^2$'si (a)'nınkinden küçük olamaz."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Kod = 1 + D₂ + 2D₃ olduğundan (a), (b)'nin kategori 3 farkını kategori 2 farkının iki katına eşitleyen "
            "kısıtlı hâlidir ($\\delta_3 = 2\\delta_2$). Kısıtsız model kısıtlı modelden daha kötü uyum veremez; gerçek "
            "farklar eşit adımlıysa iki $R^2$ birbirine yakın olur. Konu 10 Sezgi Deney 1'in varsayılan ayarında $R^2$ "
            "sayısal kodda 0,058, kukla modelinde 0,389'dur (§10.1)."
        ),
    ),
    Question(
        key="d02", concept="kontrol-kukla-katsayisini-buyutebilir", note=_note("10.3", "Tablo 10.2"),
        prompt=(
            "Kontrol değişkeni eklendiğinde kukla katsayısının mutlak değeri her zaman küçülür; Tablo 10.2'de "
            "−2,512'nin −1,811'e inmesi bu genel kuralın bir örneğidir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Ham fark, kontrollü fark ile kontrol değişkenindeki grup farkının katkısının toplamıdır. Katkı kontrollü "
            "farkla aynı işaretteyse ham fark mutlak değerce büyüktür (WAGE1'deki durum); ters işaretliyse ham fark "
            "mutlak değerce daha küçük olabilir ya da işaret değiştirebilir, bu durumda kontrol eklenince katsayı "
            "mutlak değerce büyüyebilir. Konu 10 Sezgi Deney 2'nin "
            "varsayılan ayarında (δ = −1, Δ = 2) ham farkın ortalaması yaklaşık 0,003, kontrollü farkınki yaklaşık "
            "−0,996'dır (§10.3)."
        ),
    ),
    Question(
        key="d03", concept="ters-kodlamada-t-isareti", note=_note("10.4", "Kod 10.4"),
        prompt=(
            "Kod 10.4'teki modelde `female` yerine `male` = 1 − `female` kullanılırsa `male` katsayısı 0,3011, "
            "standart hatası 0,0372 ve t istatistiği 8,085 olur; p-değeri değişmez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Kodlama ters çevrilince referans grup değişir: sabit terim erkeklerin düzeyinden kadınlarınkine iner "
            "(0,5013 − 0,3011 = 0,2002) ve kukla katsayısı işaret değiştirir. Aynı fark ölçüldüğü için standart hata "
            "aynıdır; t yalnız işaret değiştirir ve iki taraflı p-değeri aynı kalır. Tahmin edilen değerler, artıklar ve "
            "$R^2$ de değişmez (§10.4)."
        ),
    ),
    Question(
        key="d04", concept="otomatik-dusurme-referansi-gizler", note=_note("10.6"),
        prompt=(
            "Sabit terimli bir modele dört bölge kuklasının dördü de yazıldı ve yazılım kuklalardan birini kendiliğinden "
            "düşürdü. Bu durumda hangi bölgenin referans olduğunu ayrıca kontrol etmeye gerek yoktur."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Yazılım tam bağlantıyı gidermek için bir sütunu düşürebilir; ama hangi kategorinin düşeceği araştırmacının "
            "seçimi değildir ve katsayıların anlamı buna bağlıdır: her katsayı düşen kategoriye göre farktır. Referans "
            "bilinmeden katsayı yorumlanamaz; tabloda da açıkça yazılmalıdır (“Sık yapılan hata” kutusu) (§10.6)."
        ),
    ),
    Question(
        key="d05", concept="referans-degisince-tekil-p-degisebilir", note=_note("10.5", "Tablo 10.3"),
        prompt=(
            "Aynı bölge modelinde referans kategori değiştirildiğinde, bir bölge kuklasının p-değeri yüzde 5 düzeyinde "
            "anlamsızdan anlamlıya dönüşebilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Her kukla katsayısı referansa göre farktır; referans değişince katsayının karşılaştırdığı iki grup ve "
            "sınanan hipotez de değişir. Kuzeydoğu referansında Batı katsayısı 0,0331 ve p = 0,580'dir (Tablo 10.3); "
            "Güney referansında Batı katsayısı 0,1160 ve p ≈ 0,035 olur. Tahmin edilen değerler, $R^2$ ve bölge "
            "kuklalarının ortak F testi ise referanstan bağımsızdır (§10.5)."
        ),
    ),
    Question(
        key="d06", concept="sabit-etkiler-evet-anlamlilik-degil", note=_note("10.9"),
        prompt=(
            "Bir makale tablosundaki “Bölge sabit etkileri: Evet” satırı, bölge kuklalarının birlikte istatistiksel "
            "olarak anlamlı olduğunu gösterir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Bu satır yalnız bölge kuklalarının modele alındığını söyler; katsayıları, referans kategoriyi ve ortak "
            "anlamlılığı göstermez. Ortak önem için kuklaların ortak F testi ve p-değeri ayrıca raporlanmalıdır. "
            "WAGE1'de bölge kuklaları modelde yer alsa da ortak test yüzde 5'te reddedilemez: $F(3, 517) = 2{,}095$, "
            "p = 0,100 (§10.9)."
        ),
    ),
    Question(
        key="d07", concept="sifir-referans-kategoridir", note=_note("10.10"),
        prompt=(
            "Kuzeydoğu referanslı bölge modelinde üç bölge kuklasının da 0 olduğu bir çalışan hiçbir bölgede "
            "yaşamamaktadır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Her çalışan dört bölgeden birinde yaşar; üç kuklanın birden 0 olması çalışanın referans kategoride, yani "
            "Kuzeydoğu'da yaşadığını gösterir. Sıfır değeri çoğu zaman bir özelliğin yokluğu değil, referans kategorinin "
            "kodudur (Sık Yapılan Yorum Hataları, madde 1). Sabit terim bu çalışanlar için tahmin edilen düzeyi taşır "
            "(§10.10)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="genel-ortalamadan-grup-ortalamalari", note=_note("10.2"),
        prompt=(
            "Bir firmada çalışanların %40'ı uzaktan ($D = 1$), %60'ı ofiste ($D = 0$) çalışıyor. Bütün çalışanların "
            "ortalama verimlilik puanı 49'dur; verimlilik ~ D regresyonunda kukla katsayısı 5'tir. Ofiste çalışanların "
            "ortalama puanı **(1)**, uzaktan çalışanlarınki **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(47.0, 0.05, "47"), NumberBlank(52.0, 0.05, "52"))),
        explanation=(
            "Kukla regresyonunda $\\widehat\\beta_0 = \\bar Y_0$ ve $\\widehat\\delta = \\bar Y_1 - \\bar Y_0$'dır. "
            "Genel ortalama iki grup ortalamasının ağırlıklı ortalamasıdır: $\\bar Y = 0{,}6\\,\\bar Y_0 + 0{,}4\\,\\bar "
            "Y_1 = \\bar Y_0 + 0{,}4 \\times 5$. Buradan $\\bar Y_0 = 49 - 2 = 47$ ve $\\bar Y_1 = 47 + 5 = 52$. EKK "
            "doğrusu ortalamalardan geçtiği için $\\bar Y = \\widehat\\beta_0 + \\widehat\\delta\\,\\bar D$ her kukla "
            "regresyonunda geçerlidir (§10.2)."
        ),
    ),
    Question(
        key="b02", concept="ters-karsilastirmada-tam-yuzde-araligi", note=_note("10.7", "Kod 10.4"),
        prompt=(
            "Kod 10.4'teki kadın katsayısının yüzde 95 güven aralığı log ölçeğinde [−0,374; −0,228]'dir. Aynı modele "
            "göre erkeklerin kadınlara göre tam yüzde ücret farkının güven aralığı: alt sınır yüzde **(1)**, üst sınır "
            "yüzde **(2)**. (Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(25.61, 0.06, "25,61"), NumberBlank(45.35, 0.06, "45,35"))),
        explanation=(
            "Karşılaştırma yönü değişince log sınırları işaret değiştirir ve yer değiştirir: [0,228; 0,374]. Tam yüzde "
            "biçimi sınırlar ayrı ayrı dönüştürülerek bulunur: $100(e^{0{,}228} - 1) \\approx 25{,}61$ ve "
            "$100(e^{0{,}374} - 1) \\approx 45{,}35$ (yuvarlanmamış sınırlarla 45,40). Kadınların erkeklere göre aralığı "
            "[−31,22; −20,39] ile karşılaştırın: bilgi aynıdır, temel grup farklıdır; yüzde farkın büyüklüğü temel gruba "
            "bağlıdır (§10.7)."
        ),
    ),
    Question(
        key="b03", concept="referans-disi-iki-bolgenin-farki", note=_note("10.5", "Tablo 10.3"),
        prompt=(
            "Tablo 10.3'te Kuzey Merkez'in log katsayısı −0,071, Batı'nınki 0,033'tür (Kuzeydoğu referans). Diğer "
            "değişkenler sabitken Batı'nın Kuzey Merkez'e göre log farkı **(1)** (virgülden sonra üç basamak), tam yüzde "
            "farkı yaklaşık yüzde **(2)** olur (virgülden sonra iki basamak)."
        ),
        answer=FillBlanks((NumberBlank(0.104, 0.0011, "0,104"), NumberBlank(10.96, 0.06, "10,96"))),
        explanation=(
            "İki katsayı aynı referansa göre olduğundan Kuzeydoğu farkta düşer: 0,033 − (−0,071) = 0,104. Tam yüzde "
            "farkı $100(e^{0{,}104} - 1) \\approx 10{,}96$; yuvarlanmamış katsayılarla (0,1045) yaklaşık 11,01'dir. "
            "Referans Kuzey Merkez yapılsaydı Batı katsayısı doğrudan bu farkı verirdi; tahmin edilen değerler "
            "değişmezdi (§10.5)."
        ),
    ),
    Question(
        key="b04", concept="iki-kategorik-degiskende-kukla-sayisi", note=_note("10.6"),
        prompt=(
            "Sabit terimli bir modele 5 kategorili sektör değişkeni ve 3 kategorili eğitim düzeyi değişkeni birlikte "
            "kukla olarak girecek. Tam çoklu doğrusal bağlantı yaratmadan kullanılabilecek en çok kukla sayısı: "
            "**(1)**. Sabit terim çıkarılırsa bu sayı: **(2)**."
        ),
        answer=FillBlanks((NumberBlank(6.0, 0.5, "6"), NumberBlank(7.0, 0.5, "7"))),
        explanation=(
            "Sabitli modelde her kategorik değişken bir referans bırakır: (5 − 1) + (3 − 1) = 6. Sabit çıkarılınca "
            "yalnız bir değişkenin bütün kategorileri kullanılabilir (5 + 2 ya da 4 + 3 = 7): beş sektör kuklasının "
            "toplamı da üç eğitim kuklasının toplamı da her gözlemde 1'dir; ikisi birden tam alınırsa bu iki toplam "
            "birbirine eşit olur ve tam bağlantı doğar. İki gösterimde de düzeyleri belirleyen parametre sayısı yedidir "
            "(§10.6)."
        ),
    ),
    Question(
        key="b05", concept="ortak-testte-serbestlik-dereceleri", note=_note("10.8", "Tablo 10.4"),
        prompt=(
            "WAGE1 endüstri modelinde altı endüstri kuklasının ortak testi $F(6, 514) = 8{,}242$'dir (n = 526). "
            "Kısıtsız modelde sabit dışında **(1)** eğim vardır; endüstri kuklaları çıkarılmış kısıtlı modelde eğim "
            "sayısı **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(11.0, 0.5, "11"), NumberBlank(5.0, 0.5, "5"))),
        explanation=(
            "Payda serbestlik derecesi kısıtsız modelinkidir: n − k − 1 = 514, yani k = 526 − 514 − 1 = 11. Pay "
            "serbestlik derecesi kısıt sayısıdır: altı kuklanın katsayıları sıfırlanınca kısıtlı modelde 11 − 6 = 5 eğim "
            "kalır (eğitim, deneyim, deneyim², kıdem, kıdem²). Serbestlik dereceleri, çıktıda hangi modellerin "
            "karşılaştırıldığını kontrol etmenin kısa bir yoludur (§10.8)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="kukla-modelinde-r2", note=_note("10.2", "Kod 10.2"),
        prompt=(
            "Sabitli basit kukla modeli $Y = \\beta_0 + \\delta D + u$'da n gözlem vardır; $D = 1$ olanların oranı p, "
            "kukla katsayısının tahmini $\\widehat\\delta$ ve toplam kareler toplamı TKT'dir. $R^2$'yi n, p, $\\widehat\\delta$ "
            "ve TKT cinsinden yazın."
        ),
        answer=Equation(
            lhs="R^2",
            symbols=(
                Symbol("n", "n", "gözlem sayısı", 50, 500),
                Symbol("p", "p", "D = 1 olanların oranı", 0.1, 0.9),
                Symbol("d", "\\widehat\\delta", "kukla katsayısının tahmini", -3, 3, aliases=_DELTA_HAT),
                Symbol("tkt", "\\text{TKT}", "toplam kareler toplamı", 100, 5000, aliases=_TKT),
            ),
            answer="n*p*(1 - p)*d^2/tkt",
            shown="\\frac{n\\,p\\,(1 - p)\\,\\widehat\\delta^2}{\\text{TKT}}",
        ),
        explanation=(
            "Tahmin edilen değerler grup ortalamalarıdır: $\\bar Y_1 - \\bar Y = (1 - p)\\widehat\\delta$ ve $\\bar Y_0 - "
            "\\bar Y = -p\\widehat\\delta$. Böylece MKT $= np(1 - p)^2\\widehat\\delta^2 + n(1 - p)p^2\\widehat\\delta^2 = np(1 - "
            "p)\\widehat\\delta^2$ ve $R^2$ = MKT/TKT. Ücretteki değişkenliğin yalnız gruplar arası kısmı açıklanır. "
            "WAGE1'de $526 \\times 0{,}479 \\times 0{,}521 \\times 2{,}5118^2/7160{,}41 \\approx 0{,}116$ (Kod 10.2) "
            "(§10.2)."
        ),
    ),
    Question(
        key="e02", concept="ham-fark-kontrollu-fark-ve-grup-farki", note=_note("10.3", "Tablo 10.1", "Tablo 10.2"),
        prompt=(
            "Uzun model $y = \\beta_0 + \\delta d + b\\,x + u$ EKK ile tahmin ediliyor: kukla katsayısı $\\widehat\\delta$, "
            "x'in katsayısı $\\widehat b$'dir. İki grubun x ortalamaları farkı $\\Delta = \\bar x_1 - \\bar x_0$'dır. Aynı "
            "veriyle tahmin edilen kısa modelin ($y$ ~ $d$) kukla katsayısını $\\widehat\\delta$, $\\widehat b$ ve $\\Delta$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\tilde\\delta",
            symbols=(
                Symbol("d", "\\widehat\\delta", "uzun modeldeki kukla katsayısı", -3, 3, aliases=_DELTA_HAT),
                Symbol("b", "\\widehat b", "uzun modelde x'in katsayısı", -2, 2,
                       aliases=("\\hat b", "\\widehat b", "\\hatb", "\\widehatb", "b̂", "bhat")),
                Symbol("g", "\\Delta", "grupların x ortalamaları farkı", -4, 4, aliases=("\\Delta", "Δ", "Delta")),
            ),
            answer="d + b*g",
            shown="\\widehat\\delta + \\widehat b\\,\\Delta",
        ),
        explanation=(
            "Konu 6'daki eksik değişken ayrıştırması örneklemde tam geçerlidir: kısa katsayı = uzun katsayı + $\\widehat b$ "
            "× (x'in d'ye yardımcı regresyonundaki eğim). Yardımcı regresyon bir kukla regresyonudur; eğimi grup "
            "ortalamaları farkı $\\Delta$'dır. WAGE1 düzey modelinde (kontrollü kadın katsayısı −1,811; eğitim, deneyim "
            "ve kıdem katsayıları 0,572; 0,025; 0,141) Tablo 10.1'deki farklarla −1,811 + 0,572(−0,47) + 0,025(−1,13) + "
            "0,141(−2,85) ≈ −2,51: ham fark (§10.3)."
        ),
    ),
    Question(
        key="e03", concept="kukla-katsayisinin-standart-hatasi", note=_note("10.4", "Kod 10.2"),
        prompt=(
            "Sabitli basit kukla modelinde $D = 1$ grubunda $n_1$, $D = 0$ grubunda $n_0$ gözlem vardır ve regresyonun "
            "standart hatası $\\widehat\\sigma$'dır. Kukla katsayısının standart hatasını $\\widehat\\sigma$, $n_1$ ve $n_0$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\operatorname{se}(\\widehat\\delta)",
            symbols=(
                Symbol("s", "\\widehat\\sigma", "regresyonun standart hatası", 0.5, 5, aliases=_SIGMA_HAT),
                Symbol("n1", "n_1", "D = 1 grubundaki gözlem sayısı", 20, 300, aliases=("n_1",)),
                Symbol("n0", "n_0", "D = 0 grubundaki gözlem sayısı", 20, 300, aliases=("n_0",)),
            ),
            answer="s*sqrt(1/n1 + 1/n0)",
            shown="\\widehat\\sigma\\sqrt{\\frac{1}{n_1} + \\frac{1}{n_0}}",
        ),
        explanation=(
            "Konu 7'deki yapıya göre tek açıklayıcılı modelde $\\operatorname{se}(\\widehat\\delta) = \\widehat\\sigma/\\sqrt{"
            "\\sum(D_i - \\bar D)^2}$'dir. Kuklada $\\sum(D_i - \\bar D)^2 = n_1n_0/n$ olduğundan $\\operatorname{se} = "
            "\\widehat\\sigma\\sqrt{n/(n_1n_0)} = \\widehat\\sigma\\sqrt{1/n_1 + 1/n_0}$: küçük grup standart hatayı belirler. "
            "WAGE1'de $\\widehat\\sigma \\approx 3{,}476$, $n_1 = 252$, $n_0 = 274$: $3{,}476 \\times 0{,}0873 \\approx "
            "0{,}303$ (Kod 10.2'de 0,3034) (§10.4)."
        ),
    ),
    Question(
        key="e04", concept="tam-yuzde-guven-araligi-alt-siniri", note=_note("10.7", "Kod 10.4"),
        prompt=(
            "Log bağımlı değişkenli bir modelde kukla katsayısının tahmini $\\widehat\\delta$, standart hatası $s$ ve güven "
            "aralığının kritik değeri $c$'dir. Tam yüzde farkın güven aralığının alt sınırını yazın."
        ),
        answer=Equation(
            lhs="\\text{alt sınır (\\%)}",
            symbols=(
                Symbol("d", "\\widehat\\delta", "kukla katsayısının tahmini", -1, 1, aliases=_DELTA_HAT),
                Symbol("s", "s", "standart hata", 0.01, 0.2, aliases=("se", "SE", "sh", "SH")),
                Symbol("c", "c", "kritik değer", 1.6, 2.7),
            ),
            answer="100*(exp(d - c*s) - 1)",
            shown="100\\left(e^{\\widehat\\delta - c\\,s} - 1\\right)",
        ),
        explanation=(
            "Aralık önce log ölçeğinde kurulur: $\\widehat\\delta \\pm c\\,s$. $100(e^{x} - 1)$ artan bir dönüşüm olduğundan "
            "sınırlar ayrı ayrı dönüştürülür; aralık nokta tahmine göre simetrik olmaz. Kod 10.4'te $-0{,}3011 \\pm "
            "1{,}9645 \\times 0{,}0372$ sınırları (yuvarlanmamış değerlerle) [−31,22; −20,39] verir: nokta tahmin −26,00'dan alt sınır 5,22, üst "
            "sınır 5,61 puan uzaktadır (§10.7)."
        ),
    ),
    Question(
        key="e05", concept="tam-yuzdeden-kukla-katsayisina", note=_note("10.7"),
        prompt=(
            "Bir makale log ücret modelindeki bir kukla için yalnız tam yüzde farkı raporluyor: $P$ (ör. $P = -20$, "
            "yüzde 20 daha düşük demektir). Kukla katsayısını $P$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat\\delta",
            symbols=(Symbol("P", "P", "tam yüzde fark", -50, 80),),
            answer="log(1 + P/100)",
            shown="\\ln\\left(1 + \\frac{P}{100}\\right)",
        ),
        explanation=(
            "Tam yüzde fark $P = 100(e^{\\delta} - 1)$ ise $e^{\\delta} = 1 + P/100$ ve $\\delta = \\ln(1 + P/100)$. "
            "P = −26,00 için $\\ln(0{,}74) \\approx -0{,}301$: Kod 10.4'teki kadın katsayısı. P = −20 için δ ≈ −0,223; "
            "yaklaşık yorum burada −0,20 verirdi. Makalelerden katsayıyı geri elde etmek, farklı çalışmaları aynı "
            "ölçekte karşılaştırmayı sağlar (§10.7)."
        ),
    ),
)


KONU10_QUIZ = QuestionSet(
    topic_key="konu10",
    title="Konu 10: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"10.{number}" for number in range(1, 11)),
)
