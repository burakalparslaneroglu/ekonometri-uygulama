"""Konu 12 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz ve Mini Quiz maddelerini
tekrar etmez: reklam harcamasıyla artan yayılım; altı olası kaynağın ayırt edilmesi; E(u | X) = 0 iken ne değişir, ne
değişmez; 20–50, 50–100 ve 100 üzeri tahminlerde artık aralıkları; BP LM = 9,8 ve White LM = 18,5 kararları; HC1
seçimi ve tablo notu; HPRICE1 düzey çıktısının yedi maddesi (arsa ve konut büyüklüğü); F = 5,40 ve 2,10'luk ortak
testler; Tablo 12.6'nın altı maddesi; 0,080'lik eğitim katsayısıyla bütünleşik raporlama ve mini quizlerdeki sorular.
Aynı becerileri yeni bağlamlarla ve yeni sayılarla sınar: yatak odası katsayısının HC1 aralığı, dayanıklı Wald
istatistiğinin F biçimi, HC0–HC3 sıralaması ve kaldıraç; bazı sorular Sezgi sekmesinin deneylerine (Deney 1'in c = 1,5,
n = 20 ve n = 400 ayarları, Deney 2) dayanır.
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


_SIGMA_BAR = ("\\bar\\sigma", "\\bar \\sigma", "\\overline\\sigma", "\\overline \\sigma", "\\barσ", "\\bar σ",
              "\\overlineσ", "\\overline σ", "σ\u0304", "σ\u0305",
              "sigma_bar", "sigmabar", "σ_bar", "\\sigma_bar", "\\sigma", "sigma", "σ")
"""σ̄ yazımları; süslü parantezler okunmadan önce silindiği için ``\\bar{\\sigma}`` ``\\bar\\sigma`` olarak eşleşir.
Soruda başka bir σ olmadığından yalın σ da σ̄ sayılır."""


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="yanlis-bicim-heteroskedastisiteye-benzer", note=_note("12.2"),
        prompt=(
            "Gerçek ilişki $Y = 1 + 2X - 0{,}3X^2 + u$ iken $Y$ yalnız $X$ üzerine regresyona "
            "sokuluyor. Artıklar X aralığının iki ucunda çoğunlukla negatif ve mutlak değerce büyük, ortada çoğunlukla "
            "pozitif ve küçük çıkıyor. İlk yapılması gereken nedir?"
        ),
        answer=MultipleChoice(
            (
                "Heteroskedastisite kesinleştiği için HC3 ile çıkarım yapıp modeli olduğu gibi raporlamak",
                "White testi reddederse modeli terk edip başka bir veri seti aramak",
                "Uçlardaki büyük artıklı gözlemleri aykırı değer sayıp örneklemden çıkarmak",
                "Koşullu ortalamayı yeniden düşünmek: X² eklemek ve artık grafiğini yeniden incelemek",
            ),
            correct=3,
        ),
        explanation=(
            "Hata varyansı sabit olsa bile dışlanan bir karesel terim, artıkların belirli bölgelerde sistematik olarak "
            "büyümesine yol açar; bu örüntü "
            "heteroskedastisiteye benzer ama kaynağı yanlış kurulmuş koşullu ortalamadır. Dayanıklı standart hata "
            "yanlış ortalama fonksiyonunu düzeltmez; gözlem silmek de gerekçesizdir. Önce fonksiyonel biçim "
            "düzeltilir, sonra artık grafiği ve testler yeniden değerlendirilir (Modelde dışlanan yapı, madde 7) (§12.2)."
        ),
    ),
    Question(
        key="k02", concept="log-artigin-fiyat-duzeyinde-anlami", note=_note("12.4", "Şekil 12.5"),
        prompt=(
            "HPRICE1 log modelinde iki konutun artığı da 0,10'dur. Tahmin edilen log fiyatlarına karşılık gelen "
            "fiyatlar ($e^{\\widehat{\\text{lprice}}}$) 200 ve 500 bin dolardır. Bu artıklar fiyat düzeyinde ne "
            "anlama gelir?"
        ),
        answer=MultipleChoice(
            (
                "İki konut da tahminin yaklaşık %10,5 üstündedir: yaklaşık 21 ve 53 bin dolar",
                "İki konut da tahminin 0,10 bin dolar, yani 100 dolar üstündedir; fark iki konutta aynıdır",
                "Pahalı konutun yüzde sapması daha küçüktür; aynı log artık fiyat büyüdükçe küçülen bir yüzdedir",
                "Log artıklar fiyat düzeyine çevrilemez; yalnız işaretleri yorumlanabilir",
            ),
            correct=0,
        ),
        explanation=(
            "Log artık, fiyatın tahmine oranının logaritmasıdır: $e^{0{,}10} \\approx 1{,}105$, iki konut da tahminin "
            "yaklaşık %10,5 üstündedir. Bin dolar cinsinden sapma $200 \\times 0{,}105 \\approx 21$ ve "
            "$500 \\times 0{,}105 \\approx 53$'tür. Yüzde olarak benzer sapmalar yüksek fiyatlarda daha büyük mutlak "
            "sapmalara dönüşür (Ölçek ile varyans arasındaki sezgisel bağlantı, §12.2): log modelde dengeli görünen "
            "yayılım, düzey modelinde fiyatla büyüyen yayılıma karşılık gelir (Şekil 12.3 ve 12.5) (§12.4)."
        ),
    ),
    Question(
        key="k03", concept="bp-reddetmez-white-reddeder", note=_note("12.5"),
        prompt=(
            "Tek açıklayıcılı bir modelde Breusch–Pagan testi p = 0,30, White testi p = 0,001 veriyor. En makul yorum "
            "hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "İki test aynı sıfır hipotezini sınadığı için farklı sonuç veremez; hesaplardan biri hatalıdır",
                "BP reddetmediği için homoskedastisite kanıtlanmıştır; White'ın sonucu yok sayılır",
                "Varyans X ile U biçiminde ilişkili olabilir ya da ortalama fonksiyonu yanlış olabilir",
                "White testi her zaman BP'den güçlüdür; BP artık kullanılmamalıdır",
            ),
            correct=2,
        ),
        explanation=(
            "İki test aynı sıfır hipotezini sınar ama farklı alternatiflere duyarlıdır. BP'nin yardımcı regresyonu "
            "kareli artıkları yalnız X'in düzeyiyle açıklar; White X²'yi de içerir. Varyans X aralığının ortasında "
            "küçük, uçlarında büyükse doğrusal eğilim zayıftır ve BP reddetmeyebilir, White ise çoğu zaman reddeder. White yanlış "
            "fonksiyonel biçime de duyarlıdır. Konu 12 Sezgi Deney 2'nin varsayılan ayarında (U biçimli varyans, "
            "n = 100) BP örneklemlerin %17,9'unda, White %97,0'ında reddeder (§12.5)."
        ),
    ),
    Question(
        key="k04", concept="sh-orani-heteroskedastisite-izi", note=_note("12.7", "Kod 12.3", "Kod 12.4"),
        prompt=(
            "Arsa büyüklüğü katsayısında HC1/geleneksel standart hata oranı düzey modelinde $1{,}2514/0{,}6421 \\approx "
            "1{,}95$, log modelde $0{,}0415/0{,}0383 \\approx 1{,}08$'dir. Bu fark en iyi neyle uyumludur?"
        ),
        answer=MultipleChoice(
            (
                "Log modelde katsayılar küçük olduğu için standart hatalar da küçüktür; oran farkı ölçekten gelir",
                "Log modelde heteroskedastisite zayıf görünür: testler reddetmez, artık yayılımı daha dengelidir",
                "Oran yalnız HC1'in serbestlik derecesi düzeltmesini yansıtır; iki modelde aynı olmalıydı, fark "
                "yuvarlamadan gelir",
                "Log dönüşümü heteroskedastisiteyi kesin olarak ortadan kaldırmıştır; dayanıklı standart hata gereksizdir",
            ),
            correct=1,
        ),
        explanation=(
            "Oran birimden bağımsızdır: değişkenler yeniden ölçeklenince iki standart hata aynı çarpanla değişir, oran "
            "değişmez. Hata varyansı açıklayıcılarla ilişkili değilse dayanıklı ve geleneksel standart hatalar "
            "birbirine yakın çıkar; oran 1'den uzaklaştıkça heteroskedastisitenin çıkarıma etkisi büyür. Düzey "
            "modelinde testler reddeder ve oran yaklaşık 2'dir; log modelde testler reddetmez (Tablo 12.2) ve oran 1'e "
            "yakındır. HC1/HC0 oranı iki modelde de $\\sqrt{88/84} \\approx 1{,}024$'tür; HC1/geleneksel oranı ise "
            "artıkların yapısına bağlıdır. Reddetmemek homoskedastisiteyi kanıtlamaz; oran bir tanı ipucudur (§12.7)."
        ),
    ),
    Question(
        key="k05", concept="benzetim-ortalamasinin-standart-hatasi", note=_note("12.9", "Tablo 12.6"),
        prompt=(
            "Konu 12 Sezgi Deney 1'de c = 1,5 (n = 60) seçildiğinde 4.000 tekrarın eğim ortalaması 1,971, tahminlerin "
            "standart sapması 1,544'tür; gerçek eğim 2'dir. Bu fark EKK'nin yanlı olduğunu gösterir mi?"
        ),
        answer=MultipleChoice(
            (
                "Evet: 4.000 tekrarın ortalaması 1,971 olup gerçek değer 2'den farklı olduğu için EKK yanlıdır",
                "Hayır: 4.000 tekrarın ortalamasının standart hatası yaklaşık 0,024'tür; −0,029'luk fark bununla "
                "uyumludur",
                "Hayır: heteroskedastisite yalnız sabit terimi etkiler; eğim tahminleri her zaman tam 2'de kalır",
                "Evet: güçlü heteroskedastisite eğimi aşağı çeker; c büyüdükçe yanlılık da büyür",
            ),
            correct=1,
        ),
        explanation=(
            "Benzetim ortalaması da bir tahmindir: 4.000 tekrarın ortalamasının standart hatası $1{,}544/\\sqrt{4000} "
            "\\approx 0{,}0244$'tür. −0,029'luk fark bu standart hatanın yaklaşık 1,2 katıdır; yansızlıkla uyumludur. "
            "Deney her c için aynı çekilişleri kullanır; fark da standart hata da c ile birlikte büyür (c = 0,7'de −0,014 ve "
            "0,012), oranları yaklaşık 1,2'de kalır. Sıfır koşullu ortalama geçerliyken heteroskedastisite eğimi yanlı "
            "yapmaz; sorun standart hatadadır (§12.9)."
        ),
    ),
    Question(
        key="k06", concept="test-reddinde-calisma-akisi", note=_note("12.10"),
        prompt=(
            "Bir hane gıda harcaması modelinde (yatay kesit) White testi p < 0,001 veriyor; varyansın biçimi hakkında "
            "güvenilir bir bilgi yok. Notlardaki çalışma akışına en uygun adım hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Modeli terk edip heteroskedastisite göstermeyen başka bir model bulana kadar aramayı sürdürmek",
                "Varyans bilgisi olmadan tahmin edilmiş bir ağırlıkla WLS'ye geçip geleneksel standart hataları "
                "raporlamak",
                "EKK'yi koruyup belirtilmiş bir HC kovaryansıyla çıkarım yapmak; fonksiyonel biçimi ve veriyi "
                "incelemek",
                "Test reddettiği için bağımlı değişkenin logaritmasını alıp log modeli otomatik olarak raporlamak",
            ),
            correct=2,
        ),
        explanation=(
            "Heteroskedastisite iyi kurulmuş bir koşullu ortalama modelinde de görülebilir; modeli otomatik olarak terk "
            "etmek hatadır. Güvenilir varyans bilgisi yokken WLS'nin kazancı belirsizdir; logaritma da ekonomik yorumu "
            "değiştirir ve otomatik bir çözüm değildir. Temel strateji EKK katsayıları ile HC1 gibi belirtilmiş bir "
            "dayanıklı kovaryansla tekli ve ortak çıkarımdır; fonksiyonel biçim, uç gözlem ve veri kalitesi ayrıca "
            "incelenir (sekiz adımlı akış) (§12.10)."
        ),
    ),
    Question(
        key="k07", concept="ayni-ortalama-farkli-yayilim", note=_note("12.1"),
        prompt=(
            "İki bölgede aynı eğitim düzeyindeki çalışanların ortalama ücreti aynıdır; fakat bir bölgede ücretler bu "
            "ortalamanın çevresinde çok daha geniş dağılır. Bölge kuklası eklenmiş bir ücret regresyonu için hangisi "
            "doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Koşullu ortalamada bölge farkı yoktur; hata varyansı bölgeye göre değişir: heteroskedastisite",
                "Bölge kuklası anlamlı çıkmalıdır; yayılım farkı iki bölgenin ortalama farkı demektir",
                "Yayılım farkı yalnız Y'nin dağılımına aittir; hata varyansı iki bölgede aynı kalır",
                "Model yanlış kurulmuştur; koşullu ortalama bölgeye göre mutlaka farklı olmalıdır",
            ),
            correct=0,
        ),
        explanation=(
            "Koşullu ortalama ve koşullu varyans farklı sorulardır. Ortalamalar aynıysa bölge kuklasının gerçek "
            "katsayısı sıfırdır; ama $\\operatorname{Var}(u \\mid \\text{bölge})$ bölgeye göre değiştiği için hata "
            "homoskedastik değildir. Koşullu ortalama doğru olsa da geleneksel standart hatalar bu durumda güvenilmez "
            "olabilir (§12.1)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="kosulsuz-varyans-kosullu-varyansi-belirlemez", note=_note("12.1", "Şekil 12.1", "Şekil 12.2"),
        prompt=(
            "Şekil 12.1 ve 12.2'deki iki süreçte hatanın koşulsuz varyansı aynıdır; buna rağmen biri homoskedastik, "
            "diğeri heteroskedastiktir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Heteroskedastisite hatanın ne kadar büyük olduğuyla değil, koşullu varyansın X'e bağlı olup olmadığıyla "
            "ilgilidir. Şekil 12.1'de standart sapma her X'te $\\bar\\sigma \\approx 5{,}24$'tür; Şekil 12.2'de "
            "$0{,}3 + 0{,}7X^2$ ile 0,3'ten 11,5'e çıkar. İki sürecin koşulsuz varyansı aynıdır ($\\bar\\sigma^2 \\approx "
            "27{,}42$); yalnız bu varyansın X'e göre dağılışı farklıdır (§12.1)."
        ),
    ),
    Question(
        key="d02", concept="hc3-hc0-dan-kucuk-olamaz", note=_note("12.6", "Tablo 12.3"),
        prompt="Aynı modelde her katsayı için HC3 standart hatası HC0 standart hatasından küçük olamaz.",
        answer=TrueFalse(True),
        explanation=(
            "Her katsayının dayanıklı varyansı, gözleme özgü $\\omega_i$ terimlerinin negatif olmayan ağırlıklarla "
            "toplamıdır; ağırlıklar HC türüne göre değişmez (basit regresyonda $(X_i - \\bar X)^2/[\\sum_j (X_j - "
            "\\bar X)^2]^2$). HC0'da $\\omega_i = \\hat u_i^2$, HC3'te $\\hat u_i^2/(1 - h_i)^2$'dir ($0 < h_i < 1$, "
            "$h_i$ kaldıraç): hiçbir $\\omega_i$ HC0'dakinden küçük olmadığı için hiçbir katsayının HC3 varyansı "
            "HC0'ınkinden küçük olamaz. Fark yüksek kaldıraçlı "
            "gözlemlerden gelir: HPRICE1 düzey modelinde 92.681 kare fitlik arsanın kaldıracı 0,84'tür (sonraki en büyük "
            "arsa 31.000 kare fit). Diğer 87 konuttan bu kadar uzak olan bu tek gözlem, arsa katsayısının HC3 "
            "varyansının %99'unu oluşturur ve standart hatayı 1,223'ten (HC0) 7,148'e çıkarır (Uygulama, Adım 3). "
            "HC1 ise HC0'ın sabit bir katıdır; HC3'ün düzeltmesi gözlemden gözleme değişir (§12.6)."
        ),
    ),
    Question(
        key="d03", concept="dayanikli-f-ssr-ile-hesaplanmaz", note=_note("12.8"),
        prompt=(
            "Heteroskedastisiteye dayanıklı ortak test, kısıtlı ve kısıtsız modellerin artık kareleri toplamlarının "
            "farkından hesaplanan F istatistiğiyle yapılır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "SSR farkına dayanan F istatistiği homoskedastik kovaryans hesabına dayanır; kovaryans türü seçimi iki "
            "modelin SSR'lerini değiştirmez, bu yüzden bu iki SSR'den dayanıklı bir test elde edilemez. Dayanıklı ortak "
            "test, dayanıklı kovaryans matrisi $\\hat V$ ile kurulan Wald testidir ve yazılım onu F biçiminde raporlar: "
            "HPRICE1'de geleneksel $F = 6{,}610$, HC1 ile $F = 2{,}365$ (Tablo 12.5) (§12.8)."
        ),
    ),
    Question(
        key="d04", concept="ortalama-sh-dogru-kapsama-garanti-degil", note=_note("12.9", "Tablo 12.6"),
        prompt=(
            "Konu 12 Sezgi Deney 1'de c = 1,5 (n = 60) seçildiğinde HC3 standart hatalarının ortalaması (1,544) eğim "
            "tahminlerinin standart sapmasına (1,544) eşittir; bu yüzden HC3 aralıklarının kapsama oranı tam yüzde 95 "
            "olmalıdır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Kapsama her örneklemdeki aralığa bağlıdır. Ortalaması doğru olsa da HC3 standart hatası örneklemden "
            "örnekleme değişir; bazı örneklemlerde küçük kalır ve aralık gerçek değeri kaçırır. Bu ayarda HC3 "
            "kapsaması %94,35'tir. 4.000 tekrarda kapsama oranının benzetim standart hatası yaklaşık 0,37 puan "
            "olduğundan bu fark tek başına kesin kanıt değildir; ama çıkarım mantık olarak geçersizdir: ortalamanın "
            "doğru olması her örneklemde doğru standart hata demek değildir. Aynı deneyde n = 20 seçildiğinde HC3 "
            "standart hatalarının ortalaması 1,274, tahminlerin standart sapması 1,286 iken kapsama %93,7'ye düşer: "
            "standart hatanın kendisindeki belirsizlik küçük örneklemde kapsamayı düşürür (§12.9)."
        ),
    ),
    Question(
        key="d05", concept="etkili-gozlem-gerekcesiz-silinmez", note=_note("12.10"),
        prompt=(
            "Bir gözlem yalnız dayanıklı standart hatayı büyüttüğü ya da bir testin sonucunu değiştirdiği için "
            "örneklemden çıkarılmamalıdır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Gözlem çıkarma kararı sonuca değil, veriye dayanır: kayıt hatalıysa ya da örneklem tanımının dışındaysa "
            "bu belgeyle doğrulanır ve gerekçesiyle raporlanır; aksi hâlde gözlem örneklemde kalır (Sık yapılan "
            "hatalar, madde 8). HPRICE1'de 92.681 kare fitlik arsa HC3 standart hatasını büyütür, ama bu tek başına "
            "bir çıkarma gerekçesi değildir; gözlemin sonuca etkisi duyarlılık analizi olarak ayrıca "
            "raporlanabilir (§12.10)."
        ),
    ),
    Question(
        key="d06", concept="buyuk-orneklem-geleneksel-sh-duzelmez", note=_note("12.3"),
        prompt=(
            "Heteroskedastisite hangi biçimde olursa olsun, örneklem büyüdükçe geleneksel standart hataya dayanan "
            "yüzde 95 güven aralıklarının kapsama oranı kendiliğinden yüzde 95'e yaklaşır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Geleneksel formül bütün $\\sigma_i^2$'leri tek bir $\\sigma^2$ ile temsil eder. Hata varyansı "
            "$(X_i - \\bar X)^2$ ile ilişkiliyse bu formül yanlış varyansı hedefler ve örneklem büyüdükçe bu yanlış "
            "hedefe yaklaşır, doğru varyansa değil (ilişki yoksa geleneksel hesap büyük örneklemde yine geçerlidir). "
            "Dayanıklı standart hata ise büyük örneklemde doğru varyansı hedefler. Konu 12 Sezgi Deney 1'de n = 400 "
            "iken geleneksel aralıkların kapsaması %88,50, HC1'inki %94,85'tir (§12.3)."
        ),
    ),
    Question(
        key="d07", concept="artik-ile-hata-ayni-degildir", note=_note("12.4"),
        prompt=(
            "EKK artığı $\\hat u_i = Y_i - \\widehat Y_i$ ile hata terimi $u_i = Y_i - \\mathbb{E}(Y_i \\mid X_i)$ aynı "
            "büyüklüktür; artık, hatanın gözlenmiş değeridir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Hata gerçek koşullu ortalamadan sapmadır ve gözlenemez; artık tahmin edilen doğrudan sapmadır ve tahmin "
            "edilen katsayılara bağlıdır: basit regresyonda $\\hat u_i = u_i - (\\widehat\\beta_0 - \\beta_0) - "
            "(\\widehat\\beta_1 - \\beta_1)X_i$. Bu yüzden artık grafiği hata varyansı için bir tanı aracıdır, doğrudan "
            "ölçüm değildir; model yanlış kurulmuşsa artıklar dışlanan yapıyı da taşır (Artık grafiğini okuma "
            "sırası) (§12.4)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="heteroskedastik-varyans-formulu-hesabi", note=_note("12.3"),
        prompt=(
            "Dört gözlemde $X_i - \\bar X$ değerleri −2, −1, 1, 2'dir. Hata varyansları uçlarda büyükse "
            "($\\sigma_i^2$ = 4, 1, 1, 4) eğimin gerçek koşullu varyansı **(1)**, ortada büyükse ($\\sigma_i^2$ = 1, 4, "
            "4, 1) **(2)** olur. (İki durumda da $\\sigma_i^2$'lerin ortalaması 2,5 ve homoskedastik formülün değeri "
            "$2{,}5/10 = 0{,}25$'tir.)"
        ),
        answer=FillBlanks((NumberBlank(0.34, 0.001, "0,34"), NumberBlank(0.16, 0.001, "0,16"))),
        explanation=(
            "$\\operatorname{Var}(\\widehat\\beta_1 \\mid X) = \\sum (X_i - \\bar X)^2\\sigma_i^2 / [\\sum (X_i - \\bar "
            "X)^2]^2$ ve $\\sum (X_i - \\bar X)^2 = 10$. Birinci durumda $(4 \\cdot 4 + 1 + 1 + 4 \\cdot 4)/100 = 0{,}34$, "
            "ikincide $(4 + 4 + 4 + 4)/100 = 0{,}16$. Büyük varyanslı gözlemler $\\bar X$'dan uzaksa geleneksel formül "
            "(0,25) belirsizliği küçümser, yakınsa abartır: geleneksel formülün hatasının yönü önceden bilinmez "
            "(§12.3)."
        ),
    ),
    Question(
        key="b02", concept="bp-lm-hesabi-ve-karari", note=_note("12.5"),
        prompt=(
            "n = 200 gözlemli üç açıklayıcılı bir modelde Breusch–Pagan yardımcı regresyonunun $R^2$'si 0,045'tir. "
            "$LM$ = **(1)** olur. Yüzde 5 düzeyinde $\\chi^2_3$ dağılımının kritik değeri **(2)** olur (virgülden "
            "sonra üç basamak)."
        ),
        answer=FillBlanks((NumberBlank(9.0, 0.005, "9"), NumberBlank(7.815, 0.0005, "7,815"))),
        explanation=(
            "$LM = nR^2_{aux} = 200 \\times 0{,}045 = 9{,}0$; serbestlik derecesi yardımcı regresyondaki açıklayıcı "
            "sayısıdır, q = 3. $9{,}0 > 7{,}815$ olduğundan homoskedastisite yüzde 5 düzeyinde reddedilir "
            "($p \\approx 0{,}029$). Ret, katsayıların yanlı olduğunu değil, geleneksel çıkarımın güvenilmez olabileceğini "
            "gösterir (§12.5)."
        ),
    ),
    Question(
        key="b03", concept="hc1-araligi-gelenekselden-dar-olabilir", note=_note("12.7", "Kod 12.3"),
        prompt=(
            "Kod 12.3'te yatak odası katsayısı 13,8525, HC1 standart hatası 8,4786'dır; $t_{0{,}025;\\,84} = 1{,}989$. "
            "HC1 yüzde 95 güven aralığı: alt sınır **(1)**, üst sınır **(2)** (virgülden sonra üç basamak)."
        ),
        answer=FillBlanks((NumberBlank(-3.0095, 0.002, "−3,011"), NumberBlank(30.7145, 0.002, "30,716"))),
        explanation=(
            "$13{,}8525 \\pm 1{,}989 \\times 8{,}4786 = 13{,}8525 \\pm 16{,}864$: [−3,011; 30,716] (yuvarlanmamış "
            "değerlerle [−3,008; 30,713]). Geleneksel aralık (standart hata 9,0101 ile yaklaşık [−4,069; 31,774]) daha "
            "geniştir: bu katsayıda HC1 standart hatası gelenekselden küçüktür. Heteroskedastisite geleneksel standart "
            "hatayı her katsayıda aynı yönde bozmaz; arsa katsayısında HC1 aralığı genişler, burada daralır. İki "
            "aralık da sıfırı içerir (§12.7)."
        ),
    ),
    Question(
        key="b04", concept="kucuk-orneklemde-hc1-acigi", note=_note("12.9", "Tablo 12.6"),
        prompt=(
            "Konu 12 Sezgi Deney 1'de n = 20 (c = 0,7) seçildiğinde eğim tahminlerinin standart sapması 1,286, ortalama "
            "geleneksel standart hata 0,993, ortalama HC1 standart hata 1,135'tir. Geleneksel standart hata gerçek "
            "yayılımı yüzde **(1)**, HC1 yüzde **(2)** küçümser (virgülden sonra iki basamak)."
        ),
        answer=FillBlanks((NumberBlank(22.775, 0.0095, "22,78"), NumberBlank(11.755, 0.0205, "11,74"))),
        explanation=(
            "$100(1 - 0{,}993/1{,}286) \\approx 22{,}78$ ve $100(1 - 1{,}135/1{,}286) \\approx 11{,}74$ (yuvarlanmamış "
            "değerlerle 22,77 ve 11,77). HC1 büyük örneklem mantığına dayanır: açığı n = 60'ta yaklaşık %3,5 iken "
            "n = 20'de %12'ye çıkar. HC3 kaldıraç düzeltmesiyle açığın büyük kısmını kapatır (ortalama 1,274; açık "
            "yaklaşık %1). Küçük örneklemde hiçbir düzeltme bütün sorunu ortadan kaldırmaz (§12.9)."
        ),
    ),
    Question(
        key="b05", concept="log-donusumu-farki-azaltir-kaldirmaz", note=_note("12.10"),
        prompt=(
            "WAGE1'de eğitim katsayısının standart hatası ücret düzeyi modelinde 0,049 (geleneksel) ve 0,061 (HC1), log "
            "ücret modelinde 0,0069 ve 0,0080'dir. HC1/geleneksel oranı düzey modelinde **(1)**, log modelde **(2)** "
            "olur (virgülden sonra iki basamak)."
        ),
        answer=FillBlanks((NumberBlank(1.24, 0.0055, "1,24"), NumberBlank(1.155, 0.0055, "1,16"))),
        explanation=(
            "$0{,}061/0{,}049 \\approx 1{,}24$ ve $0{,}0080/0{,}0069 \\approx 1{,}16$ (yuvarlanmamış standart hatalarla "
            "1,24 ve 1,15). Log dönüşümü dayanıklı ve geleneksel standart hatalar arasındaki farkı azaltır ama "
            "kaldırmaz; bu, log modelde White testinin yüzde 5 düzeyinde reddetmesiyle uyumludur. İki modelde de "
            "dayanıklı standart hata raporlamak makuldür (§12.10)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="white-yardimci-regresyon-terim-sayisi", note=_note("12.5"),
        prompt=(
            "Modelde hiçbiri 0/1 kukla olmayan k açıklayıcı vardır. White testinin yardımcı regresyonu düzeyleri, "
            "kareleri ve bütün ikili çapraz çarpımları içeriyorsa sabit dışındaki terim sayısı q'yu k cinsinden yazın."
        ),
        answer=Equation(
            lhs="q",
            symbols=(Symbol("k", "k", "açıklayıcı sayısı", 1, 10),),
            answer="k*(k + 3)/2",
            shown="\\frac{k\\,(k + 3)}{2}",
        ),
        explanation=(
            "k düzey, k kare ve $k(k - 1)/2$ çapraz çarpım vardır: $2k + k(k - 1)/2 = k(k + 3)/2$. HPRICE1 düzey "
            "modelinde k = 3 için q = 9; açıklayıcı sayısı arttıkça terim sayısı hızla büyür ve serbestlik derecesi "
            "tüketir. 0/1 kuklanın karesi kendisi olduğundan yalnız bir kez sayılır: WAGE1'deki dört açıklayıcılı "
            "modelde (bir kukla) q = 13 (§12.5)."
        ),
    ),
    Question(
        key="e02", concept="hc1-ve-hc0-iliskisi", note=_note("12.6", "Tablo 12.3"),
        prompt=(
            "Bir katsayının HC0 standart hatası $s_0$'dır. Modelde n gözlem ve sabit dışında k açıklayıcı varsa HC1 "
            "standart hatasını $s_0$, n ve k cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\operatorname{se}_{HC1}",
            symbols=(
                Symbol("s0", "s_0", "HC0 standart hatası", 0.1, 3, aliases=("s_0", "s0", "se_HC0", "seHC0", "SE_HC0", "SEHC0")),
                Symbol("n", "n", "gözlem sayısı", 20, 500),
                Symbol("k", "k", "sabit dışındaki açıklayıcı sayısı", 1, 6),
            ),
            answer="s0*sqrt(n/(n - k - 1))",
            shown="s_0\\sqrt{\\frac{n}{n - k - 1}}",
        ),
        explanation=(
            "HC1 her kareli artığı $n/(n - k - 1)$ ile çarpar; varyans bu oranla, standart hata onun kareköküyle "
            "büyür. HPRICE1 düzey modelinde n = 88, k = 3: $1{,}22265 \\times \\sqrt{88/84} \\approx 1{,}2514$ (HC0 değeri "
            "Uygulama, Adım 3'te; HC1 değeri Kod 12.3'te). "
            "Örneklem büyüdükçe oran 1'e yaklaşır ve HC0 ile HC1 birbirine yaklaşır (§12.6)."
        ),
    ),
    Question(
        key="e03", concept="dayanikli-wald-ki-kare-ve-f-bicimi", note=_note("12.8", "Tablo 12.5"),
        prompt=(
            "Dayanıklı kovaryans matrisiyle kurulan Wald istatistiği W, q kısıt için $H_0$ doğruyken büyük örneklemde "
            "$\\chi^2_q$ dağılımına yaklaşır. Yazılımın raporladığı F biçimini W ve q cinsinden yazın."
        ),
        answer=Equation(
            lhs="F",
            symbols=(
                Symbol("W", "W", "dayanıklı Wald istatistiği", 0.5, 20),
                Symbol("q", "q", "kısıt sayısı", 1, 6),
            ),
            answer="W/q",
            shown="\\frac{W}{q}",
        ),
        explanation=(
            "F biçimi Wald istatistiğinin kısıt sayısına bölünmesidir; p-değeri $F(q, n - k - 1)$ dağılımından okunur. "
            "HPRICE1'de arsa ve yatak odası katsayılarının HC1 ile ortak testinde W = 4,730 ve F = 4,730/2 = 2,365 "
            "(Tablo 12.5): $\\chi^2_2$ ile p = 0,094, $F(2, 84)$ ile p = 0,1002. Bu örnekte F biçimi biraz daha büyük "
            "p-değeri verir; büyük örneklemde iki biçim birbirine yaklaşır (§12.8)."
        ),
    ),
    Question(
        key="e04", concept="wls-agirligi", note=_note("12.10"),
        prompt=(
            "Hata varyansının $\\operatorname{Var}(u_i \\mid X_i) = \\sigma^2X_i^2$ olduğu biliniyor ($X_i > 0$). WLS "
            "katsayıları $\\sum_i w_i(Y_i - b_0 - b_1X_i)^2$ toplamını en küçükler. Gözlem i'nin ağırlığı $w_i$'yi, "
            "sabit çarpanı 1 alarak $X_i$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="w_i",
            symbols=(Symbol("X", "X_i", "açıklayıcı değişken", 0.5, 5, aliases=("X_i", "x_i", "Xi", "xi")),),
            answer="1/X^2",
            shown="\\frac{1}{X_i^2}",
        ),
        explanation=(
            "Ağırlık hata varyansıyla ters orantılıdır: $w_i \\propto 1/\\sigma_i^2 = 1/(\\sigma^2X_i^2)$; sabit çarpan "
            "atılınca $1/X_i^2$. Büyük X'li, büyük varyanslı gözlemlerin kareli sapmaları toplamda daha az ağırlık alır. "
            "WLS doğru ağırlıklarla EKK'den daha etkin olabilir; varyans yapısı yanlış kurulursa bu kazanç kaybolabilir, "
            "bu yüzden güvenilir bilgi yoksa EKK ile dayanıklı standart hata raporlanır (§12.10)."
        ),
    ),
    Question(
        key="e05", concept="yayilimlarin-esitlendigi-nokta", note=_note("12.1", "Şekil 12.1", "Şekil 12.2"),
        prompt=(
            "Şekil 12.2'deki heteroskedastik süreçte hatanın koşullu standart sapması $a + cX^2$'dir ($X \\ge 0$, "
            "$c > 0$); Şekil 12.1'deki homoskedastik süreçte ise her X'te $\\bar\\sigma$'dır ($\\bar\\sigma > a$). İki "
            "koşullu standart sapmanın eşitlendiği $X^*$ değerini a, c ve $\\bar\\sigma$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="X^*",
            symbols=(
                Symbol("a", "a", "sabit bileşen", 0.1, 0.5),
                Symbol("c", "c", "heteroskedastisitenin gücü", 0.2, 1.5),
                Symbol("sb", "\\bar\\sigma", "homoskedastik süreçteki standart sapma", 2, 8, aliases=_SIGMA_BAR),
            ),
            answer="sqrt((sb - a)/c)",
            shown="\\sqrt{\\frac{\\bar\\sigma - a}{c}}",
        ),
        explanation=(
            "$a + cX^2 = \\bar\\sigma$ eşitliğinden $X^* = \\sqrt{(\\bar\\sigma - a)/c}$. Notlardaki süreçte a = 0,3, "
            "c = 0,7 ve $\\bar\\sigma \\approx 5{,}236$: $X^* \\approx 2{,}66$. X ~ U(0, 4) olduğundan gözlemlerin "
            "yaklaşık üçte ikisinde (X < 2,66) heteroskedastik süreçteki yayılım homoskedastik süreçtekinden küçük, "
            "üçte birinde büyüktür; hatanın koşulsuz varyansı ise iki süreçte aynıdır (§12.1)."
        ),
    ),
)


KONU12_QUIZ = QuestionSet(
    topic_key="konu12",
    title="Konu 12: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"12.{number}" for number in range(1, 11)),  # §12.11 bölüm özetidir
)
