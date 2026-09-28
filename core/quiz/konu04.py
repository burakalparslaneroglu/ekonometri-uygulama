"""Konu 4 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 4.1–4.12 ve Mini Quiz
4.1–4.12 maddelerini tekrar etmez: EKK özelliklerine ilişkin beş doğru–yanlış ifadesi; ortalaması 68 olan sınıf;
Ȳ = 50, Ŷ = 58, Y = 61 ile Y = 45, Ŷ = 47 ayrıştırmaları; TKT = 500 ve HKT = 320; dört R² hesabı; dört R² yorum
hatası; satış = 20 + 3 reklam birim dönüşümleri; dört denklemin biçim sınıflandırması; 5 + 2X, 1 + 0,03X,
4 + 20 ln X ve 0,7 + 1,4 ln X yorumları; Kod 4.4 çıktısının sıralı okunması; Tablo 4.4'te 250 kare fit ve yüzde 5
yorumları; iki satış modelli bütünleşik uygulama ve mini quizlerdeki sorular. Aynı becerileri yeni bağlamlarla ve
yeni sayılarla sınar. Standart hata, t, p-değeri ve güven aralığı Konu 7'nin; logaritmik modellerde tam yüzde
değişim Konu 9'un konusudur; sorular yalnız küçük değişim yaklaşımını kullanır.
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
    TrueFalse,
)


def _note(section: str, *objects: str) -> NoteRef:
    return NoteRef(section, 0, tuple(objects))


_SX = ("s_X", "s_x", "S_X", "S_x")
_SY = ("s_Y", "s_y", "S_Y", "S_y")


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="kucuk-sayisal-artik-toplami", note=_note("4.1", "(4.1)"),
        prompt=(
            "Bir yazılım, sabit terimli bir EKK modelinde artıkların toplamını $-3{,}2 \\times 10^{-14}$ olarak "
            "gösteriyor. Bu sayı için en doğru yorum hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Model, gözlemleri sistematik olarak olduklarından daha yüksek tahmin etmektedir",
                "Sayısal hesaplama hassasiyetinden kaynaklanır; uygulamada sıfır kabul edilir",
                "Artıkların ortalaması negatif olduğu için sabit terim bu kadar küçültülmelidir",
                "Kareli artıklar toplamının da sıfıra çok yakın olduğunu gösterir",
            ),
            correct=1,
        ),
        explanation=(
            "Eşitlik (4.1)'e göre sabit terimli EKK'de artıkların toplamı matematiksel olarak sıfırdır. Ekrandaki çok "
            "küçük değer yuvarlama ve sayısal hesaplama hassasiyetinden doğar; işareti yazılıma ve bilgisayara göre "
            "değişebilir. İşaretli toplamın sıfır olması artıkların küçük olduğunu göstermez: pozitif ve negatif "
            "artıklar birbirini dengeler, kareli artıklar toplamı büyük olabilir (§4.1)."
        ),
    ),
    Question(
        key="k02", concept="ssr-kisaltmasini-dogrulama", note=_note("4.4"),
        prompt=(
            "Bir yazılım çıktısında yalnız `SSR = 412` yazıyor. Bu sayının "
            "TKT, MKT ve HKT'den hangisi olduğuna nasıl karar verirsiniz?"
        ),
        answer=MultipleChoice(
            (
                "SSR her kaynakta artık kareleri toplamıdır; bu yüzden doğrudan HKT olarak alınır",
                "SSR her kaynakta model kareleri toplamıdır; doğrudan MKT olarak alınır",
                "SSR toplam kareler toplamının kısaltmasıdır; doğrudan TKT olarak alınır",
                "Hangi toplam olduğu, formülden ya da yazılımın açıklamasından kontrol edilir",
            ),
            correct=3,
        ),
        explanation=(
            "SSR bazı kaynaklarda artık kareleri toplamı (sum of squared residuals), bazılarında model kareleri "
            "toplamı (regression sum of squares) anlamında kullanılır. Kısaltmaya değil formüle bakılır: $\\sum "
            "\\widehat{u}_i^2$ HKT'yi, $\\sum (\\widehat{Y}_i - \\bar{Y})^2$ MKT'yi verir. Notlar karışıklığı azaltmak "
            "için TKT, MKT ve HKT kısaltmalarını kullanır (§4.4)."
        ),
    ),
    Question(
        key="k03", concept="r2-oran-olarak-okunur", note=_note("4.5", "(4.6)"),
        prompt=(
            "Farklı veri setleriyle tahmin edilen sabit terimli iki basit regresyonda $R^2$ sırasıyla 0,95 ve "
            "0,40'tır. Aşağıdakilerden hangisi kesinlikle doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Birinci modelde toplam değişkenliğin artıklarda kalan payı (HKT/TKT) daha küçüktür",
                "Birinci modelin kareli artıklar toplamı (HKT) ikinci modelinkinden küçüktür",
                "Birinci modelin eğim tahmini, ikinci modelin eğim tahmininden mutlak değerce büyüktür",
                "Birinci modelde X'in Y üzerindeki nedensel etkisi daha güçlüdür",
            ),
            correct=0,
        ),
        explanation=(
            "$R^2 = 1 - \\text{HKT}/\\text{TKT}$ olduğundan HKT/TKT birinci modelde 0,05, ikincide 0,60'tır: "
            "artıklarda kalan pay birincide daha küçüktür. HKT'nin kendisi ise TKT'ye bağlıdır; bağımlı değişkenleri "
            "ve örneklemleri farklı iki modelde birinci modelin HKT'si daha büyük bile olabilir. $R^2$ eğimin "
            "büyüklüğü hakkında bilgi vermez ve nedensellik ölçüsü değildir (§4.5, §4.6)."
        ),
    ),
    Question(
        key="k04", concept="ortak-egilim-ve-yuksek-r2", note=_note("4.6"),
        prompt=(
            "Bir ülkenin 1995–2024 yıllık verisiyle internet abone sayısı ile ortalama yaşam süresi arasında kurulan "
            "regresyonda $R^2 = 0{,}97$ bulunuyor. En uygun değerlendirme hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Abonelik yaşam süresini uzatır; yüksek $R^2$ bunu kanıtlar",
                "Uyum çok yüksek olduğu için model başka ülkelerde de aynı başarıyı gösterir",
                "Ortak zaman eğilimi yüksek $R^2$ üretebilir; yüksek uyum tek başına ekonomik ilişki göstermez",
                "$R^2$ yüksek olduğuna göre veri kalitesi iyidir, doğru değişkenler seçilmiştir ve model başka "
                "dönemlerde de kullanılabilir",
            ),
            correct=2,
        ),
        explanation=(
            "İki seri aynı dönemde güçlü biçimde yükseliyorsa regresyon çok yüksek $R^2$ üretebilir; ortak zaman "
            "eğilimi gerçek bir ekonomik ilişki yerine yanıltıcı bir uyum oluşturabilir. Yüksek $R^2$ nedenselliği "
            "kanıtlamaz; doğru değişkenlerin seçildiğini, veri kalitesini ya da başka örneklemlerdeki başarıyı da "
            "garanti etmez. Zaman serilerine özgü bu sorunlar sonraki dönemin konusudur (§4.6)."
        ),
    ),
    Question(
        key="k05", concept="soruya-uygun-bicim", note=_note("4.8", "Tablo 4.2"),
        prompt=(
            "Bir araştırmacı “Nüfusu yüzde 1 daha büyük olan ilçelerde ortalama aylık kira kaç TL farklıdır?” sorusunu "
            "cevaplamak istiyor. Hangi fonksiyonel biçim bu soruyu doğrudan cevaplar?"
        ),
        answer=MultipleChoice(
            (
                "Düzey–düzey: $\\text{kira} = \\beta_0 + \\beta_1\\,\\text{nüfus} + u$",
                "Log–düzey: $\\ln(\\text{kira}) = \\beta_0 + \\beta_1\\,\\text{nüfus} + u$",
                "Log–log: $\\ln(\\text{kira}) = \\beta_0 + \\beta_1 \\ln(\\text{nüfus}) + u$",
                "Düzey–log: $\\text{kira} = \\beta_0 + \\beta_1 \\ln(\\text{nüfus}) + u$",
            ),
            correct=3,
        ),
        explanation=(
            "Soru, açıklayıcı değişkende yüzde değişim, sonuç değişkeninde TL cinsinden (birim) değişim istiyor. "
            "Düzey–log modelde nüfus yüzde 1 arttığında kira yaklaşık $\\beta_1/100$ TL değişir. Log–düzey model "
            "birimlik nüfus farkına yüzde kira farkı, log–log model yüzde farka yüzde fark (esneklik), düzey–düzey "
            "model birimlik farka birimlik fark verir (§4.8, Tablo 4.2)."
        ),
    ),
    Question(
        key="k06", concept="egimler-farkli-degisim-turleri", note=_note("4.9", "Tablo 4.3"),
        prompt=(
            "Tablo 4.3'te dört modelin eğimleri 0,1402; 0,000402; "
            "297,911 ve 0,8727'dir. Bu sayılar hakkında hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "En güçlü ilişki düzey–log modelindedir; çünkü 297,911'lik "
                "eğim öteki üç eğimden sayısal olarak çok daha büyüktür",
                "Log–düzey modelde ilişki yok sayılabilir; çünkü eğimi sıfıra çok yakındır",
                "Her eğim farklı bir değişim türünü ölçer; sayısal büyüklükleri doğrudan karşılaştırılamaz",
                "Log–log eğimi esneklik olduğu için dört eğim de yüzde cinsinden okunur",
            ),
            correct=2,
        ),
        explanation=(
            "Eğimlerin birimleri farklıdır: düzey–düzey eğimi kare fit başına bin dolar; log–düzey eğiminin 100 katı "
            "kare fit başına yaklaşık yüzde fiyat farkı; düzey–log eğiminin 100'de biri büyüklükteki %1'lik farka "
            "karşılık gelen fiyat farkı (bin dolar); log–log eğimi esnekliktir. Log–düzey eğiminin küçüklüğü ölçü "
            "biriminden gelir: 100 kare fitlik fark yaklaşık yüzde 4,02 fiyat farkıdır (§4.9, Tablo 4.3)."
        ),
    ),
    Question(
        key="k07", concept="karsilastirilabilir-r2-sutunlari", note=_note("4.11", "Tablo 4.3", "Tablo 4.4"),
        prompt=(
            "Tablo 4.4'ü Tablo 4.3'teki dört modele genişleten bir makale tablosu şu sütunları raporluyor: (1) bağımlı "
            "değişken fiyat, açıklayıcı büyüklük; (2) bağımlı ln(fiyat), açıklayıcı ln(büyüklük); (3) bağımlı "
            "ln(fiyat), açıklayıcı büyüklük; (4) bağımlı fiyat, açıklayıcı ln(büyüklük). Dört model de aynı 88 konutla "
            "tahmin edilmiştir. Hangi iki sütunun $R^2$ değerleri doğrudan karşılaştırılabilir?"
        ),
        answer=MultipleChoice(("(1) ve (2)", "(2) ve (3)", "(1) ve (3)", "(2) ve (4)"), correct=1),
        explanation=(
            "$R^2$ yalnız aynı bağımlı değişkenin aynı örneklemdeki değişkenliğini özetleyen modeller arasında "
            "karşılaştırılabilir. (2) ve (3) log fiyatın, (1) ve (4) fiyatın değişkenliğini özetler; seçeneklerdeki "
            "öteki çiftler farklı bağımlı değişkenleri karşılaştırır. Tablo 4.3'te bu iki modelin $R^2$ değerleri "
            "0,5530 ve 0,5837'dir (§4.11, §4.6)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="iyilesme-toplamda-her-gozlemde-degil", note=_note("4.2"),
        prompt=(
            "WAGE1'de eğitim bilgisini kullanan EKK doğrusu, yalnız örneklem ortalamasını kullanan tahmine göre kareli "
            "tahmin hatasını her çalışan için küçültür."
        ),
        answer=TrueFalse(False),
        explanation=(
            "EKK kareli hataların toplamını küçültür: WAGE1'de yalnız ortalamayı kullanan tahminin toplamı (TKT) "
            "7160,4, EKK doğrusununki (HKT) 5980,7'dir. Tek tek gözlemlerde bu garanti değildir: 526 çalışanın "
            "220'sinde EKK tahmininin kareli hatası, ortalama tahmininkinden büyüktür. Model uyumu toplam üzerinden "
            "değerlendirilir (§4.2, §4.4)."
        ),
    ),
    Question(
        key="d02", concept="r2-sifir-egim-sifir", note=_note("4.5"),
        prompt="Sabit terimli basit regresyonda $R^2 = 0$ ise EKK eğim tahmini de sıfırdır.",
        answer=TrueFalse(True),
        explanation=(
            "$R^2 = \\text{MKT}/\\text{TKT}$ ve §4.5'teki türetime göre MKT, $\\widehat{\\beta}_1^{\\,2} \\sum (X_i - "
            "\\bar{X})^2$ ifadesine eşittir. X'te değişkenlik varken MKT = 0 ancak $\\widehat{\\beta}_1 = 0$ ile "
            "mümkündür: tahmin edilen doğru yataydır ve her gözlem için $\\bar{Y}$ tahmin eder. Eşdeğer olarak $R^2 = "
            "r_{XY}^2 = 0$, korelasyonun sıfır olması demektir (§4.5)."
        ),
    ),
    Question(
        key="d03", concept="olcek-degisikligi-ekonomik-degisiklik-degil", note=_note("4.7"),
        prompt=(
            "2021 ve 2025 yıllarının nominal konut fiyatlarını içeren bir veri setinde bütün fiyatları TL'den bin "
            "TL'ye çevirmek, iki yıl arasındaki enflasyon kaynaklı fiyat farkını ortadan kaldırır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Birim dönüşümünde bütün değerler aynı sayıyla bölünür: iki yılın fiyatları arasındaki oran değişmez, fark "
            "da yalnız yeni birimde yazılır. Enflasyon kaynaklı artış ise verinin kendisindeki bir değişimdir; ölçek "
            "seçimiyle ortadan kalkmaz. Birim dönüşümü ile ekonomik hareketi ayırmak gerekir (§4.7)."
        ),
    ),
    Question(
        key="d04", concept="yaklasiklik-buyuk-farkta-bozulur", note=_note("4.8", "Tablo 4.2", "Kod 4.4"),
        prompt=(
            "Kod 4.4'teki eğitim katsayısına göre bir yıllık eğitim farkı için “yaklaşık %8,27 daha yüksek ücret” "
            "yorumu, on yıllık fark için “yaklaşık %82,7 daha yüksek ücret” yorumundan daha isabetlidir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Log–düzey modelde $100\\,\\widehat{\\beta}_1\\,\\Delta X$ yaklaşık yüzde değişimdir ve küçük değişimler "
            "için uygundur (Tablo 4.2 ve altındaki not). On yıllık farkta log ücret farkı 0,827'dir; bu büyüklükte log "
            "farkı ile tam yüzde değişim arasındaki fark büyür (§0.6'daki log farkı tablosunda olduğu gibi). Tam yüzde "
            "dönüşümü Konu 9'da işlenir (§4.8)."
        ),
    ),
    Question(
        key="d05", concept="ciktidan-bicimi-belirleme", note=_note("4.10"),
        prompt=(
            "CEOSAL1 verisiyle tahmin edilen bir modelin çıktısında `Dep. Variable: lsalary` (CEO maaşının doğal "
            "logaritması) ve tek açıklayıcı değişken olarak `lsales` (firma satışlarının doğal logaritması) görülüyor; "
            "`lsales` katsayısı 0,2567'dir. Bu çıktıya göre satışları yüzde 1 daha yüksek olan firmalarda CEO maaşı "
            "örneklemde yaklaşık yüzde 0,26 daha yüksektir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Çıktıda iki değişken de logaritmiktir: model log–log biçimindedir ve eğim esnekliktir. Satışlardaki yüzde "
            "1'lik fark, maaşta yaklaşık yüzde 0,257'lik farkla ilişkilidir. Yorum bir ilişkidir; satışları artırmanın "
            "maaşı nedensel olarak artırdığını tek başına göstermez. `Dep. Variable` satırı bağımlı değişkeni, katsayı "
            "satırının adı açıklayıcı değişkeni ve biçimini söyler (§4.10, Tablo 4.2)."
        ),
    ),
    Question(
        key="d06", concept="r2-icin-ayni-orneklem", note=_note("4.12"),
        prompt=(
            "Aynı bağımlı değişkene sahip iki modelden biri 88 konutla, diğeri eksik gözlemler çıkarıldıktan sonra "
            "kalan 60 konutla tahmin edilmişse iki modelin $R^2$ değerleri doğrudan karşılaştırılabilir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Kontrol listesinin dokuzuncu sorusu, modellerin aynı bağımlı değişken ve aynı örneklem üzerinde "
            "karşılaştırılıp karşılaştırılmadığını sorar. $R^2$ bağımlı değişkenin o örneklemdeki değişkenliğine göre "
            "hesaplanır; örneklem değişince TKT de değişir ve iki oran farklı temellere dayanır (§4.12)."
        ),
    ),
    Question(
        key="d07", concept="duzey-log-yuzde-birlik-degisim", note=_note("4.9", "§4.9.3"),
        prompt=(
            "$\\widehat{\\text{ücret}} = 3 + 5\\ln(\\text{çalışan sayısı})$ modelinde (ücret dolar/saat) firmanın "
            "çalışan sayısı yüzde 1 arttığında tahmin edilen saatlik ücret yaklaşık 5 dolar artar."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Düzey–log modelde X yüzde 1 arttığında Y yaklaşık $\\beta_1/100$ birim değişir: 5/100 = 0,05 dolar. 5 "
            "dolarlık fark, ln(çalışan sayısı) bir birim arttığında, yani çalışan sayısı yaklaşık 2,72 katına "
            "çıktığında ortaya çıkar; bu küçük bir değişim değildir (§4.9, Tablo 4.2)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="artik-toplamindan-eksik-artik", note=_note("4.1", "(4.1)"),
        prompt=(
            "Sabit terimli bir EKK modelinde beş gözlemin artıklarından dördü 1,2; −0,4; 0,9 ve −2,1'dir. Beşinci "
            "artık **(1)** olmalıdır. Beş artığın kareleri toplamı (HKT) **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(0.4, 0.005, "0,4"), NumberBlank(6.98, 0.005, "6,98"))),
        explanation=(
            "Eşitlik (4.1)'e göre sabit terimli EKK'de artıkların toplamı sıfırdır; dört artığın toplamı −0,4 "
            "olduğundan beşincisi 0,4'tür. Kareli artıklar toplamı ise 1,44 + 0,16 + 0,81 + 4,41 + 0,16 = 6,98: "
            "işaretli toplam sıfır olsa da kareli toplam pozitiftir. EKK'nin en küçük yaptığı ölçüt ikincisidir (§4.1)."
        ),
    ),
    Question(
        key="b02", concept="sapmalardan-artik-ve-tahmin", note=_note("4.3", "(4.4)"),
        prompt=(
            "Bir örneklemde $\\bar{Y} = 20$'dir. Bir gözlemin toplam sapması $Y_i - \\bar{Y} = -6$, model kaynaklı "
            "sapması $\\widehat{Y}_i - \\bar{Y} = -2$'dir. Bu gözlemin artığı **(1)**, tahmin edilen değeri "
            "$\\widehat{Y}_i$ ise **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(-4, 0.005, "−4"), NumberBlank(18, 0.005, "18"))),
        explanation=(
            "(4.4)'e göre toplam sapma, model kaynaklı sapma ile artığın toplamıdır: −6 = −2 + $\\widehat{u}_i$, "
            "dolayısıyla $\\widehat{u}_i = -4$. Tahmin edilen değer 20 − 2 = 18, gözlenen değer 20 − 6 = 14'tür; artık "
            "14 − 18 = −4 ile tutarlıdır. Gözlem hem ortalamanın hem de tahmin edilen değerin altındadır (§4.3)."
        ),
    ),
    Question(
        key="b03", concept="kucuk-ornekte-kareler-toplamlari", note=_note("4.4", "(4.5)"),
        prompt=(
            "Bölüm 3'teki beş öğrenci örneğinde (Tablo 3.2 ve Tablo 3.3) toplam kareler toplamı TKT = **(1)**, model "
            "kareleri toplamı MKT = **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(338, 0.05, "338"), NumberBlank(336.4, 0.05, "336,4"))),
        explanation=(
            "TKT, Tablo 3.2'deki $Y_i - \\bar{Y}$ sapmalarının karelerinin toplamıdır: 121 + 36 + 1 + 36 + 144 = 338. "
            "Tablo 3.3'teki kareli artıkların toplamı HKT = 1,60 olduğundan (4.5)'ten MKT = 338 − 1,60 = 336,4 "
            "bulunur. Aynı değer $\\widehat{\\beta}_1^{\\,2} \\sum (X_i - \\bar{X})^2 = 2{,}9^2 \\times 40$ ile de "
            "elde edilir; bu küçük örnekte $R^2 \\approx 0{,}995$ olur (§4.4)."
        ),
    ),
    Question(
        key="b04", concept="iki-birim-birlikte-degisince-egim", note=_note("4.7", "Tablo 4.1"),
        prompt=(
            "WAGE1'de $\\widehat{\\text{wage}} = -0{,}9049 + 0{,}5414\\,\\text{educ}$ tahmin edilmiştir (ücret "
            "dolar/saat, eğitim yıl). Ücret sent/saat, eğitim ay cinsinden yazılırsa yeni eğim **(1)** olur. Eğitimi "
            "bir yıl (12 ay) fazla olan çalışan için yeni modelin tahmin ettiği ücret farkı **(2)** sent/saattir. (Ara "
            "adımlarda yuvarlamadan, virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(4.51, 0.01, "4,51"), NumberBlank(54.14, 0.02, "54,14"))),
        explanation=(
            "Y'nin birimi 100 ile çarpılınca katsayılar 100 ile çarpılır; X'in birimi 12 ile çarpılınca (1 yıl = 12 "
            "ay) eğim 12'ye bölünür: 0,5414 × 100/12 ≈ 4,51 (sent/saat)/ay. Bir yıllık fark 12 × 4,5117 ≈ 54,14 sent, "
            "yani yine 0,5414 dolardır: ekonomik ilişki değişmez, yalnız "
            "yazıldığı birim değişir. $R^2$ de aynı kalır (§4.7, Tablo 4.1)."
        ),
    ),
    Question(
        key="b05", concept="log-duzey-tahmin-ve-yaklasik-fark", note=_note("4.10", "Kod 4.4"),
        prompt=(
            "Kod 4.4'teki katsayılara göre eğitimi 12 yıl olan bir çalışanın tahmin edilen log ücreti **(1)** olur. "
            "Log farkına dayalı yaklaşımla, eğitimi 14 yıl olan bir çalışanın tahmini saatlik ücreti 12 yıl "
            "olanınkinden yaklaşık yüzde **(2)** daha yüksektir. (Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(1.58, 0.01, "1,58"), NumberBlank(16.54, 0.02, "16,54"))),
        explanation=(
            "Tahmin edilen log ücret: 0,5838 + 0,0827 × 12 ≈ 1,58. İki yıllık eğitim farkı log ücrette 2 × 0,0827 = "
            "0,1654'lük farka karşılık gelir; bunun 100 katı yaklaşık yüzde 16,54'tür (Tablo 4.2). Çıktıdaki `std "
            "err`, `t` ve `P>|t|` sütunları bu bölümde yorumlanmaz (§4.10, Kod 4.4)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="tahmin-ortalamasindan-x-ortalamasi", note=_note("4.1", "(4.2)", "(4.3)"),
        prompt=(
            "Sabit terimli EKK ile $\\widehat{Y} = b_0 + b_1 X$ doğrusu tahmin edilmiştir; tahmin edilen değerlerin "
            "örneklem ortalaması $q$ olarak veriliyor. Açıklayıcı değişkenin "
            "örneklem ortalamasını $b_0$, $b_1$ ve $q$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\bar{X}",
            symbols=(
                Symbol("b0", "b_0", "sabit tahmini", -5, 5, aliases=("b_0",)),
                Symbol("b1", "b_1", "eğim tahmini", 0.2, 2, aliases=("b_1",)),
                Symbol("q", "q", "tahmin edilen değerlerin ortalaması", 1, 20, aliases=bar_aliases("Y")),
            ),
            answer="(q - b0)/b1",
            shown="\\frac{q - b_0}{b_1}",
        ),
        explanation=(
            "(4.2)'ye göre tahmin edilen değerlerin ortalaması $\\bar{Y}$ ile aynıdır; (4.3)'e göre doğru $(\\bar{X}, "
            "\\bar{Y})$ noktasından geçer. Buradan $q = b_0 + b_1 \\bar{X}$ ve $\\bar{X} = (q - b_0)/b_1$. WAGE1'de "
            "$(5{,}896 + 0{,}9049)/0{,}5414 \\approx 12{,}56$ yıl, yani ortalama eğitim süresidir (§4.1)."
        ),
    ),
    Question(
        key="e02", concept="model-kaynakli-sapma-formulu", note=_note("4.3", "(4.4)", "§4.5"),
        prompt=(
            "Sabit terimli basit regresyonda $\\widehat{Y}_i = \\widehat{\\beta}_0 + \\widehat{\\beta}_1 X_i$ olsun. "
            "Bir gözlemin model kaynaklı sapmasını, $\\widehat{Y}_i - "
            "\\bar{Y}$, eğim tahmini ile $X_i$ ve $\\bar{X}$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat{Y}_i - \\bar{Y}",
            symbols=(
                Symbol("b1", "\\widehat{\\beta}_1", "eğim tahmini", 0.1, 2,
                       aliases=beta_hat_aliases(1) + beta_aliases(1) + ("b_1",)),
                Symbol("x", "X_i", "gözlemin X değeri", 0, 20, aliases=("X_i", "x_i", "Xi", "xi")),
                Symbol("xbar", "\\bar{X}", "X'in örneklem ortalaması", 1, 20, aliases=bar_aliases("X")),
            ),
            answer="b1*(x - xbar)",
            shown="\\widehat{\\beta}_1\\,(X_i - \\bar{X})",
        ),
        explanation=(
            "(3.9)'dan $\\widehat{\\beta}_0 = \\bar{Y} - \\widehat{\\beta}_1 \\bar{X}$; bunu tahmin edilen doğruya "
            "yerleştirmek $\\widehat{Y}_i - \\bar{Y} = \\widehat{\\beta}_1 (X_i - \\bar{X})$ verir. Model kaynaklı "
            "sapma, gözlemin X'te ortalamadan uzaklığıyla orantılıdır; X değeri ortalamaya eşit olan gözlemde "
            "sıfırdır. Bu eşitlik §4.5'teki $R^2 = r_{XY}^2$ türetiminin ilk adımıdır (§4.3, (4.4))."
        ),
    ),
    Question(
        key="e03", concept="hkt-korelasyonla", note=_note("4.5", "(4.6)"),
        prompt=(
            "Sabit terimli basit regresyonda toplam kareler toplamı TKT, X ile Y arasındaki örneklem korelasyonu "
            "$r_{XY}$ olsun. Artık kareleri toplamını (HKT) bu iki büyüklük cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\text{HKT}",
            symbols=(
                Symbol("tkt", "\\text{TKT}", "toplam kareler toplamı", 10, 1000, aliases=("\\textTKT", "TKT")),
                Symbol("r", "r_{XY}", "örneklem korelasyonu", -0.95, 0.95, aliases=("r_XY", "r_xy", "r_YX", "r_yx")),
            ),
            answer="tkt*(1 - r^2)",
            shown="\\text{TKT}\\,(1 - r_{XY}^2)",
        ),
        explanation=(
            "(4.6)'ya göre $R^2 = 1 - \\text{HKT}/\\text{TKT}$; §4.5'e göre sabit terimli basit regresyonda $R^2 = "
            "r_{XY}^2$. İkisi birleşince $\\text{HKT} = \\text{TKT}\\,(1 - r_{XY}^2)$. WAGE1'de $7160{,}414 \\times (1 "
            "- 0{,}4059^2) \\approx 5980{,}7$; bu, Kod 4.2'deki "
            "HKT'dir. Korelasyon ±1 ise bütün artıklar sıfırdır (§4.5)."
        ),
    ),
    Question(
        key="e04", concept="birim-degisince-sabit", note=_note("4.7", "Tablo 4.1"),
        prompt=(
            "Bir düzey–düzey modelde sabit tahmini $b_0$, eğim tahmini $b_1$'dir. Bağımlı değişken $c$ ile, açıklayıcı "
            "değişken $k$ ile çarpılarak yeni birimlere çevriliyor (ör. $c = 1000$: bin TL'den TL'ye). Yeni modelin "
            "sabit tahminini yazın; gerekmeyen sembolleri kullanmayın."
        ),
        answer=Equation(
            lhs="\\tilde{b}_0",
            symbols=(
                Symbol("b0", "b_0", "eski sabit", -5, 5, aliases=("b_0",)),
                Symbol("b1", "b_1", "eski eğim", 0.1, 2, aliases=("b_1",)),
                Symbol("c", "c", "bağımlı değişkenin çarpanı", 0.5, 1000),
                Symbol("k", "k", "açıklayıcı değişkenin çarpanı", 0.5, 100),
            ),
            answer="c*b0",
            shown="c\\,b_0",
        ),
        explanation=(
            "Yeni birimlerde tahmin edilen doğru $c\\widehat{Y} = c\\,b_0 + (c\\,b_1/k)(kX)$ biçimini alır: yeni sabit "
            "$c\\,b_0$, yeni eğim $c\\,b_1/k$ olur. Sabit, X = 0'daki tahmin olduğu için yalnız Y'nin biriminden "
            "etkilenir. Tablo 4.1'de X'in birimi 100'e bölündüğünde sabitin değişmemesi bu yüzdendir (§4.7, Tablo 4.1)."
        ),
    ),
    Question(
        key="e05", concept="log-log-yuzde-degisim", note=_note("4.9", "(4.9)"),
        prompt=(
            "Log–log modelde $\\ln(Y) = \\beta_0 + \\beta_1 \\ln(X) + u$ eğim tahmini $b$'dir. X yüzde $p$ (küçük bir "
            "değer) arttığında Y'deki yaklaşık yüzde değişimi yazın."
        ),
        answer=Equation(
            lhs="\\%\\Delta Y \\approx",
            symbols=(Symbol("b", "b", "esneklik tahmini", 0.1, 2), Symbol("p", "p", "X'teki yüzde değişim", 1, 10)),
            answer="b*p",
            shown="b\\,p",
        ),
        explanation=(
            "Log–log modelde eğim esnekliktir: X yüzde 1 arttığında Y yaklaşık yüzde b değişir; küçük bir yüzde p için "
            "yaklaşık yüzde b·p. HPRICE1'de büyüklük yüzde 10 daha yüksek olduğunda fiyat yaklaşık yüzde 0,8727 × 10 ≈ "
            "8,73 daha yüksektir. Değişim büyüdükçe yaklaşım bozulur (§4.9, (4.9))."
        ),
    ),
)


KONU04_QUIZ = QuestionSet(
    topic_key="konu04",
    title="Konu 4: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"4.{number}" for number in range(1, 13)),  # §4.13 bölüm özetidir
)
