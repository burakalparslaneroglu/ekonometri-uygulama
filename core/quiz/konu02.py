"""Konu 2 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 2.1–2.13 ve Mini Quiz
2.1–2.13 maddelerini tekrar etmez: 2.000 çalışan, Türkiye enflasyonu, hane anketleri, 300 firma, 81 il, 1.500 hane,
50 banka ve ihracat anketlerinin sınıflandırılması; WAGE1'in satır sıralaması; PHILLIPS'teki −1,2 enflasyonu ve
aylık–yıllık veri; CPS78_85'in dönem sayısı ve yıl göstergesi; WAGEPAN'ın gözlem birimi ve beş yıllık gözlem;
çalışma süresi, kura, reklam, gübre ve kamu borcu örneklerinin veri üretim biçimi; dondurma–boğulma, polis–suç,
eğitim–ücret ve reklam–satış ilişkileri; büyük firma, suç oranı, gönüllü sağlık programı ve varlıklı aile örnekleri;
JTRAIN2'nin gözlem sayıları, ortalamaları ve farkı; üç iddia seçimi ve çevrim içi eğitim makalesi. Aynı becerileri
yeni bağlamlarla ve yeni sayılarla sınar. Bölüm 2'de regresyon ve test yoktur.
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
        key="k01", concept="ilk-bakis-sorulari", note=_note("2.1"),
        prompt="Bir veri dosyasına ilk bakışta sorulacak sorular arasında aşağıdakilerden hangisi yer almaz?",
        answer=MultipleChoice(
            (
                "Her satır kimi veya neyi temsil ediyor?",
                "Satırların sırası ekonomik bilgi taşıyor mu?",
                "Aynı birim birden fazla kez gözlenmiş mi?",
                "Hangi regresyon komutu çalıştırılacak?",
            ),
            correct=3,
        ),
        explanation=(
            "İlk soru komut değil, dosyanın nasıl oluştuğudur: satırın temsil ettiği birim, ölçümlerin dönemi, "
            "birimlerin tekrar gözlenip gözlenmediği, satır sırasının anlamı ve değişkenlerin ölçümü (§2.1)."
        ),
    ),
    Question(
        key="k02", concept="yatay-kesiti-tanima", note=_note("2.2"),
        prompt="Aşağıdaki veri setlerinden hangisi yatay kesit verisidir?",
        answer=MultipleChoice(
            (
                "Bir devlet hastanesinin 2000–2024 dönemindeki yıllık ameliyat sayısı",
                "2024 yılında 120 belediyenin nüfusu, bütçesi ve personel sayısı",
                "Aynı 120 belediyenin 2019–2024 dönemindeki yıllık bütçesi",
                "2019 ve 2024 yıllarında farklı belediye örneklemlerinden toplanan bütçe verisi",
            ),
            correct=1,
        ),
        explanation=(
            "Yatay kesitte çok sayıda birim belirli bir zamanda bir kez gözlenir. Diğerleri sırasıyla zaman serisi, "
            "panel ve havuzlanmış yatay kesittir (§2.2; karşılaştırma: §2.6, Tablo 2.2)."
        ),
    ),
    Question(
        key="k03", concept="yapi-ve-uretim-birlikte", note=_note("2.7"),
        prompt=(
            "Bir belediye 300 mahalleyi kura ile iki gruba ayırıyor, bir gruba ücretsiz internet veriyor ve iki grubun "
            "mahallelerini 2022–2025 boyunca her yıl izliyor. Veri yapısı ve veri üretim biçimi nedir?"
        ),
        answer=MultipleChoice(
            (
                "Havuzlanmış yatay kesit, gözlemsel",
                "Yatay kesit, deneysel",
                "Panel, deneysel",
                "Panel, gözlemsel",
            ),
            correct=2,
        ),
        explanation=(
            "Aynı mahalleler birden fazla yıl izlendiği için veri paneldir; müdahale kura ile atandığı için deneyseldir. "
            "Veri yapısı ve veri üretim biçimi iki ayrı sorudur (§2.7)."
        ),
    ),
    Question(
        key="k04", concept="ceteris-paribus-yollari", note=_note("2.9"),
        prompt="Notlara göre aşağıdakilerden hangisi *ceteris paribus* hedefine yaklaşmanın yollarından biri değildir?",
        answer=MultipleChoice(
            (
                "Müdahaleyi rastgele atamak",
                "Birbirine benzer birimleri ya da uygun karşılaştırma gruplarını seçmek",
                "Gözlenen diğer özellikleri modele eklemek",
                "Örneklemi büyütmek",
            ),
            correct=3,
        ),
        explanation=(
            "Üç yol deneysel karşılaştırma, araştırma tasarımı ve ekonometrik modeldir (§2.9). Örneklemi büyütmek "
            "karşılaştırılan grupların başka koşullarda farklı olması sorununu tek başına ortadan kaldırmaz: büyük bir "
            "örneklem nedensellik sorunlarını otomatik olarak çözmez (§2.2)."
        ),
    ),
    Question(
        key="k05", concept="secilimi-tanima", note=_note("2.10", "Tablo 2.3"),
        prompt=(
            "Bir üniversite mezun memnuniyetini yalnız mezuniyet törenine katılan öğrencilere anket uygulayarak "
            "ölçüyor. Bu tasarımda en belirgin sorun hangisidir?"
        ),
        answer=MultipleChoice(("Karıştırıcı faktör", "Ters nedensellik", "Seçilim", "Örneklemin küçük olması"),
                              correct=2),
        explanation=(
            "Analize girenler rastgele olmayan bir süreçle belirlenir: törene katılanlar memnuniyeti yüksek "
            "öğrenciler olabilir. Bu süreç sonuçla ilişkili olduğu için seçilim söz konusudur (§2.10)."
        ),
    ),
    Question(
        key="k06", concept="deneyde-baslica-kaygi", note=_note("2.12", "Tablo 2.4"),
        prompt=(
            "Bir il, kura ile seçilen 40 okulda ücretsiz kahvaltı uyguluyor ve bu okulların sınav başarısını kurada "
            "seçilmeyen okullarla karşılaştırıyor. Tablo 2.4'e göre bu tür bir karşılaştırmada başlıca kaygı nedir?"
        ),
        answer=MultipleChoice(
            (
                "Grupların başka özelliklerde farklı olması",
                "Atamanın uygulanması ve genellenebilirlik",
                "Zaman sırasının bozulması",
                "Aynı kişilerin tekrar gözlenmesi",
            ),
            correct=1,
        ),
        explanation=(
            "Kura grupların sistematik başlangıç farklarını azaltmayı hedefler; kaygı kuranın gerçekten uygulanıp "
            "uygulanmadığı (ör. seçilmeyen okullara da kahvaltı verilmesi) ve sonucun başka illere genellenip "
            "genellenemeyeceğidir. Grupların başka özelliklerde farklı olması gözlemsel karşılaştırmaların (Tablo "
            "2.4'te WAGE1) başlıca kaygısıdır (§2.12, Tablo 2.4)."
        ),
    ),
    Question(
        key="k07", concept="kontrol-listesinin-ilk-adimlari", note=_note("2.13"),
        prompt="Veri veya makale okuma kontrol listesinin ilk iki sorusu hangileridir?",
        answer=MultipleChoice(
            (
                "Gözlem birimi nedir? Veri yapısı nedir?",
                "Ters nedensellik mümkün mü? Seçilim olabilir mi?",
                "Katsayı kaçtır? Yanında yıldız var mı?",
                "Sonuç hangi kurumlar için geçerlidir? Dönem nedir?",
            ),
            correct=0,
        ),
        explanation=(
            "Liste gözlem birimi ve veri yapısıyla başlar; dönem, tekrar gözlem, veri üretimi, iddianın türü, "
            "karşılaştırma, ters nedensellik, seçilim ve genellenebilirlik bunları izler (§2.13)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="panelde-degisen-ve-degismeyen", note=_note("2.5"),
        prompt=(
            "Panel veride aynı kişinin zaman içinde değişmeyen özellikleri (ör. doğum yeri) ile değişen özellikleri "
            "(ör. deneyim) ayrı ayrı düşünülebilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Aynı kişinin tekrar gözlenmesi, değişmeyen özellikler ile zaman içinde değişen özellikleri ayırmaya "
            "yardım eder. Yine de panel veri tek başına bütün karşılaştırma sorunlarını çözmez (§2.5)."
        ),
    ),
    Question(
        key="d02", concept="id-ve-yil-panel-kaniti-degil", note=_note("2.6", "Tablo 2.2"),
        prompt=(
            "2019 ve 2023 yıllarına ait 3.000 satır ve 3.000 farklı kimlik (`id`) içeren bir hane dosyası panel "
            "veridir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Panelde aynı birimler birden fazla dönemde gözlenir; burada her kimlik yalnız bir kez görünür. Farklı "
            "yılların verisi birleştirilmiş ama her yılda farklı haneler var: veri havuzlanmış yatay kesittir "
            "(§2.6, Tablo 2.2)."
        ),
    ),
    Question(
        key="d03", concept="baslangic-farki-sonuca-karisir", note=_note("2.8"),
        prompt=(
            "İki grup başlangıçta farklıysa, gruplar arasındaki sonuç farkı hem müdahaleyi hem de başlangıç "
            "farklarını yansıtabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Korelasyon veya basit grup farkı nedensel karşılaştırmayı kendiliğinden sağlamaz. Aynı kişinin müdahaleli "
            "ve müdahalesiz durumu birlikte gözlenemediği için uygun karşılaştırma grubu gerekir (§2.8)."
        ),
    ),
    Question(
        key="d04", concept="bin-birimini-okuma", note=_note("2.11", "Kod 2.7"),
        prompt="Bir hane anketinde aylık gelir bin TL cinsinden ölçülmüştür; iki grup arasındaki 2,35'lik fark 2.350 TL'dir.",
        answer=TrueFalse(True),
        explanation=(
            "Değişken bin TL birimindeyse 2,35 × 1.000 = 2.350 TL. JTRAIN2'de `re78` de bin dolar birimindedir: 1,794 "
            "bin dolar yaklaşık 1.794 dolardır. Fark yorumlanırken birim açıkça yazılır (§2.11)."
        ),
    ),
    Question(
        key="d05", concept="kontrol-grubu-atamayla-olusur", note=_note("2.11"),
        prompt="JTRAIN2'de kontrol grubu (`train` = 0), programa katılmak istemeyen kişilerden oluşur.",
        answer=TrueFalse(False),
        explanation=(
            "JTRAIN2 rastgele atamalı bir deneydir: `train` = 1 programa, `train` = 0 kontrol grubuna atananları "
            "gösterir. Kimin hangi gruba gireceğini kişilerin tercihi değil atama belirler; gönüllülüğe dayalı gruplar "
            "seçilim sorununa yol açabilirdi (§2.11, §2.7, §2.10)."
        ),
    ),
    Question(
        key="d06", concept="cari-fiyatla-donem-farki", note=_note("2.4", "Kod 2.4"),
        prompt=(
            "CPS78_85'te saatlik ücretler cari fiyatlarla ölçüldüğü için 1978 ve 1985 ortalamaları arasındaki fark "
            "fiyat düzeyindeki değişimi de içerebilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "İki dönemin ortalamaları farklı kişilerden ve farklı fiyat düzeylerinden gelir. Farkın tamamını bir "
            "politikaya bağlamak için fiyat düzeyi, çalışan bileşimi ve ekonomik koşullar da düşünülür (§2.4)."
        ),
    ),
    Question(
        key="d07", concept="talep-kanunu-ceteris-paribus", note=_note("2.9"),
        prompt=(
            "Talep kanunu, fiyat yükseldiğinde talep edilen miktarın gelir ve tercihler de değişse bile her durumda "
            "azalacağını söyler."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Talep kanunu bir *ceteris paribus* ifadesidir: diğer ilgili koşullar aynıyken fiyat artışı talep edilen "
            "miktarı azaltır. Gelir veya tercihler de değişiyorsa talepteki değişimi yalnız fiyata bağlamak güçleşir "
            "(§2.9)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="ardisik-yillar-arasi-degisim", note=_note("2.3", "Kod 2.2"),
        prompt=(
            "Kod 2.2'ye göre 1950'den 1951'e enflasyon oranı **(1)** yüzde puan, işsizlik oranı **(2)** yüzde puan "
            "değişmiştir (artışı pozitif, azalışı negatif yazın)."
        ),
        answer=FillBlanks((NumberBlank(6.6, 0.05, "6,6"), NumberBlank(-2.0, 0.05, "−2,0"))),
        explanation=(
            "Enflasyon 1,3'ten 7,9'a: 7,9 − 1,3 = 6,6 yüzde puan; işsizlik 5,3'ten 3,3'e: −2,0 yüzde puan. Yıldan yıla "
            "değişim ancak satırlar zaman sırasındayken anlamlıdır (§2.3)."
        ),
    ),
    Question(
        key="b02", concept="havuzlanmis-orneklem-buyuklugu", note=_note("2.4", "Kod 2.4"),
        prompt=(
            "CPS78_85'te 1978 örnekleminde 550, 1985 örnekleminde 534 çalışan vardır. Havuzlanmış veri setindeki "
            "toplam gözlem sayısı **(1)**, 1985 gözlemlerinin payı yüzde **(2)** olur. (Yüzdeyi virgülden sonra iki "
            "basamakla yazın.)"
        ),
        answer=FillBlanks((NumberBlank(1084, 0.5, "1.084"), NumberBlank(49.26, 0.01, "49,26"))),
        explanation=(
            "İki yatay kesit tek dosyada birleştirilir: 550 + 534 = 1.084 satır. 534/1.084 × 100 ≈ %49,26. Satırlar "
            "iki ayrı örneklemden gelir; aynı çalışan iki dönemde izlenmez (§2.4)."
        ),
    ),
    Question(
        key="b03", concept="grup-farkinin-goreli-buyuklugu", note=_note("2.11", "Kod 2.7"),
        prompt=(
            "JTRAIN2'de gözlenen 1,794 bin dolarlık fark, kontrol grubunun ortalama kazancının (4,555 bin dolar) "
            "yüzde **(1)** kadarıdır. Eğitim grubundaki 185 kişi, 445 kişilik örneklemin yüzde **(2)** kadarıdır. "
            "(Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(39.39, 0.01, "39,39"), NumberBlank(41.57, 0.01, "41,57"))),
        explanation=(
            "1,794/4,555 × 100 ≈ %39,39: fark, kontrol grubunun ortalama kazancının yaklaşık %39'u kadardır. "
            "185/445 × 100 ≈ %41,57. Farkın kendisi bin dolar birimindedir; oranlar birimsizdir (§2.11)."
        ),
    ),
    Question(
        key="b04", concept="mekanizmalarin-adlari", note=_note("2.10", "Tablo 2.3"),
        prompt=(
            "Hem açıklayıcı değişkenle hem sonuç değişkeniyle ilişkili üçüncü bir unsura **(1)** denir. Sonuç "
            "değişkeninin açıklayıcı değişkeni de etkileyebilmesine **(2)** denir."
        ),
        answer=FillBlanks((
            TextBlank(("karıştırıcı faktör", "karıştırıcı", "karıştırıcı değişken", "karıştırıcı etken",
                       "karıştırıcı unsur", "karıştırıcı faktörler", "confounding factor", "confounding variable",
                       "confounder"), "karıştırıcı faktör"),
            TextBlank(("ters nedensellik", "ters nedensellik sorunu", "ters nedensel", "ters yönlü nedensellik",
                       "karşılıklı nedensellik", "çift yönlü nedensellik", "reverse causality", "reverse causation"),
                      "ters nedensellik"),
        )),
        explanation=(
            "Karıştırıcı faktör gözlenen ilişkinin yorumunu karıştıran üçüncü unsurdur (ör. yetenek). Ters "
            "nedensellikte ilişkinin yönü sorundur: Y de X'i etkileyebilir (§2.10, Tablo 2.3)."
        ),
    ),
    Question(
        key="b05", concept="karar-agacini-uygulama", note=_note("2.1", "Şekil 2.1"),
        prompt=(
            "Karar ağacına göre birimler tekrar gözlenmiyor ve farklı dönemlerde farklı örneklemler seçilmişse veri "
            "**(1)** verisidir; tek bir birimin (ör. bir ülke ekonomisinin) ardışık dönemlerdeki seyri izleniyorsa "
            "veri **(2)** verisidir."
        ),
        answer=FillBlanks((
            TextBlank(("havuzlanmış yatay kesit", "havuzlanmış yatay kesit verisi", "havuzlanmış kesit", "havuzlanmış",
                       "havuzlanmış yatay kesitler", "pooled cross section", "pooled cross sections"),
                      "havuzlanmış yatay kesit"),
            TextBlank(("zaman serisi", "zaman serisi verisi", "zaman serileri", "time series"), "zaman serisi"),
        )),
        explanation=(
            "Ağacın ilk sorusu birimlerin tekrar gözlenip gözlenmediğidir. Tekrar yoksa ve dönemler farklıysa her "
            "dönemin örneklemi ayrıdır: havuzlanmış yatay kesit. Tek birimin ya da toplulaştırılmış bir değişkenin "
            "ardışık dönemleri zaman serisidir (§2.1, Şekil 2.1)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="log-ucretten-ucrete", note=_note("2.4", "Kod 2.3", "Kod 2.4"),
        prompt="Veri setinde saatlik ücretin doğal logaritması $l$ bulunuyor. Saatlik ücreti $l$ cinsinden yazın.",
        answer=Equation(
            lhs="\\text{wage}",
            symbols=(Symbol("l", "l", "saatlik ücretin doğal logaritması", 0.5, 3.0),),
            answer="exp(l)",
            shown="\\exp(l)",
        ),
        explanation=(
            "Doğal logaritmanın tersi üstel fonksiyondur: `wage` = exp(`lwage`). CPS78_85'te saatlik ücret bu yolla "
            "türetilir; dönem ortalamaları saatlik 6,06 ve 9,02 ABD dolarıdır (cari fiyatlarla; §2.4, Kod 2.3–2.4)."
        ),
    ),
    Question(
        key="e02", concept="havuzlanmis-genel-ortalama", note=_note("2.4", "Kod 2.4"),
        prompt=(
            "Havuzlanmış bir veri setinde birinci dönemin gözlem sayısı $n_1$ ve ortalaması $m_1$, ikinci dönemin "
            "gözlem sayısı $n_2$ ve ortalaması $m_2$'dir. Bütün gözlemlerin ortalamasını yazın."
        ),
        answer=Equation(
            lhs="\\text{genel ortalama}",
            symbols=(
                Symbol("n1", "n_1", "birinci dönemin gözlem sayısı", 100, 1000, aliases=("n_1",)),
                Symbol("m1", "m_1", "birinci dönemin ortalaması", 3, 10, aliases=("m_1",)),
                Symbol("n2", "n_2", "ikinci dönemin gözlem sayısı", 100, 1000, aliases=("n_2",)),
                Symbol("m2", "m_2", "ikinci dönemin ortalaması", 3, 10, aliases=("m_2",)),
            ),
            answer="(n1*m1 + n2*m2)/(n1 + n2)",
            shown="\\frac{n_1 m_1 + n_2 m_2}{n_1 + n_2}",
        ),
        explanation=(
            "Genel ortalama, dönem ortalamalarının gözlem sayılarıyla ağırlıklı ortalamasıdır: toplamlar n₁m₁ ve "
            "n₂m₂'dir. CPS78_85'te (550 × 6,06 + 534 × 9,02)/1.084 ≈ 7,52 ABD doları/saat (cari fiyatlarla) (§2.4)."
        ),
    ),
    Question(
        key="e03", concept="paylar-arasi-yuzde-puan-farki", note=_note("2.11"),
        prompt=(
            "Bir deneyde programa atanan grupta işsiz kalanların payı $p_1$, kontrol grubunda $p_0$'dır (0 ile 1 "
            "arasında). Programa atanan gruptaki paydan kontrol grubundaki payı çıkararak farkı yüzde puan olarak yazın."
        ),
        answer=Equation(
            lhs="\\text{fark (yüzde puan)}",
            symbols=(
                Symbol("p1", "p_1", "program grubundaki pay", 0.05, 0.95, aliases=("p_1",)),
                Symbol("p0", "p_0", "kontrol grubundaki pay", 0.05, 0.95, aliases=("p_0",)),
            ),
            answer="100*(p1 - p0)",
            shown="100\\,(p_1 - p_0)",
        ),
        explanation=(
            "Paylar yüzdeye 100 ile çarpılarak çevrilir; iki yüzdenin farkı yüzde puandır. Ör. pay programa atanan "
            "grupta 0,35, kontrol grubunda 0,30 ise fark 100 × (0,35 − 0,30) = 5 yüzde puandır; göreli fark ise "
            "yaklaşık %16,7'dir (yüzde puan: §0.6.2; deney bağlamı: §2.11)."
        ),
    ),
    Question(
        key="e04", concept="yillik-seride-gozlem-sayisi", note=_note("2.3"),
        prompt=(
            "Yıllık bir zaman serisi $a$ yılında başlıyor ve $b$ yılında bitiyor; her yıl bir gözlem var. Gözlem "
            "sayısını yazın."
        ),
        answer=Equation(
            lhs="T",
            symbols=(Symbol("a", "a", "ilk yıl", 1900, 1950), Symbol("b", "b", "son yıl", 1960, 2025)),
            answer="b - a + 1",
            shown="b - a + 1",
        ),
        explanation=(
            "İlk ve son yıl dahil sayılır: T = b − a + 1. PHILLIPS 1948–2003 dönemini kapsar: 2003 − 1948 + 1 = 56 "
            "gözlem (yıl) (§2.3)."
        ),
    ),
    Question(
        key="e05", concept="dengeli-panel-satir-sayisi", note=_note("2.5"),
        prompt=(
            "Dengeli bir panelde $N$ birimin her biri $T$ dönem gözleniyor. Veri setindeki satır (gözlem) sayısını "
            "yazın."
        ),
        answer=Equation(
            lhs="\\text{satır sayısı}",
            symbols=(
                Symbol("n", "N", "birim sayısı", 10, 600, aliases=("N",)),
                Symbol("t", "T", "dönem sayısı", 2, 20, aliases=("T",)),
            ),
            answer="n*t",
            shown="N\\,T",
        ),
        explanation=(
            "Her satır bir birim–dönem birleşimidir; dengeli panelde her birim aynı sayıda dönemde gözlenir. WAGEPAN: "
            "545 çalışan × 8 yıl = 4.360 satır (§2.5; sayılar: Veri Setleri Rehberi)."
        ),
    ),
)


KONU02_QUIZ = QuestionSet(
    topic_key="konu02",
    title="Konu 2: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir "
        "bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki "
        "egzersiz ve mini quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"2.{number}" for number in range(1, 14)),  # §2.14 bölüm özetidir
)
