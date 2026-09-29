"""Konu 5 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 5.1–5.12 ve Mini Quiz
5.1–5.12 maddelerini tekrar etmez: dört araştırma sorusunda sonuç ve temel değişken; konut, satış ve sınav modellerini
notasyonla kurma; satış = 18 + 2,5 reklam − 1,2 fiyat + 0,8 firma büyüklüğü yorumları; Ŷ = 4 + 1,5X₁ − 0,8X₂ + 2X₃
hesapları; sınav notu için yardımcı regresyonlar; Tablo 5.1'in yorum soruları (eğitim dört, kıdem üç yıl fark
dahil); 0,50'den 0,62'ye yükselen eğitim katsayısı; Denklem 5.7'de 250 kare fit, 1.500 kare fit arsa ve 200 kare fit
artı bir oda hesapları; A, B ve C modellerinin R² tablosu; Tablo 5.4'ün eleştirel okunması; dört soruda kontrol
gerekçesi; satış = 25 + 1,6 reklam − 0,9 fiyat + 0,04 çalışan bütünleşik uygulaması ve mini quizlerdeki sorular. Aynı
becerileri yeni bağlamlarla ve yeni sayılarla sınar. Eksik değişken yanlılığının formülü ve çoklu doğrusal bağlantı
Konu 6'nın; standart hata, t, p-değeri ve güven aralığı Konu 7'nin; logaritmik modellerde tam yüzde değişim Konu 9'un
konusudur. Sorular yalnız küçük değişim yaklaşımını kullanır.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol, bar_aliases, beta_aliases
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


def _bar_sub(letter: str, index: int) -> tuple[str, ...]:
    """Alt indisli ortalama için yazımlar: X̄₁, \\bar{X}_1, \\overline{X}_1, X_1bar, x1bar (büyük ve küçük harf)."""

    forms: list[str] = []
    for item in (letter.upper(), letter.lower()):
        for bar in (f"\\bar{item}", f"\\bar {item}", f"\\overline{item}", f"\\overline {item}", f"{item}̄"):
            forms += [f"{bar}_{index}", f"{bar}{index}", f"{bar}{'₀₁₂₃'[index]}"]
        forms += [f"{item}_{index}bar", f"{item}{index}bar", f"{item}bar_{index}", f"{item}bar{index}",
                  f"{item}{index}_bar", f"{item}_{index}_bar"]
    return tuple(forms)


def _sub(letter: str, index: int) -> tuple[str, ...]:
    """Alt indisli sembolün yazımları: g_1, g₁ (ad ``g1`` doğrudan okunur)."""

    return (f"{letter}_{index}", f"{letter}{'₀₁₂₃'[index]}")


_R2 = ("R^2", "R²", "R2", "r^2", "r²")
"""R² için yazımlar; ``R^2`` bir R sembolünün karesi olarak okunmasın diye tek sembole çevrilir."""


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="basit-regresyonun-cevapladigi-soru", note=_note("5.1"),
        prompt=(
            "Bir örneklemdeki 2.000 hanede aylık tüketim yalnız aylık gelirle açıklanmıştır: "
            "$\\widehat{\\text{tüketim}} = 1800 + 0{,}62\\,\\text{gelir}$ (TL). Bu basit regresyon aşağıdaki "
            "sorulardan hangisine doğrudan cevap verir?"
        ),
        answer=MultipleChoice(
            (
                "Serveti ve hane büyüklüğü aynı olan haneler arasında gelir farkı, örneklemde tüketim farkıyla "
                "nasıl ilişkilidir?",
                "Geliri 1 TL yüksek hanelerin tüketimi, servet ve hane büyüklüğü farkları ayrılmadan, ortalama ne "
                "kadar farklıdır?",
                "Bir hanenin geliri 1 TL artırılırsa aynı hanenin aylık tüketimi ne kadar artar?",
                "Harcama alışkanlıkları ve geleceğe verdiği önem aynı olan haneler arasında gelir ile tüketim nasıl "
                "ilişkilidir?",
            ),
            correct=1,
        ),
        explanation=(
            "Basit regresyon, geliri farklı bütün haneleri diğer farklarını ayırmadan karşılaştırır: eğim, iki "
            "değişkenin örneklemde birlikte nasıl hareket ettiğini özetler. Servet ve hane büyüklüğünü sabit tutan "
            "karşılaştırma çoklu regresyon gerektirir; aynı hanenin gelirinin artırılması nedensel bir sorudur; "
            "harcama alışkanlıkları ve geleceğe verilen önem gözlenmediği için dördüncü soruyu iki model de "
            "cevaplayamaz. Basit model yanlış olmak "
            "zorunda değildir, cevapladığı soru sınırlıdır (§5.1)."
        ),
    ),
    Question(
        key="k02", concept="ceteris-paribus-cumlesinin-uc-unsuru", note=_note("5.3"),
        prompt=(
            "400 kiralık dairede $\\widehat{\\text{kira}} = 1500 + 42\\,\\text{alan} - 25\\,\\text{bina yaşı} + "
            "310\\,\\text{oda}$ tahmin edilmiştir (kira TL/ay, alan m², bina yaşı yıl). Bina yaşı katsayısının doğru "
            "yorumu hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Alan ve oda sayısı aynıyken, bina yaşı bir yıl daha fazla olan dairelerin tahmin edilen aylık kirası "
                "yaklaşık 25 TL daha düşüktür",
                "Bir binanın yaşı her yıl bir artarken o binadaki bir dairenin kirası her yıl 25 TL düşer",
                "Alan ve oda sayısı aynıyken, bina yaşı bir yıl daha fazla olan dairelerin tahmin edilen aylık kirası "
                "yaklaşık yüzde 25 daha düşüktür",
                "Alan ve oda farkları ne olursa olsun, bina yaşı bir yıl fazla olan dairelerin kirası 25 TL daha "
                "düşüktür",
            ),
            correct=0,
        ),
        explanation=(
            "Ceteris paribus yorumda üç unsur bulunur: değişen değişken ve birimi (bina yaşı, bir yıl), bağımlı "
            "değişkendeki tahmin edilen fark ve birimi (25 TL/ay) ve sabit tutulan değişkenler (alan ve oda sayısı). "
            "İkinci seçenek aynı dairenin zaman içindeki değişimini söyleyen nedensel bir cümledir; üçüncüsü birimi "
            "yanlış okur (kira düzeydir, logaritma değildir); dördüncüsü diğer değişkenleri sabit tutmaz (§5.3)."
        ),
    ),
    Question(
        key="k03", concept="henuz-yorumlanmayan-cikti-sutunlari", note=_note("5.6", "Kod 5.2"),
        prompt=(
            "Kod 5.2'deki Python çıktısını bu bölümün okuma sırasıyla okurken aşağıdakilerden hangisi yapılmaz?"
        ),
        answer=MultipleChoice(
            (
                "`coef` sütunundan eğitim, deneyim ve kıdem katsayılarını okumak ve birimleriyle yorumlamak",
                "`No. Observations` satırından modelin 526 çalışanla tahmin edildiğini görmek",
                "`P>|t|` sütunundaki 0,064'e bakarak deneyimin ücretle ilişkisiz olduğuna karar vermek",
                "`R-squared` satırındaki 0,306'yı, ücretin örneklem ortalaması çevresindeki değişkenliğinin yaklaşık "
                "yüzde 30,6'sının model tarafından izlendiği biçiminde okumak",
            ),
            correct=2,
        ),
        explanation=(
            "Bu bölümde çıktıdan bağımlı değişken, gözlem sayısı, `coef` sütunu, R² ve düzeltilmiş R² okunur. Standart "
            "hata, t, p-değeri ve güven aralığı sütunları görünür ama henüz yorumlanmaz; bunlar Konu 7'nin "
            "konusudur. Ayrıca bir p-değerinden “ilişkisiz” sonucuna varmak, o bölümde de doğru bir okuma değildir "
            "(§5.6, Kod 5.2)."
        ),
    ),
    Question(
        key="k04", concept="kontrol-eklenince-katsayi-farkli-karsilastirma", note=_note("5.7"),
        prompt=(
            "250 öğrencide sınav notu haftalık çalışma saatiyle açıklanıyor: basit modelde çalışma saati katsayısı "
            "2,4 puandır. Modele önceki dönem not ortalaması eklenince katsayı 1,1 puana düşüyor. En uygun yorum "
            "hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Basit model hatalı kurulmuştur; çalışma saatinin sınav notu üzerindeki gerçek etkisi 1,1 puandır",
                "Katsayı yarıdan fazla düştüğüne göre çalışma saati ile sınav notu arasında artık güçlü bir ilişki "
                "kalmamıştır",
                "Önceki not modele eklendiği için 1,1 artık çalışma saatinin nedensel etkisi olarak okunabilir",
                "Çoklu model önceki notu aynı olan öğrencileri karşılaştırır; basit katsayı önceki notla ortak hareketi "
                "de içerir",
            ),
            correct=3,
        ),
        explanation=(
            "İki katsayı farklı karşılaştırmalar yapar. Basit model çalışma saati farklı bütün öğrencileri "
            "karşılaştırır; çok çalışan öğrencilerin önceki notları da yüksekse basit katsayı bu ortak hareketi de "
            "içerir. Çoklu model önceki notu aynı tutar. Basit model bu yüzden hatalı değildir, başka bir "
            "karşılaştırmayı cevaplar; kontrol eklemek de 1,1'i kendiliğinden nedensel etki yapmaz (§5.7)."
        ),
    ),
    Question(
        key="k05", concept="oda-katsayisinin-karsilastirdigi-konutlar", note=_note("5.8", "(5.7)"),
        prompt=(
            "Denklem (5.7) ile dört konutun fiyatı tahmin ediliyor: P (1.800 kare fit, 3 yatak odası, 6.000 kare fit "
            "arsa), Q (1.800; 4; 6.000), R (2.100; 4; 6.000) ve S (1.800; 4; 9.000). Hangi iki konutun tahmin farkı "
            "yalnız yatak odası katsayısına (13,85 bin dolar) eşittir?"
        ),
        answer=MultipleChoice(("Q ile R", "P ile R", "P ile Q", "Q ile S"), correct=2),
        explanation=(
            "Yatak odası katsayısı, büyüklüğü ve arsası aynı konutlar arasındaki oda farkıdır: yalnız P ile Q bu "
            "koşulu sağlar. Q ile R yalnız büyüklükte (300 × 0,1228 = 36,84 bin dolar), Q ile S yalnız arsada "
            "(3.000 × 0,002068 ≈ 6,20 bin dolar) ayrılır; P ile R arasında büyüklük ve oda birlikte değiştiği için fark "
            "iki katkının toplamıdır (36,84 + 13,85 = 50,69) (§5.8, Tablo 5.3)."
        ),
    ),
    Question(
        key="k06", concept="gozlem-sayisi-farkli-sutunlar", note=_note("5.10"),
        prompt=(
            "Bir makale tablosunda (1) sütunu ücreti yalnız eğitimle açıklar (gözlem sayısı 526). (2) sütununa "
            "babanın eğitim yılı da eklenmiştir; bu bilgisi eksik çalışanlar çıkarıldığı için gözlem sayısı 412'dir. "
            "(1) ile (2) arasındaki eğitim katsayısı farkını yorumlamadan önce en önemli uyarı hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "İki sütun farklı örneklemlerle tahmin edilmiştir; fark eklenen değişkenden olduğu kadar örneklem "
                "değişiminden de kaynaklanabilir",
                "Sütun (2)'nin R²'si daha yüksekse onun eğitim katsayısı daha doğrudur; bu yüzden aradaki fark "
                "tamamen (2) lehine yorumlanmalıdır",
                "Gözlem sayısı düştüğü için (2) sütunundaki bütün katsayılar kesinlikle yanlıdır ve tablodan "
                "çıkarılmalıdır",
                "Gözlem sayısı kontrol edilmez; yalnız sütunların bağımlı değişkenlerinin aynı olup olmadığına bakılır",
            ),
            correct=0,
        ),
        explanation=(
            "Okuma sırasının ikinci adımı, gözlem sayısının modeller arasında aynı olup olmadığını kontrol etmektir. "
            "(2)'de 114 çalışan eksik olduğu için iki katsayı farklı çalışan gruplarında hesaplanmıştır; fark "
            "yalnız babanın eğitiminin kontrol edilmesine bağlanamaz. Aynı model 412 çalışanla yeniden tahmin "
            "edilerek karşılaştırma yapılabilir. R² de yalnız aynı bağımlı değişken ve aynı örneklemde "
            "karşılaştırılır (§5.9, §5.10)."
        ),
    ),
    Question(
        key="k07", concept="sonuc-niteligindeki-kontrol", note=_note("5.11"),
        prompt=(
            "200 banka şubesinde şubenin uyguladığı kredi faiz oranının aylık kredi hacmiyle ilişkisi inceleniyor. "
            "Aşağıdaki değişkenlerden hangisini kontrol olarak eklemek, faiz oranının kredi hacmiyle ilişkisinin bir "
            "bölümünü yanlışlıkla ortadan kaldırma riski taşır?"
        ),
        answer=MultipleChoice(
            (
                "Şubenin bulunduğu ilçedeki yetişkin nüfus",
                "Şubenin bulunduğu ildeki hanelerin ortalama geliri",
                "Faiz kararından önce şubede görev yapan kredi uzmanı sayısı",
                "Faiz ilanından sonraki ay yapılan kredi başvurusu sayısı",
            ),
            correct=3,
        ),
        explanation=(
            "Kredi başvurusu sayısı faiz oranının bir sonucu olabilir: düşük faiz başvuruları artırabilir ve kredi "
            "hacmi kısmen bu yolla yükselebilir. Başvuru sayısını sabit tutmak bu yolu kapatır ve araştırma sorusunun "
            "bir bölümünü dışarıda bırakabilir. Nüfus, hane geliri ve kredi uzmanı sayısı faiz kararından önce "
            "belirlenir; kontrol olarak böyle bir sorun taşımaz. Değişkenlerin zaman sırası önemlidir (§5.11)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="modele-alinan-degisken-hata-terimini-bosaltmaz", note=_note("5.2", "(5.2)"),
        prompt=(
            "Denklem (5.2)'deki ücret modelinde deneyim ve kıdem açıklayıcı değişken olarak yer aldığı için, bu iki "
            "değişkenle ilişkili hiçbir faktör hata terimi $u_i$ içinde kalamaz."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Modele alınan değişken yalnız kendisini hata teriminden çıkarır. $u_i$, modelde açıkça yer almayan "
            "bütün faktörleri (yetenek, çalışma yoğunluğu, firma özellikleri, ölçüm hataları) içerir; bunlardan "
            "bazıları deneyim ya da kıdemle ilişkili olabilir (ör. verimli firmalarda çalışanlar daha uzun kalır). "
            "Bu ilişkinin katsayılar için ne anlama geldiği Konu 6'da işlenir (§5.2)."
        ),
    ),
    Question(
        key="d02", concept="paralel-cizgiler-etkilesim-yok", note=_note("5.3", "Şekil 5.1"),
        prompt=(
            "Şekil 5.1'de deneyim düzeyi 5 yıldan 35 yıla çıktıkça eğitim–ücret tahmin çizgisinin eğimi artar."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Denklem (5.5)'te eğitim ile deneyim arasında etkileşim terimi yoktur; bu yüzden üç çizginin eğimi de "
            "0,5990'dır ve çizgiler paraleldir. Deneyim 30 yıl artınca çizgi yalnız 30 × 0,0223 ≈ 0,67 dolar yukarı "
            "kayar. Eğitim katsayısının deneyim düzeyine göre değişmesine izin veren etkileşimli modeller Konu 11'de "
            "işlenir (§5.3, Şekil 5.1)."
        ),
    ),
    Question(
        key="d03", concept="ekk-katsayilari-birlikte-en-kucuk", note=_note("5.4", "(5.6)"),
        prompt=(
            "WAGE1'de Denklem (5.5)'teki katsayılardan yalnız eğitim katsayısı 0,5990 yerine 0,62 alınır, diğer üç "
            "katsayı aynı bırakılırsa 526 çalışan için kareli artıklar toplamı büyür."
        ),
        answer=TrueFalse(True),
        explanation=(
            "EKK dört katsayıyı, (5.6)'daki kareli artıklar toplamını birlikte en küçük yapacak biçimde seçer. Bu "
            "katsayılardan herhangi birini tek başına değiştirmek toplamı büyütür: WAGE1'de eğitim katsayısı 0,62 "
            "alınınca toplam yaklaşık 38,4 büyür. Katsayılar tek tek değil, aynı ölçütü birlikte küçültecek biçimde "
            "belirlenir (§5.4)."
        ),
    ),
    Question(
        key="d04", concept="sabit-tutma-birebir-eslestirme-degil", note=_note("5.5"),
        prompt=(
            "Örneklemde deneyimi ve kıdemi birebir aynı olan hiçbir çalışan çifti bulunmasa da çoklu regresyon "
            "“deneyim ve kıdem sabitken” eğitim katsayısını hesaplayabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Çoklu regresyon birebir eşleşmiş gözlemler aramaz. Deneyim ve kıdemin doğrusal katkısını hem eğitimden "
            "hem ücretten ayırır ve kalan kısımları birbiriyle ilişkilendirir; bu yüzden deneyimi ve kıdemi aynı "
            "olan çalışan çiftleri hiç bulunmasa da katsayı hesaplanır. Bu, sabit tutmanın doğrusal model üzerinden "
            "yapıldığı anlamına da gelir (§5.5)."
        ),
    ),
    Question(
        key="d05", concept="katsayi-buyuklugune-gore-kontrol-secilmez", note=_note("5.7"),
        prompt=(
            "Bir mağaza zincirinde şubelerin aylık satışı mağaza alanıyla açıklanıyor. Modele şubenin çevresindeki "
            "nüfus eklenince alan katsayısı 3,2'den 2,5'e düşüyor. Katsayı küçüldüğüne göre çevredeki nüfus modelden "
            "çıkarılmalıdır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "İki katsayı farklı karşılaştırmalar yapar: basit model alanı farklı bütün şubeleri, çoklu model çevre "
            "nüfusu aynı şubeleri karşılaştırır. Büyük mağazalar kalabalık bölgelerde açılıyorsa basit katsayı "
            "nüfusla ortak hareketi de içerir. Katsayının artması ya da azalması tek başına hangi modelin doğru "
            "olduğunu söylemez: eklenen değişken önemli bir faktör olabilir, araştırma sorusu için gereksiz olabilir "
            "ya da temel değişkenin bir sonucu olabilir. Kontrol seçimi katsayının büyüklüğüne, R²'ye ya da "
            "yazılımın seçeneklerine göre değil; ekonomik teori, değişkenlerin zaman sırası ve araştırma sorusuyla "
            "birlikte yapılır (§5.7)."
        ),
    ),
    Question(
        key="d06", concept="kirk-bes-derece-cizgisi-ve-artik-isareti", note=_note("5.8", "Şekil 5.4"),
        prompt=(
            "Şekil 5.4'te (yatay eksen gözlenen, dikey eksen tahmin edilen fiyat) 45 derece çizgisinin altında kalan "
            "bir konutun gözlenen fiyatı tahmin edilen fiyatından yüksektir; yani bu konutun artığı pozitiftir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Çizginin altında dikey eksendeki değer yatay eksendekinden küçüktür: tahmin edilen fiyat gözlenen "
            "fiyattan düşüktür. Artık gözlenen eksi tahmin edilen fiyat olduğu için pozitiftir; model bu konutu "
            "olduğundan ucuz tahmin etmiştir. Konum veya yapı kalitesi gibi modelde bulunmayan özellikler böyle "
            "artıklara yol açabilir (§5.8, Şekil 5.4)."
        ),
    ),
    Question(
        key="d07", concept="duzeltilmis-r2-negatif-olabilir", note=_note("5.9", "(5.8)"),
        prompt="Düzeltilmiş R² her zaman 0 ile 1 arasında bir değer alır.",
        answer=TrueFalse(False),
        explanation=(
            "(5.8)'de HKT/TKT oranı $(n-1)/(n-k-1)$ ile büyütülerek birden çıkarılır. Bu çarpan birden büyük olduğu "
            "için uyum zayıf ve açıklayıcı değişken sayısı gözlem sayısına göre büyükse değer sıfırın altına düşer: "
            "HKT/TKT = 0,98, $n = 30$, $k = 5$ için $1 - 0{,}98 \\times 29/24 \\approx -0{,}18$. R² ise sabit terimli "
            "modelde 0 ile 1 arasındadır (§5.9)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="gercek-gozlemde-tahmin-ve-artik", note=_note("5.4", "(5.5)"),
        prompt=(
            "WAGE1 veri setindeki ilk çalışanın eğitimi 11 yıl, potansiyel deneyimi 2 yıl, kıdemi 0 yıl ve saatlik "
            "ücreti 3,10 dolardır. Denklem (5.5)'e göre bu çalışanın tahmin edilen saatlik ücreti **(1)** dolar, "
            "artığı **(2)** dolardır. (Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(3.76, 0.005, "3,76"), NumberBlank(-0.66, 0.005, "−0,66"))),
        explanation=(
            "Tahmin: −2,8727 + 0,5990 × 11 + 0,0223 × 2 + 0,1693 × 0 ≈ 3,76 dolar. Artık: 3,10 − 3,76 = −0,66 "
            "dolar. Negatif artık, gözlenen ücretin aynı eğitim, deneyim ve kıdeme sahip çalışanlar için modelin "
            "tahmininden düşük olduğunu gösterir; tek başına modelin yanlış olduğunu göstermez (§5.4)."
        ),
    ),
    Question(
        key="b02", concept="artiklar-regresyonunu-elle-hesaplama", note=_note("5.5", "Şekil 5.2"),
        prompt=(
            "Dört gözlemlik küçük bir örnekte ücretin ve eğitimin deneyimle açıklanamayan kısımları (sabit terimli iki "
            "yardımcı regresyonun artıkları) şöyledir: ücret artıkları 0,9; 0,3; −0,5; −0,7; eğitim artıkları 1; 1; "
            "−1; −1. Ücret artıklarının eğitim artıklarına göre regresyonunun eğimi **(1)** olur. Birinci gözlemin bu "
            "doğruya göre artığı **(2)** olur. (Virgülden sonra bir basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.6, 0.005, "0,6"), NumberBlank(0.3, 0.005, "0,3"))),
        explanation=(
            "İki artık serisinin de ortalaması sıfırdır; eğim Bölüm 3'teki formülle Σ(eğitim artığı × ücret artığı) / "
            "Σ(eğitim artığı)² = (0,9 + 0,3 + 0,5 + 0,7)/4 = 0,6 olur. §5.5'e göre bu sayı, ücretin eğitim ve deneyime "
            "göre çoklu regresyonundaki eğitim katsayısıdır. Doğru (0, 0)'dan geçer; birinci gözlemin artığı 0,9 − "
            "0,6 × 1 = 0,3'tür (§5.5)."
        ),
    ),
    Question(
        key="b03", concept="r2-den-hkt", note=_note("5.9", "Tablo 5.1"),
        prompt=(
            "Tablo 5.1'in (2) sütununda R² = 0,3064'tür. Bölüm 4'te WAGE1 için toplam kareler toplamı TKT = 7160,4 ve "
            "basit modelin artık kareleri toplamı HKT = 5980,7 bulunmuştu. Buna göre çoklu modelin HKT'si yaklaşık "
            "**(1)** olur; deneyim ve kıdemin eklenmesi HKT'yi yaklaşık **(2)** azaltmıştır. (Tam sayıya "
            "yuvarlayın.)"
        ),
        answer=FillBlanks((NumberBlank(4966, 1.0, "4966"), NumberBlank(1014, 1.0, "1014"))),
        explanation=(
            "$R^2 = 1 - \\text{HKT}/\\text{TKT}$ olduğundan HKT = 7160,4 × (1 − 0,3064) ≈ 4966. Bağımlı değişken ve "
            "örneklem aynı kaldığı için TKT değişmez; artıklarda kalan değişkenlik 5980,7'den 4966'ya, yani yaklaşık "
            "1014 azalmıştır. R²'nin 0,1648'den 0,3064'e yükselmesi bu azalmanın oranla ifadesidir (§5.6, §5.9)."
        ),
    ),
    Question(
        key="b04", concept="gozlem-sayisi-ve-duzeltme-payi", note=_note("5.9", "(5.8)"),
        prompt=(
            "Beş açıklayıcı değişkenli bir modelde TKT = 1200 ve HKT = 840'tır. Model 40 gözlemle tahmin edildiyse "
            "düzeltilmiş R² **(1)**, aynı kareler toplamları 400 gözlemle elde edildiyse düzeltilmiş R² **(2)** olur. "
            "(Virgülden sonra üç basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.197, 0.0005, "0,197"), NumberBlank(0.291, 0.0005, "0,291"))),
        explanation=(
            "(5.8): n = 40 için 1 − (840/34)/(1200/39) ≈ 0,197; n = 400 için 1 − (840/394)/(1200/399) ≈ 0,291. İki "
            "durumda da R² = 1 − 840/1200 = 0,30'dur. Aynı beş katsayının serbestlik derecesi maliyeti küçük "
            "örneklemde büyüktür; gözlem sayısı arttıkça düzeltilmiş R², R²'ye yaklaşır (§5.9)."
        ),
    ),
    Question(
        key="b05", concept="esit-tahmin-icin-telafi-eden-fark", note=_note("5.12", "(5.9)"),
        prompt=(
            "Denklem (5.9)'a göre A firmasının reklamı B'ninkinden 4,5 milyon TL fazla, çalışan sayıları aynıdır. "
            "İki firmanın tahmin edilen satışı aynıysa A'nın fiyatı B'ninkinden **(1)** TL yüksektir. C firmasının "
            "fiyatı D'ninkinden 2 TL yüksek, reklamları aynıdır; tahmin edilen satışları aynıysa C'nin çalışan "
            "sayısı D'ninkinden **(2)** kişi fazladır. (Tam sayı yazın.)"
        ),
        answer=FillBlanks((NumberBlank(8, 0.005, "8"), NumberBlank(45, 0.05, "45"))),
        explanation=(
            "Tahmin farkı katsayı katkılarının toplamıdır (§5.3). A ile B: 1,6 × 4,5 − 0,9 × Δfiyat = 0, yani Δfiyat "
            "= 7,2/0,9 = 8 TL; 1 milyon TL ek reklamın tahmin edilen satış katkısını yaklaşık 1,6/0,9 ≈ 1,78 TL "
            "daha yüksek fiyat karşılar. C ile D: −0,9 × 2 + 0,04 × Δçalışan = 0, yani Δçalışan = 1,8/0,04 = 45 kişi. "
            "Bunlar modelin tahminleri arasındaki karşılaştırmalardır; reklamın ya da fiyatın nedensel etkisi "
            "olarak okunmaz (§5.12)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="kosullu-ortalama-farki-katkilarin-toplami", note=_note("5.2", "(5.3)"),
        prompt=(
            "(5.3)'teki çoklu modelde iki birim grubu karşılaştırılıyor: birinci grubun $X_1$ değeri $d_1$ birim, "
            "$X_2$ değeri $d_2$ birim daha yüksek; $X_3, \\ldots, X_k$ değerleri aynı. Birinci grubun $Y$ koşullu "
            "ortalamasından ikincininkini çıkarın. Gerekmeyen sembolleri kullanmayın."
        ),
        answer=Equation(
            lhs="\\Delta\\,\\mathbb{E}(Y \\mid X_1, \\ldots, X_k)",
            symbols=(
                Symbol("b0", "\\beta_0", "sabit parametre", -5, 5, aliases=beta_aliases(0) + _sub("b", 0)),
                Symbol("b1", "\\beta_1", "X₁'in kısmi eğimi", 0.1, 2, aliases=beta_aliases(1) + _sub("b", 1)),
                Symbol("b2", "\\beta_2", "X₂'nin kısmi eğimi", -2, -0.1, aliases=beta_aliases(2) + _sub("b", 2)),
                Symbol("b3", "\\beta_3", "X₃'ün kısmi eğimi", 0.1, 2, aliases=beta_aliases(3) + _sub("b", 3)),
                Symbol("d1", "d_1", "X₁'deki fark", 1, 5, aliases=_sub("d", 1)),
                Symbol("d2", "d_2", "X₂'deki fark", 1, 5, aliases=_sub("d", 2)),
            ),
            answer="b1*d1 + b2*d2",
            shown="\\beta_1 d_1 + \\beta_2 d_2",
        ),
        explanation=(
            "(5.3)'te koşullu ortalama parametrelerde doğrusaldır: iki grubun farkında sabit parametre ve değeri aynı "
            "olan değişkenlerin terimleri birbirini götürür, geriye $\\beta_1 d_1 + \\beta_2 d_2$ kalır. Her kısmi "
            "eğim, diğer açıklayıcı değişkenler sabitken kendi değişkeninin katkısını verir; birden fazla değişken "
            "farklıysa katkılar toplanır. Örneklemdeki karşılığı katsayı tahminleriyle yazılır (§5.2, §5.3)."
        ),
    ),
    Question(
        key="e02", concept="sabit-tahmini-ortalamalardan", note=_note("5.4"),
        prompt=(
            "Sabit terimli EKK ile $\\widehat{Y} = b_0 + b_1 X_1 + b_2 X_2$ tahmin edilmiştir. Sabit tahmini $b_0$'ı "
            "$Y$'nin ortalaması, açıklayıcı değişkenlerin ortalamaları ve eğim tahminleri cinsinden yazın."
        ),
        answer=Equation(
            lhs="b_0",
            symbols=(
                Symbol("ybar", "\\bar{Y}", "Y'nin örneklem ortalaması", 5, 20, aliases=bar_aliases("Y")),
                Symbol("x1bar", "\\bar{X}_1", "X₁'in örneklem ortalaması", 1, 20, aliases=_bar_sub("X", 1)),
                Symbol("x2bar", "\\bar{X}_2", "X₂'nin örneklem ortalaması", 1, 20, aliases=_bar_sub("X", 2)),
                Symbol("b1", "b_1", "X₁'in eğim tahmini", 0.1, 2, aliases=_sub("b", 1)),
                Symbol("b2", "b_2", "X₂'nin eğim tahmini", -2, -0.1, aliases=_sub("b", 2)),
            ),
            answer="ybar - b1*x1bar - b2*x2bar",
            shown="\\bar{Y} - b_1 \\bar{X}_1 - b_2 \\bar{X}_2",
        ),
        explanation=(
            "Sabit terimli çoklu EKK'de artıkların toplamı sıfırdır; tahmin edilen değerlerin ortalaması "
            "$\\bar{Y}$'dir. Tahmin edilen değerlerin ortalaması $b_0 + b_1\\bar{X}_1 + b_2\\bar{X}_2$ olduğundan "
            "$b_0 = \\bar{Y} - b_1\\bar{X}_1 - b_2\\bar{X}_2$: tahmin denklemi ortalamalar noktasından geçer. WAGE1'de "
            "5,896 − 0,5990 × 12,563 − 0,0223 × 17,017 − 0,1693 × 5,105 ≈ −2,873 (§5.4)."
        ),
    ),
    Question(
        key="e03", concept="arindirilmis-degisken-yardimci-artik", note=_note("5.5", "Şekil 5.2"),
        prompt=(
            "Eğitimin ($X_1$) deneyim ($X_2$) ve kıdeme ($X_3$) göre yardımcı regresyonu şöyle tahmin edilmiştir: "
            "$\\widehat{X}_1 = g_0 + g_2 X_2 + g_3 X_3$. Eğitimi $X_1$, deneyimi $X_2$ ve kıdemi $X_3$ olan bir "
            "çalışanın “arındırılmış eğitim” değerini (Şekil 5.2'nin yatay eksenindeki değer) yazın."
        ),
        answer=Equation(
            lhs="\\text{arındırılmış eğitim}",
            symbols=(
                Symbol("x1", "X_1", "eğitim (yıl)", 6, 18, aliases=("X_1", "X₁", "X1", "x_1", "x₁")),
                Symbol("x2", "X_2", "deneyim (yıl)", 1, 40, aliases=("X_2", "X₂", "X2", "x_2", "x₂")),
                Symbol("x3", "X_3", "kıdem (yıl)", 0, 20, aliases=("X_3", "X₃", "X3", "x_3", "x₃")),
                Symbol("g0", "g_0", "yardımcı regresyonun sabiti", 10, 15, aliases=_sub("g", 0)),
                Symbol("g2", "g_2", "deneyimin yardımcı katsayısı", -0.2, -0.01, aliases=_sub("g", 2)),
                Symbol("g3", "g_3", "kıdemin yardımcı katsayısı", 0.01, 0.2, aliases=_sub("g", 3)),
            ),
            answer="x1 - g0 - g2*x2 - g3*x3",
            shown="X_1 - (g_0 + g_2 X_2 + g_3 X_3)",
        ),
        explanation=(
            "Arındırılmış eğitim, yardımcı regresyonun artığıdır: gözlenen eğitimden, deneyim ve kıdemle doğrusal "
            "olarak tahmin edilen eğitim çıkarılır. Pozitif değer, çalışanın eğitiminin deneyim ve kıdemine göre "
            "beklenen düzeyden yüksek olduğunu gösterir. Gerçek bir kişinin yeni eğitim değişkeni değil, örneklem "
            "sapmasıdır; gözlenmeyen faktörler bu işlemle ayrılmaz (§5.5)."
        ),
    ),
    Question(
        key="e04", concept="log-bagimli-coklu-modelde-yuzde-fark", note=_note("5.6", "Tablo 5.1"),
        prompt=(
            "İkinci el otomobil ilanlarında $\\widehat{\\ln(\\text{fiyat})} = c_0 + c_1\\,\\text{motor hacmi} + "
            "c_2\\,\\text{yaş}$ tahmin edilmiştir (motor hacmi litre, yaş yıl). Motor hacmi aynı, yaşı $a$ yıl fazla "
            "olan otomobilin tahmin edilen fiyatındaki yaklaşık yüzde farkı yazın (küçük değişim yaklaşımı). Gerekmeyen "
            "sembolleri kullanmayın."
        ),
        answer=Equation(
            lhs="\\%\\Delta\\,\\widehat{\\text{fiyat}} \\approx",
            symbols=(
                Symbol("c0", "c_0", "sabit", 5, 8, aliases=_sub("c", 0)),
                Symbol("c1", "c_1", "motor hacmi katsayısı", 0.05, 0.3, aliases=_sub("c", 1)),
                Symbol("c2", "c_2", "yaş katsayısı", -0.15, -0.02, aliases=_sub("c", 2)),
                Symbol("a", "a", "yaş farkı (yıl)", 1, 3),
            ),
            answer="100*c2*a",
            shown="100\\,c_2\\,a",
        ),
        explanation=(
            "Bağımlı değişken logaritmik olduğunda motor hacmi aynıyken log fiyattaki tahmin edilen fark $c_2 a$'dır; "
            "100 ile çarpımı küçük değişimler için yaklaşık yüzde farktır. Motor hacmi aynı tutulduğu için $c_1$ "
            "girmez. Tablo 5.1'in "
            "(3) sütununda eğitim katsayısı 0,0920: deneyim ve kıdem aynıyken bir ek eğitim yılı yaklaşık %9,2 ücret "
            "farkıyla ilişkilidir. Tam yüzde değişim Konu 9'dadır (§5.6)."
        ),
    ),
    Question(
        key="e05", concept="duzeltilmis-r2-r2-cinsinden", note=_note("5.9", "(5.8)"),
        prompt=(
            "Sabit terimli, $k$ açıklayıcı değişkenli ve $n$ gözlemli bir modelin düzeltilmiş $R^2$'sini (5.8) "
            "yardımıyla $R^2$, $n$ ve $k$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\bar{R}^2",
            symbols=(
                Symbol("r2", "R^2", "belirleme katsayısı", 0.05, 0.95, aliases=_R2),
                Symbol("n", "n", "gözlem sayısı", 30, 1000),
                Symbol("k", "k", "sabit dışındaki açıklayıcı değişken sayısı", 1, 8),
            ),
            answer="1 - (1 - r2)*(n - 1)/(n - k - 1)",
            shown="1 - (1 - R^2)\\,\\frac{n-1}{n-k-1}",
        ),
        explanation=(
            "(5.8)'de pay ve paydayı düzenleyelim: $\\bar{R}^2 = 1 - (\\text{HKT}/\\text{TKT})\\,(n-1)/(n-k-1)$ ve "
            "$\\text{HKT}/\\text{TKT} = 1 - R^2$. Yeni değişken R²'yi azaltmasa da $(n-1)/(n-k-1)$ çarpanını büyütür; "
            "uyum artışı bu maliyeti karşılamazsa düzeltilmiş R² düşer. WAGE1'de 1 − 0,6936 × 525/522 ≈ 0,3024, "
            "Tablo 5.1'deki değerdir (§5.9)."
        ),
    ),
)


KONU05_QUIZ = QuestionSet(
    topic_key="konu05",
    title="Konu 5: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"5.{number}" for number in range(1, 13)),  # §5.13 bölüm özetidir
)
