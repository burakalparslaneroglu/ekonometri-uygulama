"""Konu 6 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz 6.1–6.13 ve Mini Quiz
6.1–6.13 maddelerini tekrar etmez: 0,42'lik katsayıya ilişkin dört ifadenin sınıflandırılması; TL–kuruş, yetenek,
gönüllü anket, X ile X² ve hata değişkenliği durumları; 3,5; 4,4; 3,8; 4,1; 4,2 tahminleri; Tablo 6.3'ün yorum
soruları ve 30.000 tekrar; dört araştırmada hata teriminde kalan faktör; yetenek, deneyim, ürün kalitesi, okul
kaynakları ve semt geliriyle yanlılığın yönü; Y = 5 + 1,2X − 2Z + u, Z = 3 + 0,4X + r hesapları; Tablo 6.6 ve
0,0701 × (−1,4682) çarpımı; tam bağlantı için beş model tasarımı; Tablo 6.7 ve Tablo 6.8'in yorum soruları;
korelasyon 0,85 ve VIF 6 olan konut modeli; iki reklam modelli bütünleşik sınav hazırlığı ve mini quizlerdeki sorular
(ör. aylık ve yıllık gelir, X ile X², tek eksik değişkende yanlılığın yönü).
Aynı becerileri yeni bağlamlarla ve yeni sayılarla sınar. Standart hata, t, p-değeri ve güven aralığı Konu 7'nin;
heteroskedastisiteye dayanıklı çıkarım Konu 12'nin konusudur; sorular bunları kullanmaz.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol, beta_aliases
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


def _sum_of_squares(name: str) -> tuple[str, ...]:
    """Kareler toplamı yazımları: \\text{TKT}_j, \\mathrm{TKT}_j, \\operatorname{TKT}_j, TKT_j, TKTj (büyük ve küçük
    harf); süslü parantezler okunmadan önce silinir."""

    forms: list[str] = []
    for item in (name, name.lower()):
        for command in ("\\text", "\\mathrm", "\\operatorname", ""):
            forms += [f"{command}{item}_j", f"{command}{item}"]
        forms.append(f"{item}j")
    return tuple(forms)


def _greek(latex_name: str, letter: str, word: str, index: int | None = None) -> tuple[str, ...]:
    """Yunan harfli sembolün yazımları: \\delta_0, δ_0, δ₀, delta_0, delta0 (alt indis yoksa \\rho, ρ)."""

    if index is None:
        return (f"\\{latex_name}", letter)
    small = "₀₁₂₃"[index]
    return (f"\\{latex_name}_{index}", f"\\{latex_name}{index}", f"{letter}_{index}", f"{letter}{index}",
            f"{letter}{small}", f"{word}_{index}", f"{word}{index}")


_SIGMA_Z = ("\\sigma_Z", "σ_Z", "σZ", "sigma_Z", "sigmaZ", "s_Z", "S_Z", "\\sigma_z", "σ_z", "σz", "sigma_z", "s_z")
_RHO_XZ = tuple(f"{rho}_{pair}" for rho in ("\\rho", "ρ", "rho") for pair in ("XZ", "ZX", "xz", "zx")) + tuple(
    f"r_{pair}" for pair in ("XZ", "ZX", "xz", "zx")) + ("\\rho", "ρ")
"""Korelasyon yazımları: ρ_XZ, \\rho_{zx}, notlardaki r_{XZ}; alt indissiz r, (6.7)'de hata terimi olduğu için yoktur."""
_SIGMA_X = ("\\sigma_X", "σ_X", "σX", "sigma_X", "sigmaX", "s_X", "S_X", "\\sigma_x", "σ_x", "σx", "sigma_x", "s_x")


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="cebirsel-ozellik-varsayim-gerektirmez", note=_note("6.1"),
        prompt=(
            "Eğitim ve deneyimle kurulmuş sabit terimli bir EKK ücret modeli için aşağıdaki ifadelerden hangisi hiçbir "
            "model varsayımı gerektirmez, yani EKK'nin hesaplanabildiği her örneklemde doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Hata terimlerinin ($u_i$) örneklemdeki toplamı sıfırdır",
                "Modelde bulunmayan yetenek, eğitim katsayısını yukarı yönlü etkilemez",
                "Artıkların toplamı sıfırdır; tahmin edilen değerlerin ortalaması Ȳ'dir",
                "Deneyim modelden çıkarılsa da eğitim katsayısı aynı kalır",
            ),
            correct=2,
        ),
        explanation=(
            "Artıkların toplamının sıfır olması ve tahmin edilen değerlerin ortalamasının Ȳ'ye eşit olması EKK'nin "
            "cebirsel sonuçlarıdır (§4.1, §5.4); her veri tablosunda geçerlidir. Hata terimi gözlenmez; örneklemdeki "
            "toplamının sıfır olması garanti değildir (artık ile hata aynı şey değildir, §6.5). Dışlanan yeteneğin "
            "katsayıyı etkilememesi sıfır koşullu ortalamaya dayanır; deneyim çıkarılınca eğitim katsayısı "
            "değişebilir (WAGE1'de 0,6443'ten 0,5414'e, §6.8). Hesaplama ile güvenilir yorum farklı aşamalardır "
            "(§6.1)."
        ),
    ),
    Question(
        key="k02", concept="a4-ihlalini-ayirt-etme", note=_note("6.2", "Tablo 6.1"),
        prompt=(
            "Kira, dairenin alanı ve oda sayısıyla açıklanıyor. Aşağıdaki durumlardan hangisi öncelikle sıfır "
            "koşullu ortalama varsayımını (A4) tehdit eder?"
        ),
        answer=MultipleChoice(
            (
                "Kira, alan ve oda sayısı her dairenin ilanından hatasız kaydedilmiştir",
                "Kira TL yerine bin TL cinsinden kaydedilmiştir",
                "Veri, şehirdeki kiralık dairelerden rastgele seçilmiş 400 daireden oluşur",
                "Manzara gözlenmiyor; manzaralı daireler hem daha pahalı hem daha büyük",
            ),
            correct=3,
        ),
        explanation=(
            "Manzara hata teriminde kalır ve modeldeki alanla birlikte hareket eder: büyük dairelerde hata teriminin "
            "ortalaması pozitif olur, $\\mathbb{E}(u \\mid \\text{alan}, \\text{oda}) = 0$ kuşkulu hâle gelir (A4). "
            "Diğer üç durum hiçbir varsayımı bozmaz: verinin hatasız kaydedilmesi bir sorun değildir, birim dönüşümü "
            "yalnız katsayıların ölçeğini değiştirir, rastgele seçim A2'nin istediğidir (§6.2, Tablo 6.1)."
        ),
    ),
    Question(
        key="k03", concept="kura-kosullu-ortalamayi-savunur", note=_note("6.5", "(6.2)"),
        prompt=(
            "Bir belediye meslek kursu bursunu başvuranlara dağıtıyor; araştırmacı bursu alıp almamanın ($X$) "
            "sonraki yılki gelirle ($Y$) ilişkisini inceliyor. Hangi dağıtım biçiminde $\\mathbb{E}(u \\mid X) = 0$ "
            "varsayımı en kolay savunulur?"
        ),
        answer=MultipleChoice(
            (
                "Burs, başvuranlar arasında kurayla dağıtılmıştır",
                "Burs, mülakatta en motive görünen adaylara verilmiştir",
                "Burs, geçmiş yıl geliri en yüksek olan adaylara verilmiştir",
                "Burs, kursa kendi isteğiyle devam etmeyi seçenlere verilmiştir",
            ),
            correct=0,
        ),
        explanation=(
            "Kura, bursu hata terimindeki motivasyon, yetenek ve geçmiş gelir gibi faktörlerden bağımsız kılar; bu "
            "faktörlerin ortalaması burs alanlarla almayanlarda sistematik olarak farklılaşmaz. Diğer üç kuralda burs, "
            "geliri de etkileyen özelliklere göre verilir ve hata teriminin ortalaması iki grupta farklı olur. "
            "Gözlemsel veride varsayım ekonomik teori, veri üretim süreci ve araştırma tasarımıyla savunulur (§6.5; "
            "rastgele atama için Bölüm 2)."
        ),
    ),
    Question(
        key="k04", concept="isaret-tablosunu-yeni-baglamda-uygulama", note=_note("6.6", "Tablo 6.4"),
        prompt=(
            "Kira ($Y$) bina yaşıyla ($X$) açıklanıyor; merkeze uzaklık ($Z$) modelde yok. Diğer unsurlar sabitken "
            "merkezden uzak dairelerin kirası daha düşüktür ($\\beta_2 < 0$) ve eski binalar merkeze daha yakındır "
            "(bina yaşı ile uzaklık negatif ilişkili, $\\delta_1 < 0$). Kısa modeldeki bina yaşı katsayısı için "
            "hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Yanlılık negatiftir: kısa model katsayısı aşağı yönlü etkilenir",
                "Yanlılık pozitiftir: kısa model katsayısı yukarı yönlü etkilenir",
                "İki ilişki de negatif olduğu için etkiler birbirini götürür; yanlılık sıfırdır",
                "Z gözlenmediği için yanlılığın yönü hakkında hiçbir şey söylenemez",
            ),
            correct=1,
        ),
        explanation=(
            "(6.9)'a göre yanlılık $\\beta_2\\delta_1$'dir; iki negatif sayının çarpımı pozitiftir (Tablo 6.4, son "
            "satır). Eski binalar merkeze yakın olduğu için kısa modelde bina yaşı, merkeze yakınlığın kiradaki "
            "avantajını da taşır; eski binaların kira farkı olduğundan daha az olumsuz görünebilir. Z gözlenmese de "
            "iki ilişkinin işaretleri ekonomik bilgiyle değerlendirilebilir (§6.6)."
        ),
    ),
    Question(
        key="k05", concept="yas-ve-dogum-yili-tam-baglanti", note=_note("6.9", "(6.13)"),
        prompt=(
            "2024'te toplanan bir anket verisinde bütün gözlemlerde yaş = 2024 − doğum yılıdır. Ücret, yaş ve doğum "
            "yılıyla birlikte sabit terimli bir modelle açıklanmak isteniyor. Hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "İki değişken yalnız yüksek korelasyonludur; katsayılar hesaplanır ama örneklemden örnekleme çok "
                "değişir",
                "Doğum yılı 1900'den itibaren ölçülürse (1985 yerine 85) iki katsayı ayrı ayrı tahmin edilebilir",
                "Gözlem sayısı artırılırsa iki değişkenin katsayıları ayrı ayrı tahmin edilebilir hâle gelir",
                "Yaş, sabit terim ile doğum yılının tam doğrusal birleşimidir; ayrı katsayılar benzersiz tahmin "
                "edilemez",
            ),
            correct=3,
        ),
        explanation=(
            "Yaş = 2024 × 1 − 1 × doğum yılı kimliği bütün gözlemlerde geçerlidir: yaş, sabit terim sütunu ile doğum "
            "yılının tam doğrusal birleşimidir ((6.13)'teki biçim, a = 2024 ve b = −1). Kimlik her yeni gözlemde de "
            "geçerli olacağı için gözlem sayısını artırmak sorunu çözmez; doğum yılını 1900'den ölçmek de yalnız "
            "kimliği yaş = 124 − yeni değişken biçimine getirir. Veri yalnız birleşik etkiyi belirler; değişkenlerden "
            "biri modelden çıkarılır (§6.9)."
        ),
    ),
    Question(
        key="k06", concept="uygunsuz-mekanik-cozum", note=_note("6.12"),
        prompt=(
            "Buğday verimi modelinde gübre ve sulama miktarı çok yakın hareket ediyor (çiftçiler ikisini birlikte "
            "artırıyor) ve VIF değerleri yüksek. Aşağıdaki tepkilerden hangisi notlardaki uygun tepkilerden biri "
            "**değildir**?"
        ),
        answer=MultipleChoice(
            (
                "Gübre ile sulamanın ayrı ayrı değiştiği ek gözlemler (ör. deneme parselleri) toplamak",
                "Araştırma sorusu ikisinin ortak ilişkisiyle ilgiliyse bu amacı açıkça belirtmek",
                "Sulamanın katsayısı beklenen işarette çıkmadığı için sulamayı modelden silmek",
                "Sonuçların model seçimine duyarlılığını şeffaf biçimde raporlamak",
            ),
            correct=2,
        ),
        explanation=(
            "Yalnız katsayı işareti hoş görünmediği için değişken silmek notlarda uygun olmayan mekanik çözümler "
            "arasındadır; sulama verimle ilişkili önemli bir faktörse onu silmek gübre katsayısında eksik değişken "
            "yanlılığı da doğurabilir. Ayrı değişkenlik sağlayan veri toplamak, araştırma sorusu ortak ilişkiyle "
            "ilgiliyse bunu belirtmek ve duyarlılığı raporlamak uygun tepkilerdir (§6.12)."
        ),
    ),
    Question(
        key="k07", concept="yanlilik-ve-baglanti-belirtileri", note=_note("6.13", "Tablo 6.10"),
        prompt=(
            "Bir araştırmacı iki durum raporluyor. (i) İki açıklayıcı değişkenin VIF değeri 150'dir; küçük bir veri "
            "düzeltmesinde iki katsayı ters yönlerde büyük değişiyor. (ii) Modelde "
            "olmayan bir faktör hem sonuçla hem temel açıklayıcı değişkenle ilişkili görünüyor. Hangi eşleştirme "
            "doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "(i) eksik değişken yanlılığının, (ii) yüksek çoklu doğrusal bağlantının belirtisidir",
                "(i) yüksek çoklu doğrusal bağlantının, (ii) eksik değişken yanlılığının belirtisidir",
                "(i) ve (ii) birlikte yalnız yüksek çoklu doğrusal bağlantının belirtisidir",
                "(i) ve (ii) birlikte yalnız eksik değişken yanlılığının belirtisidir",
            ),
            correct=1,
        ),
        explanation=(
            "(i) Tablo 6.8'deki örüntüdür: yüksek bağlantı sıfır koşullu ortalama altında katsayıları yanlı yapmaz, "
            "ayrı katsayıları küçük veri değişikliklerine karşı hassaslaştırır. (ii) "
            "Eksik değişken yanlılığının temel kaynağıdır: dışlanan faktör hem sonuçla hem dahil edilen değişkenle "
            "ilişkilidir ve katsayı yanlış merkez çevresinde toplanabilir (§6.13, Tablo 6.10)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="orneklem-degiskenligi-gerekli", note=_note("6.2", "Tablo 6.1"),
        prompt=(
            "Örneklemdeki bütün çalışanların eğitimi 12 yılsa, sabit terimli bir ücret modelinde eğitim katsayısı "
            "benzersiz biçimde tahmin edilemez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Eğitim sütunu bütün gözlemlerde 12'dir, yani sabit terim sütununun 12 katıdır. A3 hem tam doğrusal "
            "birleşimi dışlar hem de ilgili örneklem değişkenliğini ister (Tablo 6.1): eğitimde değişkenlik yoksa "
            "eğitim farkına karşılık gelen ücret farkı veriden öğrenilemez. Basit regresyonda da eğim formülünün "
            "paydası Σ(Xᵢ − X̄)² sıfır olurdu (§6.2)."
        ),
    ),
    Question(
        key="d02", concept="yansizlik-ortalama-medyan-degil", note=_note("6.3", "(6.3)"),
        prompt=(
            "Yansız bir tahmin edicinin tekrarlı örneklemlerdeki tahminlerinin yarısı gerçek parametrenin üstünde, "
            "yarısı altında olmak zorundadır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "(6.3)'e göre yansızlık, tahminlerin beklenen değerinin (uzun dönem ortalamasının) gerçek parametreye "
            "eşit olmasıdır. Ortalama ile medyan farklı olabilir: tahminlerin dağılımı çarpıksa çoğunluk bir tarafta, "
            "az sayıda uzak tahmin diğer tarafta bulunabilir ve ortalama yine gerçek parametreye eşittir. Yansızlık "
            "dağılımın ortalamasına ilişkindir; simetri gerektirmez (§6.3)."
        ),
    ),
    Question(
        key="d03", concept="kosulsuz-ortalamanin-sifir-olmasi-yetmez", note=_note("6.4", "Tablo 6.3"),
        prompt=(
            "Tablo 6.3'ün ikinci senaryosunda ($u_i = 0{,}8X_i + \\varepsilon_i$; $X_i$ ve $\\varepsilon_i$ standart "
            "normal) hata teriminin koşulsuz ortalaması sıfırdır; buna rağmen sıfır koşullu ortalama varsayımı "
            "bozulmuştur."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$\\mathbb{E}(u_i) = 0{,}8\\,\\mathbb{E}(X_i) + \\mathbb{E}(\\varepsilon_i) = 0$; fakat "
            "$\\mathbb{E}(u_i \\mid X_i) = 0{,}8X_i$ ve X büyüdükçe artar. A4, hata teriminin her X değerinde "
            "ortalamasının sıfır olmasını ister; genel ortalamanın sıfır olması yetmez. Bu yüzden tahminler 1,5 "
            "yerine 2,3 çevresinde toplanır (§6.4, Tablo 6.3)."
        ),
    ),
    Question(
        key="d04", concept="ayristirma-orneklemde-tam", note=_note("6.8", "(6.10)", "(6.11)", "(6.12)"),
        prompt=(
            "WAGE1'deki $0{,}5414 = 0{,}6443 + (0{,}0701)(-1{,}4682)$ ayrıştırması yalnız yaklaşık olarak sağlanır; "
            "aradaki küçük fark örnekleme hatasından kaynaklanır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Kısa model katsayısı, uzun model katsayısı ile dışarıda kalan değişkenin katsayısı ve yardımcı eğimin "
            "çarpımının toplamına örneklemde tam olarak eşittir; bu, EKK'nin cebirsel bir özelliğidir. Yazılı "
            "sayılardaki fark yalnız yuvarlamadandır: 0,6443 − 0,1029 = 0,5414. Örnekleme dalgalanması, beklenen "
            "değer formülü (6.8) ile tek örneklemdeki tahmin arasındaki farkta ortaya çıkar (Tablo 6.5'te 4,067 ve "
            "4,072) (§6.8)."
        ),
    ),
    Question(
        key="d05", concept="ortak-iliski-veriden-belirlenir", note=_note("6.10"),
        prompt=(
            "Konut fiyatı modelinde brüt alan ile net alan çok yakın hareket ediyorsa (aralarında tam doğrusal kimlik "
            "yok), iki alan "
            "ölçüsünün fiyatla ortak ilişkisi de veriden belirlenemez."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Yüksek fakat tam olmayan bağlantıda veri ortak hareketi iyi belirler; zorlaşan, bu ortak ilişkinin iki "
            "değişken arasında nasıl paylaştırılacağıdır. Tablo 6.7'de bağlantı yükseldikçe $\\widehat\\beta_1$ "
            "tahminlerinin standart sapması 0,129'dan 3,069'a çıkarken katsayı toplamınınki bütün düzeylerde 0,090'da "
            "kalır (§6.10, Tablo 6.7)."
        ),
    ),
    Question(
        key="d06", concept="veri-duyarliligi-baglantiyla-artar", note=_note("6.11", "Tablo 6.8"),
        prompt=(
            "Konu 6 Deney 3'te gürültü σ = 1 seçilirse ($X_1$–$X_2$ korelasyonu yaklaşık 0,7), ilk gözlemin $X_2$ "
            "değerindeki 0,08'lik artış ayrı katsayıları Tablo 6.8'deki gibi yaklaşık 0,22 değiştirir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Tablo 6.8'deki büyük değişim, korelasyonun yaklaşık 0,999 olmasından kaynaklanır: $X_1$ ile $X_2$'nin "
            "birbirinden bağımsız hareketi çok az olduğu için tek bir gözlem bu az bilgiyi belirgin biçimde "
            "değiştirir. Korelasyon 0,7 civarındayken ayrı katsayılar için bol bağımsız hareket vardır: Deney 3'te "
            "σ = 1 iken aynı değişiklik $\\widehat\\beta_1$'i yaklaşık 0,002, $\\widehat\\beta_2$'yi yaklaşık 0,0002 "
            "değiştirir (§6.11, Tablo 6.8)."
        ),
    ),
    Question(
        key="d07", concept="iki-sorun-birlikte-olabilir", note=_note("6.13", "Tablo 6.10"),
        prompt=(
            "Aynı ücret modelinde yetenek dışarıda kalmış olabilir (eksik değişken yanlılığı riski) ve modeldeki iki "
            "deneyim ölçüsü neredeyse aynı bilgiyi taşıyabilir (yüksek bağlantı); bu iki sorun birbirini dışlamaz."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Eksik değişken yanlılığı katsayının hangi merkez çevresinde toplandığıyla, yüksek bağlantı ayrı "
            "katsayıların ne kadar kesin ayrıştırıldığıyla ilgilidir. Kaynakları farklıdır (Tablo 6.10): biri dışlanan "
            "faktörün ilişkilerinden, diğeri dahil edilen değişkenlerin birbirine yakınlığından doğar. Bir model aynı "
            "anda hem yanlı hem yüksek bağlantılı olabilir (§6.13)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="yanlilik-ve-tek-tahminin-sapmasi", note=_note("6.3", "(6.4)"),
        prompt=(
            "Gerçek parametre $\\beta_1 = 2$'dir. Tahmin edicinin tekrarlı örneklemlerdeki beklenen değeri 2,3'tür. "
            "Bir araştırmacının örnekleminde ise $\\widehat{\\beta}_1 = 1{,}8$ bulunmuştur. Tahmin edicinin yanlılığı "
            "**(1)**, bu tek tahminin gerçek parametreden sapması ($\\widehat{\\beta}_1 - \\beta_1$) **(2)** olur. "
            "(Virgülden sonra bir basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.3, 0.005, "0,3"), NumberBlank(-0.2, 0.005, "−0,2"))),
        explanation=(
            "(6.4)'e göre yanlılık $\\mathbb{E}(\\widehat{\\beta}_1) - \\beta_1 = 2{,}3 - 2 = 0{,}3$'tür; tahmin "
            "edicinin özelliğidir. Tek tahminin sapması 1,8 − 2 = −0,2'dir ve örneklemden örnekleme değişir. Yukarı "
            "yönlü yanlı bir tahmin edici tek bir örneklemde gerçek değerin altında kalabilir; tek tahminin "
            "sapmasından yanlılık okunamaz (§6.3)."
        ),
    ),
    Question(
        key="b02", concept="ayni-cekilislerle-merkez-kaymasi", note=_note("6.4", "Tablo 6.3"),
        prompt=(
            "Tablo 6.3'teki benzetimin ikinci senaryosu aynı $X$ ve $\\varepsilon$ çekilişleriyle $u_i = -0{,}5X_i + "
            "\\varepsilon_i$ için tekrarlanıyor. 3.000 eğim tahmininin ortalaması yaklaşık **(1)**, standart sapması "
            "yaklaşık **(2)** olur. (Virgülden sonra üç basamak yazın; Konu 6 Deney 1'de γ = −0,5 seçerek kontrol "
            "edebilirsiniz.)"
        ),
        answer=FillBlanks((NumberBlank(1.001, 0.0015, "1,001"), NumberBlank(0.117, 0.0005, "0,117"))),
        explanation=(
            "Hata terimindeki −0,5X bileşeni regresyonda X'in katsayısına eklenir: her örneklemde eğim tahmini, "
            "birinci senaryodakinden tam olarak 0,5 küçüktür. Ortalama 1,501 − 0,5 = 1,001 (kuramsal merkez 1,5 − "
            "0,5 = 1,0), standart sapma ise değişmez: 0,117. Sıfır koşullu ortalamanın bozulması tahminlerin "
            "merkezini kaydırır, yayılımını değiştirmez; yanlılık bu kez aşağı yönlüdür (§6.4, Tablo 6.3)."
        ),
    ),
    Question(
        key="b03", concept="yanlilik-isareti-tersine-cevirebilir", note=_note("6.7", "Tablo 6.5", "(6.8)", "Tablo 6.4"),
        prompt=(
            "Konu 6 Deney 2'de $\\delta = -0{,}7$ seçilsin ($\\beta_X = 2$, $\\beta_Z = 3$, $n = 5.000$, tohum 305); "
            "yardımcı eğim $\\hat\\delta = -0{,}711$ bulunur. Notlardaki gibi yardımcı eğimle hesaplanan beklenen kısa "
            "model eğimi (6.8) **(1)**, bu beklenen eğimin gerçek katsayıdan farkı (yanlılık) **(2)** olur. (Virgülden "
            "sonra üç basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(-0.133, 0.0005, "−0,133"), NumberBlank(-2.133, 0.0005, "−2,133"))),
        explanation=(
            "(6.8): $\\beta_X + \\beta_Z\\hat\\delta = 2 + 3 \\times (-0{,}711) = -0{,}133$; yanlılık "
            "$\\beta_Z\\hat\\delta = -2{,}133$. Z, Y ile pozitif, X ile negatif ilişkili olduğu için yanlılık aşağı "
            "yönlüdür (Tablo 6.4) ve gerçek katsayı 2'yi aşacak kadar büyüktür: Deney 2'de tahmin edilen kısa model "
            "eğimi −0,128 çıkar, yani işaret bile tersine döner. Uzun model gerçek değere yakındır (§6.7)."
        ),
    ),
    Question(
        key="b04", concept="baska-dislanan-degiskenle-ayristirma", note=_note("6.8", "(6.10)"),
        prompt=(
            "WAGE1'de kısa modelin eğitim katsayısı 0,5414'tür (Denklem 6.10). Deneyim yerine kıdem dışarıda kalan "
            "değişken olsun: kıdemin eğitime göre yardımcı regresyonunda eğim −0,1466, uzun modelde (ücret ~ eğitim "
            "+ kıdem) kıdem katsayısı 0,1896'dır. Kıdemin dışarıda kalmasından gelen katkı **(1)**, uzun modeldeki "
            "eğitim katsayısı **(2)** olur. (Virgülden sonra dört basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(-0.0278, 0.00015, "−0,0278"), NumberBlank(0.5691, 0.00015, "0,5691"))),
        explanation=(
            "Katkı, dışarıda kalan değişkenin uzun modeldeki katsayısı ile yardımcı eğimin çarpımıdır: 0,1896 × "
            "(−0,1466) ≈ −0,0278. Kısa katsayı = uzun katsayı + katkı olduğundan uzun modelde eğitim katsayısı "
            "0,5414 + 0,0278 ≈ 0,5691'dir (yuvarlanmış sayılarla 0,5692). Kıdem ücretle pozitif, eğitimle negatif "
            "ilişkili olduğu için dışlanması eğitim katsayısını aşağı çeker; Uygulama Adım 1–2'de kıdemi seçerek "
            "aynı sonucu görebilirsiniz (§6.8)."
        ),
    ),
    Question(
        key="b05", concept="iki-degiskende-vif-korelasyondan", note=_note("6.10"),
        prompt=(
            "İki açıklayıcı değişkenli bir modelde her değişkenin yardımcı regresyonundaki $R^2$, iki değişken "
            "arasındaki korelasyonun karesidir. Korelasyon 0,95 ise VIF **(1)**, 0,99 ise VIF **(2)** olur. "
            "(Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(10.26, 0.005, "10,26"), NumberBlank(50.25, 0.005, "50,25"))),
        explanation=(
            "Yardımcı regresyon tek açıklayıcı değişkenli bir regresyondur; Bölüm 4'e göre $R^2 = r^2$. VIF = "
            "1/(1 − r²): 1/(1 − 0,9025) ≈ 10,26 ve 1/(1 − 0,9801) ≈ 50,25. Korelasyon 0,95'ten 0,99'a çıkınca VIF "
            "yaklaşık beş katına çıkar; Tablo 6.7'de de korelasyon bire yaklaştıkça VIF ve ayrı katsayıların "
            "değişkenliği hızla büyür (§6.10)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="dislanan-faktorle-kosullu-ortalama", note=_note("6.5", "(6.2)"),
        prompt=(
            "Ücret modelinde hata terimi $u = \\gamma\\,\\text{yetenek} + \\varepsilon$ biçimindedir; "
            "$\\mathbb{E}(\\text{yetenek} \\mid \\text{eğitim}) = a + b\\,\\text{eğitim}$ ve "
            "$\\mathbb{E}(\\varepsilon \\mid \\text{eğitim}) = 0$'dır. $\\mathbb{E}(u \\mid \\text{eğitim})$'i $\\gamma$, $a$, "
            "$b$ ve eğitim ($x$) cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\mathbb{E}(u \\mid \\text{eğitim})",
            symbols=(
                Symbol("gamma", "\\gamma", "yeteneğin ücretle ilişkisi", 0.2, 2, aliases=_greek("gamma", "γ", "gamma")),
                Symbol("a", "a", "koşullu ortalamanın sabiti", -2, 2),
                Symbol("b", "b", "yetenek ile eğitimin ilişkisi", 0.1, 1),
                Symbol("x", "x", "eğitim (yıl)", 6, 18, aliases=("\\texteğitim", "\\textEğitim", "eğitim", "egitim", "Eğitim", "Egitim")),
            ),
            answer="gamma*(a + b*x)",
            shown="\\gamma\\,(a + b\\,x)",
        ),
        explanation=(
            "Koşullu beklenen değer doğrusaldır: $\\mathbb{E}(u \\mid \\text{eğitim}) = \\gamma\\,"
            "\\mathbb{E}(\\text{yetenek} \\mid \\text{eğitim}) + \\mathbb{E}(\\varepsilon \\mid \\text{eğitim}) = "
            "\\gamma(a + "
            "b\\,x)$. γ ≠ 0 (yetenek ücretle ilişkili) ve b ≠ 0 (yetenek eğitimle ilişkili) ise koşullu ortalama "
            "eğitim düzeyine göre sistematik biçimde değişir ve A4 bozulur. γ = 0 ya da b = 0 ise bu kaynak eğime "
            "yanlılık getirmez: §6.6'daki iki koşulun ekonomik okumasıdır (§6.5)."
        ),
    ),
    Question(
        key="e02", concept="yanlilik-korelasyon-ve-standart-sapmalarla", note=_note("6.6", "(6.9)"),
        prompt=(
            "Tek eksik değişken durumunda (6.9)'a göre yanlılık $\\beta_2\\delta_1$'dir. Yardımcı regresyonun eğimi "
            "kovaryansın $X$'in varyansına oranıdır: $\\delta_1 = \\sigma_{XZ}/\\sigma_X^2$ (§0.9'daki "
            "$\\hat\\beta_1 = s_{xy}/s_x^2$'nin anakütle karşılığı); korelasyon ise $\\rho = \\sigma_{XZ}/(\\sigma_X "
            "\\sigma_Z)$'dir (§0.5). Yanlılığı $\\beta_2$, $\\rho$, $\\sigma_Z$ ve $\\sigma_X$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\operatorname{Bias}(\\widehat{\\widetilde\\beta}_1)",
            symbols=(
                Symbol("b2", "\\beta_2", "Z'nin Y ile ilişkisi", -3, 3, aliases=beta_aliases(2)),
                Symbol("rho", "\\rho", "Z ile X arasındaki korelasyon", -0.9, 0.9,
                       aliases=_RHO_XZ),
                Symbol("sz", "\\sigma_Z", "Z'nin standart sapması", 0.5, 4, aliases=_SIGMA_Z),
                Symbol("sx", "\\sigma_X", "X'in standart sapması", 0.5, 4, aliases=_SIGMA_X),
            ),
            answer="b2*rho*sz/sx",
            shown="\\beta_2\\,\\rho\\,\\frac{\\sigma_Z}{\\sigma_X}",
        ),
        explanation=(
            "İki tanım birleşince $\\delta_1 = \\sigma_{XZ}/\\sigma_X^2 = \\rho\\,\\sigma_X\\sigma_Z/\\sigma_X^2 = "
            "\\rho\\,\\sigma_Z/\\sigma_X$. Bu yüzden yanlılık $\\beta_2\\rho\\,\\sigma_Z/\\sigma_X$ olur. "
            "Dışlanan değişken sonuç için önemli değilse ($\\beta_2 = 0$) ya da X ile ilişkisizse ($\\rho = 0$) "
            "yanlılık ortadan kalkar; işareti $\\beta_2$ ile $\\rho$'nun işaretlerinin çarpımıdır (Tablo 6.4) (§6.6)."
        ),
    ),
    Question(
        key="e03", concept="kisa-model-sabitinin-beklenen-degeri", note=_note("6.6", "(6.5)", "(6.7)"),
        prompt=(
            "Gerçek model $Y = \\beta_0 + \\beta_1X + \\beta_2Z + u$ ve yardımcı ilişki $Z = \\delta_0 + \\delta_1X + "
            "r$ olsun (6.5 ve 6.7). $Z$ dışarıda bırakıldığında kısa modelin sabit teriminin beklenen değerini yazın. "
            "Gerekmeyen sembolleri kullanmayın."
        ),
        answer=Equation(
            lhs="\\mathbb{E}(\\widehat{\\widetilde\\beta}_0)",
            symbols=(
                Symbol("b0", "\\beta_0", "gerçek sabit", -5, 5, aliases=beta_aliases(0)),
                Symbol("b1", "\\beta_1", "X'in gerçek katsayısı", 0.1, 2, aliases=beta_aliases(1)),
                Symbol("b2", "\\beta_2", "Z'nin gerçek katsayısı", -3, 3, aliases=beta_aliases(2)),
                Symbol("d0", "\\delta_0", "yardımcı ilişkinin sabiti", -3, 3, aliases=_greek("delta", "δ", "delta", 0)),
                Symbol("d1", "\\delta_1", "yardımcı ilişkinin eğimi", -1, 1, aliases=_greek("delta", "δ", "delta", 1)),
            ),
            answer="b0 + b2*d0",
            shown="\\beta_0 + \\beta_2\\,\\delta_0",
        ),
        explanation=(
            "Yardımcı ilişkiyi gerçek modele yerleştirelim: $Y = (\\beta_0 + \\beta_2\\delta_0) + (\\beta_1 + "
            "\\beta_2\\delta_1)X + (\\beta_2 r + u)$. Kısa model bu doğruyu tahmin eder: eğimin beklenen değeri (6.8)'deki "
            "$\\beta_1 + \\beta_2\\delta_1$, sabitinki $\\beta_0 + \\beta_2\\delta_0$ olur. Z'nin X = 0'daki "
            "ortalama düzeyi ($\\delta_0$) sabit terime, X ile birlikte hareket eden kısmı ($\\delta_1$) eğime "
            "taşınır (§6.6)."
        ),
    ),
    Question(
        key="e04", concept="tam-baglantida-tanimlanan-birlesim", note=_note("6.9", "(6.13)"),
        prompt=(
            "(6.13)'teki gibi $X_3 = a + bX_1 + cX_2$ kimliği bütün gözlemlerde geçerli olsun. $Y = \\beta_0 + "
            "\\beta_1X_1 + \\beta_2X_2 + \\beta_3X_3 + u$ modelinde $X_3$ yerine kimlik yazılıyor ($X_1$ ve $X_2$ "
            "modelde kalıyor). Veri, $X_1$'in katsayısı olarak hangi parametre birleşimini belirleyebilir? Gerekmeyen "
            "sembolleri kullanmayın."
        ),
        answer=Equation(
            lhs="X_1\\text{'in katsayısı}",
            symbols=(
                Symbol("beta0", "\\beta_0", "sabit parametre", -5, 5, aliases=beta_aliases(0)),
                Symbol("beta1", "\\beta_1", "X₁'in parametresi", 0.1, 2, aliases=beta_aliases(1)),
                Symbol("beta2", "\\beta_2", "X₂'nin parametresi", 0.1, 2, aliases=beta_aliases(2)),
                Symbol("beta3", "\\beta_3", "X₃'ün parametresi", 0.1, 2, aliases=beta_aliases(3)),
                Symbol("a", "a", "kimliğin sabiti", -5, 5),
                Symbol("b", "b", "kimlikte X₁'in katsayısı", 0.5, 12),
                Symbol("c", "c", "kimlikte X₂'nin katsayısı", -3, 3),
            ),
            answer="beta1 + b*beta3",
            shown="\\beta_1 + b\\,\\beta_3",
        ),
        explanation=(
            "X₃'ü kimlikle değiştirelim: $\\beta_3X_3 = \\beta_3a + \\beta_3bX_1 + \\beta_3cX_2$. Model $(\\beta_0 + "
            "a\\beta_3) + (\\beta_1 + b\\beta_3)X_1 + (\\beta_2 + c\\beta_3)X_2$ biçimine iner; veri yalnız bu "
            "birleşimleri belirler, $\\beta_1$ ile $\\beta_3$ ayrı ayrı tanımlanamaz. Notlardaki aylık–yıllık gelir "
            "örneğinde (yıllık = 12 × aylık) aynı hesap $\\beta_1 + 12\\beta_2$ birleşimini verir (§6.9)."
        ),
    ),
    Question(
        key="e05", concept="vif-bagimsiz-kalan-degiskenlik", note=_note("6.12", "Kod 6.4", "Tablo 6.9"),
        prompt=(
            "Kod 6.4'teki yardımcı regresyonda $X_j$ diğer açıklayıcı değişkenlere göre tahmin edilir. Bu regresyonun "
            "toplam kareler toplamı $\\text{TKT}_j$, artık kareleri toplamı $\\text{HKT}_j$ ($X_j$'nin diğer "
            "değişkenlerden bağımsız kalan değişkenliği) olsun. $\\operatorname{VIF}_j$'yi bu iki büyüklük cinsinden "
            "yazın."
        ),
        answer=Equation(
            lhs="\\operatorname{VIF}_j",
            symbols=(
                Symbol("tkt", "\\text{TKT}_j", "yardımcı regresyonun toplam kareler toplamı", 50, 500,
                       aliases=_sum_of_squares("TKT")),
                Symbol("hkt", "\\text{HKT}_j", "yardımcı regresyonun artık kareleri toplamı", 1, 45,
                       aliases=_sum_of_squares("HKT")),
            ),
            answer="tkt/hkt",
            shown="\\frac{\\text{TKT}_j}{\\text{HKT}_j}",
        ),
        explanation=(
            "$R_j^2 = 1 - \\text{HKT}_j/\\text{TKT}_j$ olduğundan $1 - R_j^2 = \\text{HKT}_j/\\text{TKT}_j$ ve "
            "$\\operatorname{VIF}_j = 1/(1 - R_j^2) = \\text{TKT}_j/\\text{HKT}_j$. VIF, $X_j$'nin toplam "
            "değişkenliğinin diğer değişkenlerden bağımsız kalan kısmına oranıdır: bağımsız hareket azaldıkça VIF büyür. "
            "Tablo 6.9'da konut büyüklüğü için VIF 1,419: bağımsız kalan değişkenlik toplamın yaklaşık 1/1,419 ≈ %70'i "
            "(§6.12, Tablo 6.9)."
        ),
    ),
)


KONU06_QUIZ = QuestionSet(
    topic_key="konu06",
    title="Konu 6: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"6.{number}" for number in range(1, 14)),  # bölüm özeti numarasızdır
)
