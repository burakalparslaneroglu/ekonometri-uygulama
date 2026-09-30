"""Konu 11 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz ve Mini Quiz maddelerini
tekrar etmez: 15 + 2X − 4D additif modeli; 10 + 3X + 5D − 1,5DX modelinin grup denklemleri, üç noktadaki farkı,
kesişme noktası ve pozitif etkileşim yorumu; dört yapı için gerekli terimler ve D'siz model kısıtı; 1,5 + 0,06X −
0,20D + 0,015DX log modelinin eğimleri, log ve tam yüzde farkları, sıfır noktası ve γ₁ > 0 yorumu; 20 + 4X + 6D −
0,5DX modelinin 10'da merkezlenmesi; WAGE1 çıktısının maddeleri (eğimler, H₀: γ₁ = 0 kararı, 12 yılda log fark, 16
yılda tam yüzde fark, merkezlenmemiş modelde kadın katsayısının anlamı, nedensel ayrımcılık yorumu); HPRICE1
çıktısının maddeleri (eğimler, etkileşim işaretinin yorumu, 10.000 ve 15.000 kare fitte fark, eğim eşitliği kararı,
kolonyal katsayısının anlamsızlığı ile etkileşim, 50.000 kare fit); etkileşim hipotezlerini kurma (X = 0 ve X = 10'da
fark ve kısıt sayıları dahil); `Y ~ D * X + Z` formülünün terimleri, `Y ~ D + X + D:X + Z` eşdeğerliği,
`Y ~ D:X + Z`'nin dışarıda bıraktığı ana etkiler ve Tablo 11.6'nın maddeleri; çevrim içi–reklam bütünleşik yorumu ve
mini quizlerdeki sorular. Aynı becerileri yeni bağlamlarla ve yeni sayılarla sınar; bazı sorular Sezgi sekmesinin
deneylerine dayanır. Heteroskedastisiteye dayanıklı çıkarım Konu 12'nin konusudur; sorular geleneksel standart hataları
kullanır.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol, beta_aliases, beta_hat_aliases
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


def _gamma(index: int) -> tuple[str, ...]:
    """γ₀, γ₁ için öğrencinin yazabileceği biçimler (şapkalı ve şapkasız)."""

    number, subscript, hat = str(index), "₀₁"[index], "̂"
    return (f"\\gamma_{number}", f"\\gamma{number}", f"γ_{number}", f"γ{number}", f"γ{subscript}",
            f"gamma_{number}", f"gamma{number}", f"\\hat\\gamma_{number}", f"\\widehat\\gamma_{number}",
            f"\\hat\\gamma{number}", f"\\widehat\\gamma{number}", f"γ{hat}_{number}", f"γ{hat}{number}",
            f"γ{hat}{subscript}", f"gammahat_{number}", f"gammahat{number}", f"gamma{number}hat", f"ghat{number}",
            f"g{number}hat")


def _coefficient(name: str, index: int, low: float, high: float, meaning: str) -> Symbol:
    """β₀, β₁ ya da γ₀, γ₁ katsayısı; öğrenci şapkalı ya da şapkasız yazabilir."""

    if name == "b":
        return Symbol(f"b{index}", f"\\beta_{index}", meaning, low, high,
                      aliases=(*beta_aliases(index), *beta_hat_aliases(index), f"b_{index}"))
    return Symbol(f"g{index}", f"\\gamma_{index}", meaning, low, high, aliases=(*_gamma(index), f"g_{index}"))


_R2_UR = ("R_UR^2", "R_UR²", "R²_UR", "R2_UR", "R2UR", "R^2_UR", "R_ur^2", "R_ur²", "R²_ur", "R2_ur", "R2ur")
_R2_R = ("R_R^2", "R_R²", "R²_R", "R2_R", "R2R", "R^2_R", "R_r^2", "R_r²", "R²_r", "R2_r", "R2r")
"""R² yazımları (Konu 8 setiyle aynı); ``R^2_{UR}`` önce ``R_UR^2`` biçimine çevrilir (``core.quiz.expression.parse``)."""


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="additif-modelden-egim-farki-cikmaz", note=_note("11.1"),
        prompt=(
            "Bir araştırmacı $\\text{satış} = \\beta_0 + \\beta_1\\,\\text{reklam} + \\gamma_0\\,\\text{online} + u$ "
            "additif modelini tahmin ediyor ve raporuna “çevrim içi firmalarda reklamın satışa katkısı daha büyüktür” "
            "yazıyor. Bu iddia bu modelden çıkarılabilir mi?"
        ),
        answer=MultipleChoice(
            (
                "Hayır: model iki gruba aynı reklam eğimini dayatır; eğim farkı için etkileşim gerekir",
                "Evet: $\\gamma_0$ pozitif ve anlamlıysa çevrim içi firmaların eğimi de daha büyüktür",
                "Hayır: çevrim içi firmaların reklam eğimi $\\beta_1 + \\gamma_0$'dır; iddia bu toplamın testini gerektirir",
                "Evet: $R^2$ yüksekse additif model eğim farklarını da doğru yansıtır",
            ),
            correct=0,
        ),
        explanation=(
            "Additif modelde iki grubun doğruları paraleldir: reklam eğimi her iki grupta da $\\beta_1$'dir ve "
            "$\\gamma_0$ yalnız düzey (sabit) farkını ölçer; $\\beta_1 + \\gamma_0$ bir eğim değildir. Model eğim "
            "farkını tahmin etmediği için bu konuda hiçbir şey söyleyemez; soru $\\text{online} \\times \\text{reklam}$ "
            "etkileşimini ve onun katsayısının testini gerektirir. Ortak eğim varsayımı veriyle ayrıca değerlendirilir "
            "(§11.1)."
        ),
    ),
    Question(
        key="k02", concept="ic-ice-yapilarda-r2-siralamasi", note=_note("11.3", "Tablo 11.2"),
        prompt=(
            "Aynı veriyle Tablo 11.2'deki dört yapı tahmin ediliyor: (A) $1, X$; (B) $1, X, D$; (C) $1, X, DX$; "
            "(D) $1, X, D, DX$. $R^2$'ler için hangisi her veri setinde kesinlikle doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "(B)'nin $R^2$'si (C)'ninkinden büyük ya da ona eşittir",
                "(D)'nin $R^2$'si ne (B)'ninkinden ne (C)'ninkinden küçüktür",
                "(C)'nin $R^2$'si (A)'nınkinden küçük olabilir: gereksiz etkileşim uyumu bozar",
                "(A)'nın $R^2$'si (B)'ninkinden büyük olabilir: gereksiz kukla uyumu bozar",
            ),
            correct=1,
        ),
        explanation=(
            "(B) ve (C), (D)'nin birer kısıtlı hâlidir: (B) $\\gamma_1 = 0$, (C) $\\gamma_0 = 0$ kısıtını dayatır; (A) "
            "ikisini birden. Terim eklemek EKK'de artık kareleri toplamını artıramaz; bu yüzden genel model en az "
            "kısıtlı modeller kadar iyi uyar, (A) da (B) ve (C)'den iyi uyamaz. (B) ile (C) iç içe değildir: hangisinin "
            "daha yüksek $R^2$ vereceği veriye bağlıdır. Model seçimi yalnız $R^2$'ye dayanmaz; notlar genel modelle "
            "başlamayı en güvenli yol sayar, hiyerarşi ilkesi de etkileşim varken ana etkilerin tutulmasını ister "
            "(§11.3)."
        ),
    ),
    Question(
        key="k03", concept="tam-yuzde-fark-dogrusal-degismez", note=_note("11.4"),
        prompt=(
            "Bir $\\ln(\\text{ücret})$ etkileşim modelinde iki grup arasındaki log fark X = 8, 12 ve 16 yılda sırasıyla "
            "−0,10; −0,30 ve −0,50'dir (her dört yılda 0,20 azalıyor). Aynı noktalardaki tam yüzde farklar nedir?"
        ),
        answer=MultipleChoice(
            (
                "%−10; %−30; %−50: log farkı 100 ile çarpmak yeterlidir",
                "%10,52; %34,99; %64,87: üstel dönüşüm farkı büyütür",
                "%−9,52; %−25,92; %−39,35: adımlar eşit değildir",
                "%−9,52; %−19,04; %−28,56: her dört yılda 9,52 puan",
            ),
            correct=2,
        ),
        explanation=(
            "Tam yüzde fark $100[\\exp(\\gamma_0 + \\gamma_1X) - 1]$'dir: $100(e^{-0{,}10} - 1) \\approx -9{,}52$, "
            "$100(e^{-0{,}30} - 1) \\approx -25{,}92$, $100(e^{-0{,}50} - 1) \\approx -39{,}35$. Log fark X ile doğrusal "
            "değişse de yüzde fark doğrusal değişmez: adımlar 16,40 ve 13,43 puandır. %10,52… işaretleri ters "
            "çevirir; ×100 yaklaşık yorumdur ve büyük farklarda hata büyür (§11.4)."
        ),
    ),
    Question(
        key="k04", concept="belirli-noktada-farki-merkezlemeyle-sinama", note=_note("11.8"),
        prompt=(
            "$Y = \\beta_0 + \\beta_1X + \\gamma_0D + \\gamma_1DX + u$ modelinde (X merkezlenmemiş) araştırma sorusu "
            "şudur: “X = 5 noktasında iki grup farklı mı?” Bu soru en doğrudan nasıl sınanır?"
        ),
        answer=MultipleChoice(
            (
                "$\\gamma_0$ ve $\\gamma_1$'in t testleri ayrı ayrı yapılır; ikisi de reddedilmezse fark yoktur",
                "$H_0: \\gamma_0 = \\gamma_1 = 0$ ortak F testi yapılır; bu test X = 5'teki farkı sınar",
                "Sabit terimin t testi yapılır; sabit, X = 5'teki grup farkını gösterir",
                "X, 5 etrafında merkezlenir ($X - 5$) ve yeni modelde D katsayısının t testi yapılır",
            ),
            correct=3,
        ),
        explanation=(
            "X = 5'teki fark $\\gamma_0 + 5\\gamma_1$'dir; hipotez $H_0: \\gamma_0 + 5\\gamma_1 = 0$'dır. X'i 5 "
            "etrafında merkezlemek D katsayısını tam bu farka dönüştürür; onun t testi hipotezi doğrudan sınar (aynı "
            "sonuç doğrusal kısıt testiyle de alınır). $\\gamma_0$'ın t testi yalnız X = 0'daki farkı, ortak F testi "
            "iki doğrunun bütünüyle aynı olup olmadığını sınar (§11.8)."
        ),
    ),
    Question(
        key="k05", concept="iki-etkilesimli-formulun-terimleri", note=_note("11.9"),
        prompt=(
            "`statsmodels`'te `y ~ D * X + D * Z` formülü modele sabit dışında hangi terimleri koyar?"
        ),
        answer=MultipleChoice(
            (
                "D, X, Z ve `D:X`: ikinci `*` yeni bir etkileşim eklemez",
                "D, X, `D:X`, Z ve `D:Z`: beş terim; D bir kez girer",
                "D, X, `D:X`, D, Z ve `D:Z`: D iki kez girer, altı terim",
                "Yalnız `D:X` ve `D:Z`: `*` yalnız çarpım terimlerini ekler",
            ),
            correct=1,
        ),
        explanation=(
            "`D * X`, D ve X ana etkilerini ve `D:X` etkileşimini; `D * Z`, D ve Z ana etkilerini ve `D:Z` etkileşimini "
            "ekler. Formül arayüzü aynı terimi iki kez koymaz: D bir kez girer ve model $1 + D + X + D{:}X + Z + D{:}Z$ "
            "olur. Bu modelde hem X hem Z eğimi gruba göre değişebilir; D katsayısı X = 0 ve Z = 0 noktasındaki "
            "farktır (§11.9)."
        ),
    ),
    Question(
        key="k06", concept="anlamsiz-ana-etki-grup-farki-yok-demek-degil", note=_note("11.10"),
        prompt=(
            "Merkezlenmiş bir etkileşim modelinde kukla katsayısı merkez noktasında 0,8 (p = 0,41), etkileşim katsayısı "
            "0,6 (p = 0,003) çıkıyor. En doğru yorum hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Gruplar arasında fark yoktur: kukla katsayısı anlamsızdır",
                "Kukla anlamsız olduğundan etkileşim de yorumlanamaz; ikisi birlikte modelden çıkarılmalıdır",
                "Merkezdeki fark sıfırdan ayırt edilemez ama fark X ile değişir; başka X'lerde anlamlı olabilir",
                "İki p-değeri çeliştiğine göre model yanlış kurulmuştur",
            ),
            correct=2,
        ),
        explanation=(
            "Kukla katsayısı yalnız merkez noktasındaki farktır; etkileşimli modelde fark $\\gamma_0 + \\gamma_1X^c$ "
            "ile X'e göre değişir. Farkın merkezde sıfırdan ayrılamaması, merkezden uzaktaki farkların da sıfır olduğu "
            "anlamına gelmez; eğim farkının anlamlı olması farkın X ile değiştiğine kanıttır. İki test farklı "
            "hipotezleri sınar; çelişki yoktur (Sık Yapılan Hatalar, madde 6) (§11.10)."
        ),
    ),
    Question(
        key="k07", concept="sutunlar-arasinda-kukla-katsayisinin-anlami", note=_note("11.9", "Tablo 11.6"),
        prompt=(
            "Tablo 11.6'da Kadın katsayısı additif sütunda −0,3011, etkileşimli sütunda −0,2973'tür. Bir öğrenci "
            "“etkileşim eklenince cinsiyet farkı küçüldü” diyor. Hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "İki sayı farklı büyüklükleri ölçer: biri her eğitimde ortak fark, öteki 12 yıldaki fark",
                "Öğrenci haklıdır: iki katsayı aynı farkı ölçer ve fark 0,0038 log birim küçülmüştür",
                "İki katsayı aynı farkı ölçer; aradaki 0,0038'lik fark yalnız yuvarlamadan kaynaklanır",
                "Additif model yanlış kurulduğu için −0,3011 hiçbir grup farkını ölçmez, yorumlanamaz",
            ),
            correct=0,
        ),
        explanation=(
            "Additif modelde kadın katsayısı ortak eğim varsayımıyla her eğitim düzeyinde aynı kabul edilen farktır. "
            "Etkileşimli modelde eğitim 12 yıl etrafında merkezli olduğundan katsayı 12 yıllık eğitimdeki farktır; "
            "fark eğitimle $-0{,}0072$ hızında değişir. Katsayıları sütunlar arasında karşılaştırmadan önce her "
            "sütunda katsayının neyi ölçtüğü okunur (§11.9)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="negatif-etkilesim-negatif-egim-degildir", note=_note("11.2", "Tablo 11.1"),
        prompt=(
            "Etkileşimli modelde $\\gamma_1 < 0$ ise $D = 1$ grubunda X ile Y arasındaki ilişki negatiftir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "$\\gamma_1$ iki eğim arasındaki farktır; $D = 1$ grubunun eğimi $\\beta_1 + \\gamma_1$'dir. Örneğin "
            "$\\beta_1 = 0{,}09$ ve $\\gamma_1 = -0{,}01$ ise $D = 1$ grubunun eğimi 0,08'dir: pozitif, yalnız referans "
            "gruptakinden küçük. Eğimin işareti iki katsayının toplamından okunur (Tablo 11.1) (§11.2)."
        ),
    ),
    Question(
        key="d02", concept="ana-etki-ortalama-egim-degildir", note=_note("11.3"),
        prompt=(
            "`y ~ D * X` modelinde X'in katsayısı, iki grubun X eğimlerinin ortalamasıdır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Etkileşim varken X'in ana etkisi $D = 0$ referans grubunun eğimidir; $D = 1$ grubunun eğimi ona etkileşim "
            "katsayısı eklenerek bulunur. Hiyerarşi ilkesi kutusunun uyardığı gibi ana etkiler etkileşim varken “genel "
            "etki” değil, diğer değişkenin referans değerindeki koşullu etkilerdir. Referans grup değiştirilirse X "
            "katsayısı da değişir (§11.3)."
        ),
    ),
    Question(
        key="d03", concept="merkezleme-kukla-sh-degisir-etkilesim-sh-degismez", note=_note("11.5"),
        prompt=(
            "Merkezleme noktası c değiştirildiğinde kukla katsayısının standart hatası değişebilir; etkileşim "
            "katsayısının standart hatası ise değişmez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Merkezleme modelin parametrelerini başka bir referans noktasında ifade eder: kukla katsayısı X = c'deki farka ($\\gamma_0 + \\gamma_1c$) "
            "dönüşür ve belirsizliği c'ye bağlıdır; veri merkezinde en küçük, veri aralığının dışında büyüktür. "
            "Etkileşim katsayısı ve standart hatası, tahmin edilen değerler ve $R^2$ değişmez. Konu 11 Sezgi Deney 1'in "
            "varsayılan ayarında fark tahmininin ampirik standart sapması c = 0'da 0,209, X = 13'te 0,047'dir (§11.5)."
        ),
    ),
    Question(
        key="d04", concept="anlamsiz-etkilesim-esitligi-kanitlamaz", note=_note("11.6", "Kod 11.2"),
        prompt=(
            "WAGE1 etkileşim modelinde etkileşim katsayısı anlamsız olduğu için ($-0{,}0072$, p = 0,5935) eğitimin log "
            "ücretle ilişkisinin kadın ve erkek çalışanlarda aynı olduğu kanıtlanmıştır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Reddedememek eşitliği kanıtlamaz. Etkileşimin yüzde 95 güven aralığı $-0{,}0072 \\pm 1{,}965 \\times 0{,}0136 "
            "\\approx [-0{,}0339;\\ 0{,}0195]$'tir (yuvarlanmamış değerlerle [−0,0339; 0,0194]): sıfırı içerir, ama "
            "kadınlarda eğitimin getirisinin yılda yaklaşık 3,4 yüzde puana kadar daha düşük ya da 1,9 puana kadar daha "
            "yüksek olmasıyla da uyumludur. Doğru ifade, bu örneklemde eğim farkına ilişkin istatistiksel kanıt "
            "bulunmadığıdır (§11.6)."
        ),
    ),
    Question(
        key="d05", concept="ortak-f-merkezlemeden-bagimsiz", note=_note("11.8"),
        prompt=(
            "İki grubun doğrularının aynı olduğu hipotezinin ($H_0: \\gamma_0 = \\gamma_1 = 0$) ortak F testi, X'in hangi "
            "noktada merkezlendiğine bağlı değildir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Merkezleme hangi c seçilirse seçilsin kısıtsız modelin ($1, X, D, DX$) ve kısıtlı modelin ($1, X$) "
            "tahmin edilen değerlerini değiştirmez; iki modelin SSR'leri ve F istatistiği aynı kalır. Buna karşılık "
            "$\\gamma_0$'ın t testi merkez noktasındaki farkı sınar ve c ile değişir. Konu 11 Sezgi Deney 3'te c "
            "değiştirildiğinde F'nin ret oranı aynı kalır, $\\gamma_0$'ınki değişir (§11.8)."
        ),
    ),
    Question(
        key="d06", concept="etkilesim-nedensel-mekanizma-degildir", note=_note("11.10"),
        prompt=(
            "Gözlemsel bir yatay kesit verisinde sendika üyeliği × kıdem etkileşimi anlamlı çıkarsa, sendika üyeliğinin "
            "kıdemin ücretle ilişkisini nedensel olarak değiştirdiği gösterilmiş olur."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Etkileşim koşullu bir örüntüyü, yani kıdem–ücret ilişkisinin sendikalı ve sendikasız çalışanlarda farklı "
            "olduğunu modeller. Farkın nedeni sektör, firma büyüklüğü, iş güvencesi ya da gözlenmeyen özellikler "
            "olabilir; sendikalı çalışanlar bu özellikler bakımından farklı seçilmiş olabilir. Nedensel yorum için "
            "araştırma tasarımı ve ek varsayımlar gerekir (Sık Yapılan Hatalar, madde 10) (§11.10)."
        ),
    ),
    Question(
        key="d07", concept="additif-egim-agirlikli-ortalama", note=_note("11.1"),
        prompt=(
            "Gerçek model $Y = \\beta_0 + \\beta_1X + \\gamma_0D + \\gamma_1DX + u$, D ile X bağımsız ve $D = 1$ grubunun "
            "payı p ise, yalnız $1, X, D$ içeren additif modelin X eğimi iki grup eğiminin grup paylarıyla "
            "ağırlıklandırılmış ortalamasını hedefler."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Additif modelin X eğimi büyük örneklemde $\\beta_1 + p\\,\\gamma_1$ değerine yaklaşır: referans eğimi "
            "$\\beta_1$ ile $D = 1$ grubunun eğimi $\\beta_1 + \\gamma_1$'in $1 - p$ ve p ağırlıklı ortalaması. Tek bir "
            "eğim iki ilişkiyi özetler; hiçbir grubun eğimi değildir. Konu 11 Sezgi Deney 2 bunu gösterir: varsayılan "
            "ayarda 0,075; p = 0,1'de 0,095, p = 0,9'da 0,055 (§11.1)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="merkezlenmemis-kukla-katsayisi-veri-disinda", note=_note("11.7", "Kod 11.4"),
        prompt=(
            "Kod 11.4'teki HPRICE1 modelinde arsa büyüklüğü merkezlenmeden, yalnız 1.000 kare fit biriminde "
            "(`lotsize/1000`) kullanılsaydı kolonyal katsayısı notlardaki yuvarlanmış katsayılarla **(1)** bin dolar "
            "olurdu (virgülden sonra dört basamak). Bu katsayının standart hatası 22,20 ise t istatistiği **(2)** olur "
            "(virgülden sonra üç basamak)."
        ),
        answer=FillBlanks((NumberBlank(52.3079, 0.0006, "52,3077"), NumberBlank(2.356, 0.0015, "2,356"))),
        explanation=(
            "Merkezlenmemiş modelde kolonyal katsayısı arsa büyüklüğü sıfırken farktır: $8{,}2437 - 4{,}4064 \\times "
            "(0 - 10) = 52{,}3077$ (yuvarlanmamış katsayılarla 52,3081); $t = 52{,}3077/22{,}20 \\approx 2{,}356$ ve "
            "p ≈ 0,021. Model sıfır kare fitlik arsada kolonyal konutların 52 bin dolar daha pahalı olduğunu ve farkın "
            "“anlamlı” olduğunu söyler; oysa örneklemde en küçük arsa 1.000 kare fittir ve 10.000 kare fitteki fark "
            "(8,24) p = 0,5709 ile sıfırdan ayırt edilemez. Kukla katsayısının anlamlılığı referans noktasına aittir; "
            "veri dışındaki nokta ekstrapolasyondur (§11.7)."
        ),
    ),
    Question(
        key="b02", concept="hprice1-farkin-sifirlandigi-arsa", note=_note("11.7", "Kod 11.4"),
        prompt=(
            "Kod 11.4'teki HPRICE1 modelinde tahmin edilen kolonyal–diğer fiyat farkı $8{,}2437 - 4{,}4064\\,"
            "\\text{lotsize10k}$'dır; `lotsize10k`, 10.000 kare fit etrafında merkezlenmiş ve 1.000 kare fit "
            "biriminde ölçülmüş arsa büyüklüğüdür. Fark `lotsize10k` = **(1)** iken sıfırdır (virgülden sonra üç basamak); bu, "
            "yaklaşık **(2)** kare fitlik arsaya karşılık gelir (tam sayı)."
        ),
        answer=FillBlanks((NumberBlank(1.871, 0.0015, "1,871"), NumberBlank(11871.0, 2.0, "11.871"))),
        explanation=(
            "$8{,}2437 - 4{,}4064\\,\\text{lotsize10k} = 0$ eşitliğinden $\\text{lotsize10k} = 8{,}2437/4{,}4064 "
            "\\approx 1{,}871$; $\\text{lotsize} = 10.000 + 1.000 \\times 1{,}871 \\approx 11.871$ kare fit. Daha "
            "küçük arsalarda model kolonyal konutlar için daha yüksek, daha büyüklerde daha düşük fiyat tahmin eder "
            "(Tablo 11.5). Nokta veri aralığının içindedir, ama çevresindeki farklar sıfırdan ayırt edilemez; nokta "
            "tahmini belirsizliğiyle birlikte okunur (§11.7)."
        ),
    ),
    Question(
        key="b03", concept="olcek-degisince-katsayilar", note=_note("11.7", "Kod 11.4"),
        prompt=(
            "Kod 11.4'teki modelde arsa büyüklüğü yine 10.000 kare fit etrafında merkezlenip 1.000 yerine 100 kare fit "
            "biriminde ölçülseydi (`(lotsize - 10000)/100`), etkileşim katsayısı **(1)** (virgülden sonra dört basamak), "
            "kolonyal katsayısı **(2)** olurdu (virgülden sonra dört basamak)."
        ),
        answer=FillBlanks((NumberBlank(-0.4406, 0.00011, "−0,4406"), NumberBlank(8.2437, 0.00011, "8,2437"))),
        explanation=(
            "Birim 10 kat küçülünce aynı arsa farkı 10 kat büyük bir sayıyla ölçülür; eğim ve eğim farkı 10'a bölünür: "
            "$-4{,}4064/10 \\approx -0{,}4406$ (kolonyal olmayanların eğimi de yaklaşık 0,6002 olur). Merkez "
            "değişmediği için kolonyal katsayısı yine 10.000 kare fitteki farktır: 8,2437. Ölçekleme yorumu "
            "değiştirmeden birimleri değiştirir; t istatistikleri ve $R^2$ aynı kalır (§11.7)."
        ),
    ),
    Question(
        key="b04", concept="ayri-regresyonlardan-etkilesim-katsayilari", note=_note("11.2"),
        prompt=(
            "İki grubun ayrı ayrı tahmin edilen basit regresyonları $D = 0$: $\\widehat Y = 2{,}0 + 0{,}5X$ ve "
            "$D = 1$: $\\widehat Y = 3{,}5 + 0{,}3X$'tir. Aynı veriyle yalnız $D$, $X$ ve $DX$ içeren `y ~ D * X` modeli "
            "tahmin edilirse $\\widehat\\gamma_0$ = **(1)** ve $\\widehat\\gamma_1$ = **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(1.5, 0.005, "1,5"), NumberBlank(-0.2, 0.005, "−0,2"))),
        explanation=(
            "Yalnız $D$, $X$ ve $DX$ içeren etkileşimli model her gruba kendi sabitini ve eğimini verir; EKK iki grubun "
            "ayrı regresyonlarıyla aynı doğruları bulur. $\\widehat\\beta_0 = 2{,}0$, $\\widehat\\beta_1 = 0{,}5$; "
            "$\\widehat\\gamma_0 = 3{,}5 - 2{,}0 = 1{,}5$ ve $\\widehat\\gamma_1 = 0{,}3 - 0{,}5 = -0{,}2$. Modele "
            "gruplar için ortak eğimli başka bir değişken eklenirse bu eşitlik bozulur (§11.2)."
        ),
    ),
    Question(
        key="b05", concept="iki-dogru-testinin-serbestlik-dereceleri", note=_note("11.8", "Kod 11.2"),
        prompt=(
            "Kod 11.2'deki WAGE1 etkileşim modelinde (n = 526; kadın kuklası `female`, merkezlenmiş eğitim `educ12`, "
            "etkileşim `female:educ12`, deneyim `exper` ve kıdem `tenure`) $H_0: \\gamma_0 = \\gamma_1 = 0$ ortak F "
            "testi için kısıtlı model de tahmin ediliyor. Kısıtlı modelin artık serbestlik derecesi **(1)**, kısıtsız "
            "modelin artık serbestlik derecesi **(2)** olur."
        ),
        answer=FillBlanks((NumberBlank(522.0, 0.5, "522"), NumberBlank(520.0, 0.5, "520"))),
        explanation=(
            "Kısıtlı model `female` ve `female:educ12` terimlerini çıkarır; sabitle birlikte `educ12`, `exper` ve "
            "`tenure` kalır ($k = 3$): $526 - 3 - 1 = 522$. Kısıtsız modelde $k = 5$: $526 - 5 - 1 = 520$. İki "
            "serbestlik derecesinin farkı kısıt sayısıdır ($q = 522 - 520 = 2$); notlardaki $F = 32{,}785$ "
            "$F(2, 520)$ dağılımıyla değerlendirilir. Eğim eşitliği testinde (tek kısıt) pay 1 olur ve $F = t^2$'dir "
            "(§11.8)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="additif-kukla-katsayisinin-hedefi", note=_note("11.1"),
        prompt=(
            "Gerçek model $Y = \\beta_0 + \\beta_1X + \\gamma_0D + \\gamma_1DX + u$'dur ve D ile X bağımsızdır; X'in "
            "ortalaması $\\mu$'dür. Additif model ($1, X, D$) tahmin edilirse D katsayısı hangi büyüklüğü hedefler? "
            "$\\gamma_0$, $\\gamma_1$ ve $\\mu$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\gamma_{\\text{additif}}",
            symbols=(
                _coefficient("g", 0, -2, 2, "etkileşimli modelde X = 0'daki fark"),
                _coefficient("g", 1, -0.5, 0.5, "eğim farkı"),
                Symbol("mu", "\\mu", "X'in ortalaması", 1, 20, aliases=("\\mu", "μ", "µ", "mu")),
            ),
            answer="g0 + g1*mu",
            shown="\\gamma_0 + \\gamma_1\\,\\mu",
        ),
        explanation=(
            "$\\gamma_1DX = \\gamma_1\\mu D + \\gamma_1D(X - \\mu)$ yazılabilir. Atlanan $\\gamma_1D(X - \\mu)$ "
            "teriminin D ve X üzerine yardımcı regresyonunda D'nin katsayısı sıfırdır (D ile X bağımsız); bu yüzden "
            "additif modelin D katsayısı X'in ortalamasındaki farkı, $\\gamma_0 + \\gamma_1\\mu$'yü hedefler ve bu "
            "farkı her X'e yayar. Konu 11 Sezgi Deney 2'nin varsayılan ayarında additif D katsayısının ortalaması "
            "−0,451, X = 13'teki gerçek fark −0,450'dir (§11.1)."
        ),
    ),
    Question(
        key="e02", concept="belirli-yuzde-farkin-x-degeri", note=_note("11.4"),
        prompt=(
            "$\\ln(Y)$ etkileşim modelinde gruplar arasındaki log fark $\\gamma_0 + \\gamma_1X$'tir ($\\gamma_1 \\neq 0$). "
            "Tam yüzde farkın P'ye eşit olduğu X değerini $\\gamma_0$, $\\gamma_1$ ve P cinsinden yazın."
        ),
        answer=Equation(
            lhs="X_P",
            symbols=(
                _coefficient("g", 0, -0.5, 0.5, "X = 0'daki log fark"),
                _coefficient("g", 1, 0.01, 0.05, "eğim farkı"),
                Symbol("P", "P", "tam yüzde fark", -30, 30),
            ),
            answer="(log(1 + P/100) - g0)/g1",
            shown="\\frac{\\ln\\left(1 + \\frac{P}{100}\\right) - \\gamma_0}{\\gamma_1}",
        ),
        explanation=(
            "$100[\\exp(\\gamma_0 + \\gamma_1X) - 1] = P$ eşitliğinden $\\gamma_0 + \\gamma_1X = \\ln(1 + P/100)$ ve "
            "$X = [\\ln(1 + P/100) - \\gamma_0]/\\gamma_1$. P = 0 için sonuç log farkın sıfır olduğu $-\\gamma_0/\\gamma_1$ "
            "noktasıdır. Bulunan X veri aralığının dışındaysa sonuç ekstrapolasyondur ve dikkatle raporlanır (§11.4)."
        ),
    ),
    Question(
        key="e03", concept="merkezlenmis-sabit", note=_note("11.5"),
        prompt=(
            "Özgün model $Y = \\beta_0 + \\beta_1X + \\gamma_0D + \\gamma_1DX + u$, merkezlenmiş model "
            "$Y = \\alpha_0 + \\beta_1(X - c) + \\gamma_0^{c}D + \\gamma_1D(X - c) + u$'dur. Merkezlenmiş modelin sabiti "
            "$\\alpha_0$'ı $\\beta_0$, $\\beta_1$ ve c cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\alpha_0",
            symbols=(
                _coefficient("b", 0, -5, 5, "özgün modelin sabiti"),
                _coefficient("b", 1, -2, 2, "referans grubun eğimi"),
                Symbol("c", "c", "merkez noktası", 1, 20),
            ),
            answer="b0 + b1*c",
            shown="\\beta_0 + \\beta_1\\,c",
        ),
        explanation=(
            "İki model aynı doğruları tanımlar. $D = 0$ ve $X = c$ iken merkezlenmiş model $\\alpha_0$'ı, özgün model "
            "$\\beta_0 + \\beta_1c$'yi verir; ikisi eşittir. $\\alpha_0$, referans grubun X = c'deki koşullu beklenen "
            "değeridir; tahmini $\\widehat\\alpha_0$ bu noktadaki tahmin edilen değerdir. Aynı yolla $\\gamma_0^{c} = "
            "\\gamma_0 + \\gamma_1c$ bulunur; eğimler ve etkileşim katsayısı değişmez (§11.5)."
        ),
    ),
    Question(
        key="e04", concept="iki-dogru-esitligi-f-r2-bicimi", note=_note("11.8"),
        prompt=(
            "`y ~ D * X` modeli ($R^2_{UR}$) ile `y ~ X` modeli ($R^2_{R}$) aynı n gözlemle tahmin ediliyor. "
            "$H_0: \\gamma_0 = \\gamma_1 = 0$ için F istatistiğini $R^2_{UR}$, $R^2_{R}$ ve n cinsinden yazın."
        ),
        answer=Equation(
            lhs="F",
            symbols=(
                Symbol("Ru", "R^2_{UR}", "kısıtsız modelin R²'si", 0.3, 0.6, aliases=_R2_UR),
                Symbol("Rr", "R^2_{R}", "kısıtlı modelin R²'si", 0.05, 0.29, aliases=_R2_R),
                Symbol("n", "n", "gözlem sayısı", 30, 500),
            ),
            answer="((Ru - Rr)/2)/((1 - Ru)/(n - 4))",
            shown="\\frac{(R^2_{UR} - R^2_{R})/2}{(1 - R^2_{UR})/(n - 4)}",
        ),
        explanation=(
            "İki kısıt ($q = 2$) vardır; kısıtsız modelde üç açıklayıcı değişken ($D$, $X$, $DX$; $k = 3$) bulunduğundan "
            "payda serbestlik derecesi $n - 3 - 1 = n - 4$'tür. Aynı bağımlı değişken ve aynı gözlemlerle SSR'li biçim "
            "$R^2$'li biçime eşittir: $\\text{SSR} = (1 - R^2)\\,\\text{TKT}$ ve TKT sadeleşir. Sonuç $F(2, n - 4)$ "
            "dağılımıyla değerlendirilir (§11.8)."
        ),
    ),
    Question(
        key="e05", concept="ozgun-birimde-kosullu-fark", note=_note("11.7"),
        prompt=(
            "Kod 11.4'teki HPRICE1 modelinde arsa büyüklüğü `lotsize10k` olarak girer: 10.000 kare fit etrafında "
            "merkezlenmiş ve 1.000 kare fit biriminde ölçülmüştür. L arsa büyüklüğünü bin kare fit cinsinden göstersin "
            "(ör. 12.500 kare fit için L = 12,5). Kolonyal–diğer fiyat farkını $\\gamma_0$ (kolonyal katsayısı), "
            "$\\gamma_1$ (etkileşim katsayısı) ve L cinsinden yazın (sayıları binlik ayırıcı kullanmadan yazın)."
        ),
        answer=Equation(
            lhs="\\text{fark}(L)",
            symbols=(
                _coefficient("g", 0, -20, 20, "kolonyal katsayısı"),
                _coefficient("g", 1, -8, 8, "etkileşim katsayısı"),
                Symbol("L", "L", "arsa büyüklüğü (bin kare fit)", 3, 30),
            ),
            answer="g0 + g1*(L - 10)",
            shown="\\gamma_0 + \\gamma_1\\,(L - 10)",
        ),
        explanation=(
            "Modelde $\\text{lotsize10k} = (\\text{lotsize} - 10.000)/1.000$'dir; $\\text{lotsize} = 1.000L$ olduğundan "
            "$\\text{lotsize10k} = L - 10$ ve fark $\\gamma_0 + \\gamma_1\\,\\text{lotsize10k} = \\gamma_0 + "
            "\\gamma_1(L - 10)$ olur. L = 12,5 için $8{,}2437 - 4{,}4064 \\times 2{,}5 \\approx -2{,}77$ bin dolar "
            "(Tablo 11.5). Katsayılar hangi merkez ve birimle "
            "tahmin edildiyse fark o dönüşümle geri yazılır (§11.7)."
        ),
    ),
)


KONU11_QUIZ = QuestionSet(
    topic_key="konu11",
    title="Konu 11: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"11.{number}" for number in range(1, 11)),  # §11.11 bölüm özetidir
)
