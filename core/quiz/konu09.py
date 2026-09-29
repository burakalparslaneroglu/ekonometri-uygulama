"""Konu 9 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların Egzersiz ve Mini Quiz maddelerini
tekrar etmez: beş modelin parametrelerde doğrusallığı; satış–reklam modelinde birim dönüşümleri ve 0,35'lik
standartlaştırılmış katsayı; β = 0,06, üç birimlik değişim, β = −0,10 ve 0,45'lik log–log katsayısı; 10 + 4X −
0,10X² modelinin marjinal etkileri, tam değişimi ve dönüm noktası; WAGE1 karesel modelinin yedi maddesi; 5 + 3X −
0,05X² modelinin 20'de merkezlenmesi; Tablo 9.7'nin yedi maddesi; Tablo 9.8'in sekiz maddesi (15 yıllık deneyimde
marjinal etki dahil); reklam–satış örneğinde yanlış biçim; altı yanlış cümlenin düzeltilmesi ve mini quizlerdeki
sorular. Aynı becerileri yeni bağlamlarla ve yeni sayılarla sınar. Artık grafikleriyle tanı Konu 12'nin, etkileşim
terimleri Konu 11'in konusudur; sorular bunları kullanmaz.
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


def _estimate(index: int, low: float, high: float, meaning: str) -> Symbol:
    """Tahmin edilmiş katsayı ``b_j``; öğrenci β̂ ya da β yazımıyla da girebilir."""

    return Symbol(f"b{index}", f"b_{index}", meaning, low, high,
                  aliases=(*beta_hat_aliases(index), *beta_aliases(index)))


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="standart-katsayi-yayilima-bagli", note=_note("9.2"),
        prompt=(
            "İki ülkenin verisiyle ayrı ayrı tahmin edilen $\\ln(\\text{ücret})$ modellerinde eğitimin katsayısı "
            "aynıdır: 0,08. $\\ln(\\text{ücret})$'in standart sapması iki ülkede de 0,5'tir; eğitimin standart sapması "
            "A ülkesinde 2 yıl, B ülkesinde 4 yıldır. Eğitimin standartlaştırılmış katsayıları için hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "İkisi de 0,08'dir: standartlaştırma katsayıyı değiştirmez",
                "A'da 0,64, B'de 0,32: yayılım büyüdükçe katsayı küçülür",
                "A'da 0,32, B'de 0,64: eğim aynı, B'de yayılım daha büyük",
                "Karşılaştırılamaz: standart katsayı yalnız tek örneklemde tanımlıdır",
            ),
            correct=2,
        ),
        explanation=(
            "Y ve X standartlaştırılınca eğim ölçek kuralıyla $s_X/s_Y$ ile çarpılır: A'da 0,08 × 2/0,5 = 0,32, B'de "
            "0,08 × 4/0,5 = 0,64. Bir yıllık eğitim farkının log ücretle ilişkisi iki ülkede aynıdır; B'de eğitim daha "
            "çok değiştiği için bir standart sapmalık fark daha çok yıl demektir. Standartlaştırılmış katsayı ilişkinin "
            "gücünü değişkenin örneklemdeki yayılımıyla birlikte taşır; nedensel önem sıralaması değildir. Konu 9 Sezgi "
            "Deney 1 aynı noktayı gösterir (§9.2)."
        ),
    ),
    Question(
        key="k02", concept="tam-yuzde-azalista-asimetrik", note=_note("9.3", "Tablo 9.4"),
        prompt=(
            "Log–düzey modelinde $\\beta = 0{,}08$'dir. Tablo 9.4'e göre $X$ 4 birim arttığında $Y$ tam hesapla %37,71 "
            "artar. $X$ 4 birim azalırsa $Y$'deki tam yüzde değişim nedir?"
        ),
        answer=MultipleChoice(
            (
                "%−37,71",
                "%−32,00",
                "%−33,32",
                "%−27,39",
            ),
            correct=3,
        ),
        explanation=(
            "Tam formül $100(e^{\\beta\\Delta X} - 1)$ azalışta da geçerlidir: $\\Delta X = -4$ için "
            "$100(e^{-0{,}32} - 1) \\approx -27{,}39$. %−37,71 artıştaki değişimin işaretini çevirir, %−32,00 "
            "yaklaşık yorumdur, %−33,32 bir birimlik tam artışın (%8,33) dört katını alır. $e^{-0{,}32} = 1/e^{0{,}32}$ olduğundan azalış artışın tersidir: "
            "Y %37,71 artıp aynı 4 birimlik azalışla başlangıç düzeyine dönerse dönüş, yüksek düzeye göre yaklaşık "
            "%27,4'lük düşüştür. Yüzde değişim başlangıç düzeyine göre hesaplandığı için artış ve azalış simetrik "
            "değildir; yaklaşık yorum (±%32) bu farkı göstermez (§9.3)."
        ),
    ),
    Question(
        key="k03", concept="pozitif-karesel-terim-disarida-donum", note=_note("9.4"),
        prompt=(
            "Tahmin edilen model $\\widehat Y = 2 + 0{,}5X + 0{,}02X^2$'dir ve örneklemde $X$ 0 ile 20 arasındadır. "
            "Hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Eğri X = 12,5'te tepe yapar; bu noktadan sonra Y azalır",
                "Etki X arttıkça büyür; dönüm noktası (−12,5) veri aralığı dışındadır",
                "X² katsayısı çok küçük olduğu için etki her düzeyde yaklaşık 0,5'tir",
                "Eğri X = −12,5'te tepe yapar; veri aralığında etki X arttıkça küçülür",
            ),
            correct=1,
        ),
        explanation=(
            "Marjinal etki $0{,}5 + 2(0{,}02)X = 0{,}5 + 0{,}04X$'tir: X = 0'da 0,5, X = 20'de 1,3. $\\widehat\\beta_2 > "
            "0$ eğrinin yukarı doğru büküldüğünü gösterir; dönüm noktası $-0{,}5/(2 \\times 0{,}02) = -12{,}5$ bir diptir "
            "ve gözlenen aralığın dışında kalır. Veri aralığında eğrinin yalnız artan ve giderek dikleşen bölümü "
            "görülür. Küçük görünen 0,02, X² ile çarpıldığı için 20 birimde etkiyi 0,5'ten 1,3'e çıkarır (§9.4)."
        ),
    ),
    Question(
        key="k04", concept="donum-noktasi-sonrasi-veri-destegi", note=_note("9.5"),
        prompt=(
            "WAGE1 karesel modelinde dönüm noktası deneyim için yaklaşık 24,76, kıdem için yaklaşık 30,15 yıldır. "
            "Deneyimi dönüm noktasını aşan 147, kıdemi dönüm noktasını aşan 6 çalışan vardır. Hangisi doğrudur?"
        ),
        answer=MultipleChoice(
            (
                "Deneyim eğrisinin azalan kısmı veride görece iyi desteklenir; kıdeminki çok az gözleme dayanır",
                "İki dönüm noktası da veri aralığı dışında kaldığı için iki eğrinin azalan kısmı da hesaplanamaz",
                "Kıdemin dönüm noktası daha büyük olduğu için kıdem eğrisinin azalan kısmı daha güvenilirdir",
                "Dönüm noktasını aşan çalışanlarda ek deneyim ya da ek kıdem ücreti nedensel olarak düşürür",
            ),
            correct=0,
        ),
        explanation=(
            "İki dönüm noktası da gözlenen aralığın içindedir (deneyim 1–51, kıdem 0–44 yıl). Deneyimde 526 çalışandan "
            "147'si dönüm noktasının ötesindedir; eğrinin azalan kısmı gerçek gözlemlerle desteklenir. Kıdemde yalnız 6 "
            "çalışan vardır: azalan kısım büyük ölçüde karesel biçimin veriye dayattığı şekilden gelir ve kırılgandır. "
            "Dönüm noktasının büyüklüğü güvenilirlik ölçüsü değildir; gözlemsel veride azalan kısım nedensel okunmaz "
            "(§9.5)."
        ),
    ),
    Question(
        key="k05", concept="merkezleme-kare-katsayisini-degistirmez", note=_note("9.6", "Tablo 9.6"),
        prompt=(
            "WAGE1 karesel modelinde deneyim 10 yılda merkezlenirse ($X_c = \\text{deneyim} - 10$), $X_c^2$ teriminin "
            "katsayısı ne olur?"
        ),
        answer=MultipleChoice(
            (
                "0,01746 olur: 10 yıldaki eğim kare terimin katsayısına geçer",
                "Sıfır olur: merkezleme eğriliği modelden tamamen kaldırır",
                "−0,0592 olur: kare terim merkezlemeyle 100 kat ölçeklenir",
                "−0,000592 olarak kalır: eğrilik referans noktasına bağlı değildir",
            ),
            correct=3,
        ),
        explanation=(
            "$X = X_c + 10$ yazılırsa $\\beta_1X + \\beta_2X^2 = (\\beta_1 + 20\\beta_2)X_c + \\beta_2X_c^2 + "
            "(10\\beta_1 + 100\\beta_2)$ olur. Kare terimin katsayısı değişmez; merkezleme yalnız doğrusal terimin "
            "katsayısını (10 yıldaki eğime, 0,01746'ya) ve sabiti değiştirir. Eğrinin bükülme derecesi hangi noktadan "
            "ölçüldüğüne bağlı değildir; R², HKT ve tahmin edilen değerler de aynı kalır (Tablo 9.6) (§9.6)."
        ),
    ),
    Question(
        key="k06", concept="yuvarlanmis-ciktida-sifir-standart-hata", note=_note("9.8", "Kod 9.2", "Tablo 9.8"),
        prompt=(
            "Kod 9.2'de `expersq` satırında `coef` sütununda `-0.0006`, `std err` sütununda `0.000`, `t` sütununda "
            "`-5.189` görünüyor. Bu satır nasıl okunur?"
        ),
        answer=MultipleChoice(
            (
                "Standart hata sıfırdır; katsayı hiçbir belirsizlik olmadan bilinir",
                "Değerler yuvarlanmıştır; t yuvarlanmamış değerlerle (≈ −0,000592/0,000114) hesaplanır",
                "Standart hata sıfır olduğu için t tanımsızdır; çıktıdaki −5,189 hatalıdır",
                "Kare terim çok küçük olduğu için yazılım onu modelden düşürmüş, satırı yalnız göstermiştir",
            ),
            correct=1,
        ),
        explanation=(
            "Çıktı sütunları sabit sayıda ondalıkla yazılır: 0,000114 üç basamağa yuvarlanınca `0.000` görünür. t "
            "istatistiği yuvarlanmamış değerlerle hesaplanır; altı basamakla −0,000592/0,000114 ≈ −5,19. Tablo 9.8 aynı terimi "
            "−0,000592 (0,000114) biçiminde verir. Katsayı ve standart hata, X² büyük sayılar aldığı için bu "
            "ölçektedir; t için önemli olan ikisinin oranıdır (§9.8)."
        ),
    ),
    Question(
        key="k07", concept="yanlis-bicim-sifir-kosullu-ortalama", note=_note("9.9"),
        prompt=(
            "Gerçek ilişki karesel ($\\beta_2 \\neq 0$) iken yalnız $X$ ile doğrusal model tahmin edilirse hata terimi "
            "$v = \\beta_2X^2 + u + \\text{doğrusal yaklaşım farkı}$ olur. Bu durum öncelikle hangi varsayımı tehdit "
            "eder?"
        ),
        answer=MultipleChoice(
            (
                "Normallik: hatanın normal ve simetrik bir dağılıma sahip olması",
                "Tam doğrusal bağlantı olmaması: sütunların doğrusal bağımsızlığı",
                "Sıfır koşullu ortalama: hatanın X'e göre ortalaması sıfır",
                "Rassal örnekleme: gözlemlerin anakütleden rassal seçilmesi",
            ),
            correct=2,
        ),
        explanation=(
            "Dışarıda bırakılan eğrilik ($\\beta_2X^2$) X'in fonksiyonudur; bu yüzden hata X ile sistematik ilişki taşır "
            "ve $\\mathbb{E}(v \\mid X) \\neq 0$ olur. Sonuç eksik değişken yanlılığına benzer: doğrusal eğim sabit bir "
            "etkiyi değil, örneklemdeki X dağılımına bağlı bir ortalama eğimi tahmin eder ve katsayı yanlış ekonomik "
            "soruyu cevaplayabilir. Sorun v'nin dağılımının biçimi (normallik) değil, X'e göre koşullu ortalamasıdır; "
            "tam doğrusal bağlantı ve örnekleme süreci de bu hatadan etkilenmez (§9.9)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="iki-degerli-x-ile-kare-tam-baglanti", note=_note("9.1"),
        prompt=(
            "Bir örneklemde $X$ yalnız 2 ve 5 değerlerini alıyorsa $Y = \\beta_0 + \\beta_1X + \\beta_2X^2 + u$ "
            "modelinde tam çoklu doğrusal bağlantı vardır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "İki değer alan X için $X^2$ her zaman sabit ile X'in doğrusal birleşimidir: burada $X^2 = 7X - 10$ (4 = 14 − "
            "10; 25 = 35 − 10). Sabit, X ve X² sütunları tam doğrusal bağımlıdır; iki noktadan sonsuz sayıda parabol "
            "geçtiği için $\\beta_1$ ile $\\beta_2$ ayrı ayrı belirlenemez. X en az üç farklı değer aldığında böyle bir "
            "tam ilişki kurulamaz, çünkü $X^2$ X'in doğrusal olmayan bir fonksiyonudur. WAGE1'de deneyim ile karesi "
            "arasındaki ≈ 0,96'lık korelasyon tam bağlantı değildir; yalnız standart hataları büyütür (§9.1)."
        ),
    ),
    Question(
        key="d02", concept="log-bagimli-degiskende-birim-sabiti-kaydirir", note=_note("9.2", "Kod 9.2"),
        prompt=(
            "Bağımlı değişkeni $\\ln(\\text{ücret})$ olan WAGE1 modelinde ücret dolar yerine sent cinsinden yazılırsa "
            "(100 ile çarpılırsa) eğitim katsayısı 0,0845'ten 8,45'e çıkar."
        ),
        answer=TrueFalse(False),
        explanation=(
            "$\\ln(100 \\cdot \\text{ücret}) = \\ln(100) + \\ln(\\text{ücret})$: bağımlı değişkene her gözlemde aynı "
            "sabit ($\\ln 100 \\approx 4{,}605$) eklenir. Yalnız sabit terim 4,605 artar (Kod 9.2'de 0,2016'dan "
            "yaklaşık 4,807'ye) ve onun t değeri değişir; eğim katsayıları ile t değerleri, bütün standart hatalar ve R² "
            "aynı kalır. Bağımlı değişkeni ücret olan "
            "düzey modelinde ise bütün katsayılar 100 ile çarpılır. Log modelde eğimler birimden bağımsız yüzde yorumu "
            "taşır (§9.2)."
        ),
    ),
    Question(
        key="d03", concept="tam-degisim-ile-marjinal-etki-farki", note=_note("9.4", "Tablo 9.5"),
        prompt=(
            "Karesel modelde $X$'in $x$'ten $x + 1$'e çıkmasıyla oluşan tahmin edilen tam değişim ile $x$ noktasındaki "
            "marjinal etki arasındaki fark, her $x$ düzeyinde $\\widehat\\beta_2$'ye eşittir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Tam değişim $\\widehat\\beta_1 + \\widehat\\beta_2(2x + 1)$, marjinal etki $\\widehat\\beta_1 + "
            "2\\widehat\\beta_2x$'tir; farkları $\\widehat\\beta_2$'dir ve x'e bağlı değildir. $\\widehat\\beta_2 < 0$ "
            "ise bir birimlik tam değişim her noktada marjinal etkiden biraz küçüktür. WAGE1 log ücret modelinde fark "
            "−0,000592 log birimdir (yaklaşık −0,06 yüzde puan); Tablo 9.5'in üçüncü sütunu ayrıca tam yüzde dönüşümü "
            "uyguladığı için iki sütun arasındaki fark bundan biraz ayrılır (§9.4)."
        ),
    ),
    Question(
        key="d04", concept="gereksiz-terim-orneklem-disi-tahmin", note=_note("9.7"),
        prompt=(
            "Bir modele çok sayıda gereksiz terim eklenerek elde edilen $R^2$ artışı, modelin örneklem dışındaki "
            "gözlemleri de daha iyi tahmin edeceğini gösterir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "$R^2$ ek terimle azalmaz; gereksiz terimler örneklemdeki tesadüfi dalgalanmaları izleyerek uyumu yapay "
            "olarak artırabilir. Bu uyum yeni gözlemlere taşınmaz; aşırı esnek bir model örneklem dışında daha kötü "
            "tahmin edebilir. Düzeltilmiş $R^2$, ortak F testi, teori ve basitlik bu yüzden birlikte kullanılır "
            "(“Sık yapılan hata” kutusu) (§9.7)."
        ),
    ),
    Question(
        key="d05", concept="kare-terim-yildizi-marjinal-etki-degil", note=_note("9.8", "Tablo 9.8"),
        prompt=(
            "Tablo 9.8'in (2) sütununda Deneyim² katsayısının üç yıldızlı olması, deneyimin marjinal etkisinin her "
            "deneyim düzeyinde yüzde 1'de anlamlı olduğunu gösterir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Yıldız, $H_0: \\beta_{\\text{deneyim}^2} = 0$ hipotezini, yani eğriliğin varlığını sınar. Marjinal etki "
            "$\\beta_1 + 2\\beta_2 \\cdot \\text{deneyim}$ başka bir büyüklüktür ve anlamlılığı deneyim düzeyine göre "
            "değişir: dönüm noktası (≈ 24,76 yıl) çevresinde tahmin edilen marjinal etki sıfıra yakındır ve anlamlı "
            "değildir. Belirli bir düzeydeki marjinal etki, iki katsayının birleşimi olarak ayrıca sınanır (§9.8)."
        ),
    ),
    Question(
        key="d06", concept="hiyerarsi-ilkesi-sifirda-sifir-egim", note=_note("9.10"),
        prompt=(
            "Bir modele $X^2$ eklenirken $X$'in düzey terimi çıkarılırsa ($Y = \\beta_0 + \\beta_2X^2 + u$), tahmin "
            "edilen eğrinin eğimi $X = 0$'da sıfıra zorlanır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$Y = \\beta_0 + \\beta_2X^2$ modelinde eğim $2\\beta_2X$'tir; X = 0'da her zaman sıfırdır ve dönüm noktası "
            "X = 0'a sabitlenir. Veri bu kısıtı desteklemese de model onu dayatır. Hiyerarşi ilkesi bu yüzden X² varken "
            "X'i de modelde tutar; X'in katsayısı dönüm noktasının veriden belirlenmesini sağlar (Sık Yapılan Hatalar, "
            "madde 2) (§9.10)."
        ),
    ),
    Question(
        key="d07", concept="ayni-egim-sayisinda-ayni-siralama", note=_note("9.7", "Tablo 9.7"),
        prompt=(
            "Tablo 9.7'deki $M_2$ (deneyim karesel) ve $M_3$ (kıdem karesel) aynı sayıda eğim içerdiği için $R^2$'ye "
            "göre sıralamaları ile düzeltilmiş $R^2$'ye göre sıralamaları aynıdır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Düzeltilmiş $R^2 = 1 - (1 - R^2)(n - 1)/(n - k - 1)$'dir; n ve k aynıysa $R^2$'nin artan bir "
            "fonksiyonudur. $M_2$ ve $M_3$'te k = 4: $R^2$ (0,3595 > 0,3341) ile düzeltilmiş $R^2$ (0,3545 > 0,3290) "
            "aynı sırayı verir. Ceza sıralamayı yalnız eğim sayısı farklı modellerde değiştirebilir. İki model iç içe "
            "olmadığı için aralarında ortak F testi de uygulanamaz (§9.7)."
        ),
    ),
    # --- Boşluk doldurma ----------------------------------------------------------
    Question(
        key="b01", concept="karesel-modelde-cok-yillik-fark", note=_note("9.5", "Kod 9.2"),
        prompt=(
            "WAGE1 karesel modelinin katsayılarıyla (deneyim 0,0293; deneyim² −0,000592) eğitim ve kıdem aynıyken "
            "deneyimi 15 yıl olan bir çalışanla 5 yıl olan bir çalışan arasındaki tahmin edilen log ücret farkı **(1)** "
            "olur (virgülden sonra dört basamak). Bu fark, tam yüzde olarak yüzde **(2)** daha yüksek ücrete karşılık "
            "gelir (virgülden sonra iki basamak)."
        ),
        answer=FillBlanks((NumberBlank(0.1746, 0.0002, "0,1746"), NumberBlank(19.08, 0.02, "19,08"))),
        explanation=(
            "Log ücret farkı $0{,}0293(15 - 5) - 0{,}000592(15^2 - 5^2) = 0{,}2930 - 0{,}1184 = 0{,}1746$. Kare terimi "
            "yok sayıp yalnız deneyim katsayısını kullanmak farkı 0,2930 gösterirdi; kare terim bunun yaklaşık %40'ını "
            "geri alır. Tam yüzde "
            "karşılığı $100(e^{0{,}1746} - 1) \\approx 19{,}08$: deneyimi 15 yıl olan çalışanın tahmin edilen ücreti "
            "yaklaşık %19 daha yüksektir. Doğrusal modelde ($M_1$, deneyim katsayısı 0,0041) aynı fark yalnız 0,041 "
            "olurdu (§9.5)."
        ),
    ),
    Question(
        key="b02", concept="makale-tablosundan-kidem-etkisi", note=_note("9.8", "Tablo 9.8"),
        prompt=(
            "Tablo 9.8'in (2) sütunundaki katsayılarla (kıdem 0,0371; kıdem² −0,000616) bir ek kıdem yılının yaklaşık "
            "yüzde ilişkisi kıdem 5 yıl iken yüzde **(1)**, kıdem 25 yıl iken yüzde **(2)** olur. (Virgülden sonra iki "
            "basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(3.09, 0.011, "3,09"), NumberBlank(0.63, 0.011, "0,63"))),
        explanation=(
            "Kıdem 5 yıl iken marjinal etki $0{,}0371 + 2(-0{,}000616)(5) = 0{,}03094$ (yaklaşık %3,09), 25 yıl iken "
            "$0{,}0371 - 0{,}0308 = 0{,}0063$ (yaklaşık %0,63); yuvarlanmamış katsayılarla %3,10 ve %0,63. Kıdemin tek başına katsayısı (0,0371) yalnız "
            "kıdem = 0'daki eğimdir; kare terim her ek yılda eğimi $2 \\times 0{,}000616$ kadar azaltır. Makale "
            "tablosunda düzey ve kare katsayıları birlikte okunur (§9.8)."
        ),
    ),
    Question(
        key="b03", concept="tek-terim-eklemede-f-ve-t", note=_note("9.7", "Tablo 9.7", "Kod 9.2"),
        prompt=(
            "Tablo 9.7'de $M_2$'nin HKT'si 95,011, $M_4$'ünki 93,911'dir; n = 526 ve $M_4$'te 5 eğim vardır. $M_2$ ile "
            "$M_4$'ü karşılaştıran F istatistiği HKT biçimiyle **(1)** olur. Bu değerin karekökü: **(2)**. "
            "(Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(6.09, 0.01, "6,09"), NumberBlank(2.47, 0.01, "2,47"))),
        explanation=(
            "$M_4$, $M_2$'ye yalnız kıdem² terimini ekler: tek kısıt (q = 1). $F = (95{,}011 - 93{,}911)/(93{,}911/520) "
            "\\approx 6{,}09$ ve $\\sqrt{6{,}09} \\approx 2{,}47$: Kod 9.2'deki `tenursq` satırının t değerinin mutlak "
            "değeri (2,468). Tek kısıtta F = t² olduğundan Tablo 9.7'deki HKT karşılaştırması ile çıktıdaki t testi aynı "
            "kararı verir; kıdem² yüzde 5'te anlamlıdır (p = 0,014) (§9.7)."
        ),
    ),
    Question(
        key="b04", concept="merkezleme-x-ile-karesinin-korelasyonu", note=_note("9.6"),
        prompt=(
            "Bir örneklemde $X$ yalnız 1, 2, 3, 4 ve 5 değerlerini birer kez alıyor. $X$ ile $X^2$ arasındaki örneklem "
            "korelasyonu yaklaşık **(1)**; $X$ ortalamasında merkezlenince ($X_c = X - 3$) $X_c$ ile $X_c^2$ arasındaki "
            "korelasyon **(2)** olur. (Virgülden sonra iki basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(0.98, 0.006, "0,98"), NumberBlank(0.0, 0.005, "0,00"))),
        explanation=(
            "$X^2$ = 1, 4, 9, 16, 25 değerleri X ile neredeyse doğrusal artar: korelasyon ≈ 0,981. Merkezlenince "
            "$X_c$ = −2, −1, 0, 1, 2 ve $X_c^2$ = 4, 1, 0, 1, 4 olur; kare simetrik olduğu için $\\sum X_c X_c^2 = -8 - "
            "1 + 0 + 1 + 8 = 0$, kovaryans ve korelasyon sıfırdır. Merkezleme X ile X² arasındaki örneklem "
            "korelasyonunu azaltabilir; tahmin edilen değerleri, R²'yi ve eğriyi değiştirmez, içselliği de gidermez "
            "(§9.6)."
        ),
    ),
    Question(
        key="b05", concept="iki-katina-cikma-bilesik-etki", note=_note("9.3", "Kod 9.2"),
        prompt=(
            "Kod 9.2'deki eğitim katsayısı 0,0845'tir. Diğer değişkenler aynıyken tahmin edilen ücretleri iki kat farklı "
            "olan iki çalışan arasındaki eğitim farkı tam formülle **(1)** yıldır. Yaklaşık formül ($\\%\\Delta Y \\approx "
            "100\\,\\beta\\,\\Delta X$) %100'lük artış için kullanılırsa **(2)** yıl bulunur. (Virgülden sonra iki "
            "basamak yazın.)"
        ),
        answer=FillBlanks((NumberBlank(8.20, 0.01, "8,20"), NumberBlank(11.83, 0.01, "11,83"))),
        explanation=(
            "Tam formülle $e^{0{,}0845\\,\\Delta X} = 2$ olmalıdır: $\\Delta X = \\ln 2/0{,}0845 \\approx 8{,}20$ yıl. "
            "Yaklaşık formül (100/8,45 ≈ 11,83 yıl) bileşik değişimi yok sayar: her yılın artışı bir önceki yılın daha "
            "yüksek düzeyine uygulanır. Büyük değişimlerde yaklaşık yorum bu yüzden yanıltır; dört yıllık değişim için "
            "Tablo 9.4'ün son satırındaki %32 ile %37,71 farkı da buradan gelir (§9.3)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="e01", concept="tepe-noktasinda-tahmin-edilen-deger", note=_note("9.4"),
        prompt=(
            "Karesel modelde $\\widehat Y = b_0 + b_1X + b_2X^2$ ve $b_2 < 0$'dır. Eğrinin tepe noktasındaki tahmin "
            "edilen $Y$ değerini $b_0$, $b_1$ ve $b_2$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widehat Y(X^{*})",
            symbols=(
                _estimate(0, 0, 5, "sabit"),
                _estimate(1, 0.5, 3, "X'in katsayısı"),
                _estimate(2, -0.5, -0.05, "X²'nin katsayısı (negatif)"),
            ),
            answer="b0 - b1^2/(4*b2)",
            shown="b_0 - \\frac{b_1^2}{4\\,b_2}",
        ),
        explanation=(
            "Tepe noktası $X^* = -b_1/(2b_2)$'dir. Yerine konursa $\\widehat Y(X^*) = b_0 - b_1^2/(2b_2) + "
            "b_1^2/(4b_2) = b_0 - b_1^2/(4b_2)$; $b_2 < 0$ olduğundan tepe değeri $b_0$'dan büyüktür. WAGE1'de "
            "deneyimin log ücrete katkısı ($0{,}0293X - 0{,}000592X^2$) en çok $0{,}0293^2/(4 \\times 0{,}000592) "
            "\\approx 0{,}36$ log birime, yaklaşık 24,76 yılda ulaşır (§9.4)."
        ),
    ),
    Question(
        key="e02", concept="bir-birimlik-tam-degisimin-sifirlandigi-nokta", note=_note("9.5", "Tablo 9.5"),
        prompt=(
            "Karesel modelde $X$'in $x$'ten $x + 1$'e çıkmasıyla oluşan tam değişim $b_1 + b_2(2x + 1)$'dir. Bu bir "
            "birimlik tam değişimin sıfır olduğu $x$ değerini $b_1$ ve $b_2$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="x_0",
            symbols=(
                _estimate(1, 0.5, 3, "X'in katsayısı"),
                _estimate(2, -0.5, -0.05, "X²'nin katsayısı"),
            ),
            answer="-b1/(2*b2) - 1/2",
            shown="-\\frac{b_1}{2\\,b_2} - \\frac{1}{2}",
        ),
        explanation=(
            "$b_1 + b_2(2x + 1) = 0$ eşitliğinden $x = -b_1/(2b_2) - 1/2$: dönüm noktasından yarım birim önce. Bir "
            "birimlik adım, x ile x + 1 arasındaki eğrinin ortalama eğimidir; adımın orta noktası dönüm noktasına "
            "geldiğinde değişim sıfırlanır. WAGE1'de 24,76 − 0,5 ≈ 24,26: 24'ten 25'e geçişte değişim hâlâ çok az "
            "pozitiftir, 25'ten 26'ya negatiftir; Tablo 9.5'te 25 yıldaki bir yıllık tam değişim %−0,09'dur (§9.5)."
        ),
    ),
    Question(
        key="e03", concept="log-log-buyuk-degisimde-tam-yuzde", note=_note("9.3"),
        prompt=(
            "Log–log modelde $\\ln(Y) = \\beta_0 + \\beta_1\\ln(X) + u$ ve eğim tahmini $b$'dir. $X$ yüzde $p$ "
            "arttığında ($p$ büyük olabilir) $Y$'deki tam yüzde değişimi $b$ ve $p$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\%\\Delta Y",
            symbols=(
                Symbol("b", "b", "log–log eğim tahmini (esneklik)", 0.2, 1.5,
                       aliases=(*beta_hat_aliases(1), *beta_aliases(1))),
                Symbol("p", "p", "X'teki yüzde artış", 5, 80),
            ),
            answer="100*((1 + p/100)^b - 1)",
            shown="100\\left[\\left(1 + \\frac{p}{100}\\right)^{b} - 1\\right]",
        ),
        explanation=(
            "X, $(1 + p/100)$ katına çıkınca ln(X) $\\ln(1 + p/100)$ kadar, ln(Y) de $b\\ln(1 + p/100)$ kadar artar. "
            "Y'nin yeni düzeyi eskisinin $e^{b\\ln(1 + p/100)} = (1 + p/100)^b$ katıdır. Küçük p için sonuç yaklaşık "
            "$b\\,p$'dir (esneklik yorumu); b = 0,5 ve p = 50 için tam değişim %22,47, yaklaşık değer %25'tir. Büyük "
            "yüzde değişimlerde esneklik yorumu da tam formülle düzeltilir (§9.3)."
        ),
    ),
    Question(
        key="e04", concept="merkezlenmis-modelin-sabiti", note=_note("9.6", "Tablo 9.6"),
        prompt=(
            "Karesel model $\\widehat Y = b_0 + b_1X + b_2X^2$, $X_c = X - c$ ile $\\widehat Y = a_0 + a_1X_c + "
            "a_2X_c^2$ biçiminde yeniden yazılıyor. Merkezlenmiş modelin sabiti $a_0$'ı $b_0$, $b_1$, $b_2$ ve $c$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs="a_0",
            symbols=(
                _estimate(0, -2, 5, "ham modelin sabiti"),
                _estimate(1, 0.5, 3, "X'in katsayısı"),
                _estimate(2, -0.5, 0.5, "X²'nin katsayısı"),
                Symbol("c", "c", "merkez noktası", 1, 20),
            ),
            answer="b0 + b1*c + b2*c^2",
            shown="b_0 + b_1\\,c + b_2\\,c^2",
        ),
        explanation=(
            "$X_c = 0$, yani X = c noktasında merkezlenmiş model $a_0$'ı verir; aynı noktada ham model $b_0 + b_1c + "
            "b_2c^2$'yi verir. İki model aynı tahmin edilen değerleri ürettiği için $a_0$ bu değere eşittir: "
            "merkezlenmiş sabit, X = c'deki (diğer değişkenler sıfırken) tahmin edilen Y'dir. WAGE1'de deneyim 10 "
            "yılda merkezlenince sabit $10 \\times 0{,}0293 - 100 \\times 0{,}000592 \\approx 0{,}234$ artar (§9.6)."
        ),
    ),
    Question(
        key="e05", concept="dogrusal-egim-ortalamadaki-marjinal-etki", note=_note("9.9"),
        prompt=(
            "Gerçek model $Y = \\beta_0 + \\beta_1X + \\beta_2X^2 + u$ iken araştırmacı $X^2$'yi dışarıda bırakıyor. "
            "$X$, $\\mu$ ortalaması çevresinde simetrik dağılıyorsa $X^2$'nin $X$'e yardımcı regresyonundaki eğim "
            "$2\\mu$'dür. Konu 6'daki eksik değişken formülüyle kısa model eğiminin merkezini $\\beta_1$, $\\beta_2$ ve "
            "$\\mu$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="\\widetilde\\beta_1",
            symbols=(
                Symbol("b1", "\\beta_1", "X'in gerçek katsayısı", 0.5, 3, aliases=beta_aliases(1)),
                Symbol("b2", "\\beta_2", "X²'nin gerçek katsayısı", -0.5, 0.5, aliases=beta_aliases(2)),
                Symbol("mu", "\\mu", "X'in ortalaması", 1, 20, aliases=("\\mu", "μ")),
            ),
            answer="b1 + 2*b2*mu",
            shown="\\beta_1 + 2\\,\\beta_2\\,\\mu",
        ),
        explanation=(
            "Eksik değişken formülünde kısa eğimin merkezi $\\beta_1 + \\beta_2\\delta_1$'dir; burada dışlanan "
            "değişken $X^2$ ve $\\delta_1 = 2\\mu$'dür. Sonuç $\\beta_1 + 2\\beta_2\\mu$, marjinal etkinin X'in "
            "ortalamasındaki değeridir: doğrusal eğim sabit bir etki değil, X dağılımına bağlı bir ortalama eğimdir ve "
            "X'in aralığı değişirse değişir. Konu 9 Sezgi Deney 3'ün DGP'sinde ($\\beta_1 = 2$, $\\beta_2 = -\\gamma$, "
            "X ~ U(0, 10)) bu değer $2 - 10\\gamma$'dır: γ = 0,15 için 0,5 (§9.9)."
        ),
    ),
)


KONU09_QUIZ = QuestionSet(
    topic_key="konu09",
    title="Konu 9: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Notlardaki egzersiz ve mini "
        "quiz maddelerini tekrar etmez, yeni bağlamlarla onları tamamlar."
    ),
    sections=tuple(f"9.{number}" for number in range(1, 11)),  # §9.11 bölüm özetidir
)
