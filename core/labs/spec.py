"""Ders notu uygulamalarının dilden bağımsız tanım şeması.

Bir uygulama tanımı dört şeyi birlikte besler:

1. Uygulama sekmesindeki adım adım anlatım,
2. Uygulamanın kendi hesabı (``core.labs.runner``),
3. Python ve R kodu (``core.codegen``),
4. Notlardaki sayılarla karşılaştıran testler.

Bir sayı veya işlem yalnız burada değişir; diğer dört çıktı kendiliğinden izler.

Adlandırma: ``frame`` gözlem düzeyindeki bir veri çerçevesidir (her satır bir gözlem);
``table`` ise bir sonuç tablosudur (satır adları kategoriler, sütunlar sayılar); ``model`` tahmin
edilmiş bir regresyondur.

Etkileşimli spesifikasyon: bir adım denetimler (``Choice``, ``MultiChoice``, ``NumberChoice``) ve bu
seçimlerden işlem listesi üreten bir ``build`` fonksiyonu taşıyabilir. Varsayılan seçimler notlardaki
spesifikasyondur; ``LabStep.operations`` her zaman varsayılan seçimlerin işlem listesidir. Başka seçimler
``LabSpec.resolve`` ile yeni bir tanım üretir; notlardan farklı adımların (ve onların sonucunu kullanan
sonraki adımların) notlarla karşılaştırması kaldırılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Callable, Mapping, Union

from core.labs.expr import Expr


class ReproClass(str, Enum):
    """İki dilde (Python ve R) aynı sayının hangi anlamda beklenebileceği."""

    EXACT = "birebir"
    CONVENTION = "ayar"
    DISTRIBUTIONAL = "dagilim"


REPRO_DESCRIPTIONS = {
    ReproClass.EXACT: (
        "Birebir aynı",
        "Deterministik hesap. Python ve R aynı sayıyı ondalık düzeyinde verir.",
    ),
    ReproClass.CONVENTION: (
        "Ayar sabitlenince aynı",
        "Paketlerin varsayılan ayarları farklı. Kodda ayar açıkça sabitlendiği için sonuç aynıdır; "
        "varsayılan ayarla çalıştırırsanız farklı sayı görebilirsiniz.",
    ),
    ReproClass.DISTRIBUTIONAL: (
        "Yalnız dağılımda aynı",
        "Rastgele çekiliş içerir. Python ve R farklı rastgele sayı üreteci kullandığı için aynı tohum aynı "
        "çekilişi vermez; sonuçlar rastgele çekiliş farkı kadar değişir.",
    ),
}

STATISTICS = (
    "count", "sum", "mean", "median", "mode", "mode_freq", "prod", "min", "max", "var", "std", "nunique", "value",
)
"""``value``: koşulu sağlayan tek gözlemin değeri (ör. A mağazasının memnuniyeti). ``mode``: tek mod
(birden fazla değer en yüksek frekansa sahipse hata); ``mode_freq``: modun frekansı; ``prod``: çarpım;
``var`` ve ``std``: örneklem varyansı s² ve standart sapması s (payda n − 1); ``nunique``: farklı değer sayısı."""
PAIR_STATISTICS = ("cov", "corr")
"""İki değişkenli istatistikler: örneklem kovaryansı s_xy (payda n − 1) ve Pearson korelasyonu r."""
DISTRIBUTIONS = ("normal", "uniform", "beta", "gamma", "exponential")
"""Sürekli çekilişler: normal (μ, σ), tek-düze (a, b), beta (a, b), gamma (biçim, ölçek) ve üstel (μ, σ = μ)."""
COUNT_DISTRIBUTIONS = {"binomial": 2, "poisson": 1, "hypergeometric": 3}
"""Sayım çekilişlerinin dağılımları ve parametre sayıları: binom (n, p), Poisson (λ), hipergeometrik (N, r, n)."""
DENSITIES = ("normal", "uniform", "exponential", "gamma")
"""Yoğunluk grafiğinin dağılımları ve iki parametresi: normal (μ, σ), tek-düze (a, b), üstel (μ, σ = μ; notlardaki
ortalama süre parametrelemesi) ve gamma (biçim k, oran r; üstel anakütleden n gözlemin ortalaması X̄ için k = r = n)."""
Parameter = Union[float, str]
"""Grafik parametresi: sayı ya da önceden hesaplanmış bir skalerin adı (kodda aynı adlı değişken)."""
BOX_ROWS = (
    "en_kucuk", "q1", "medyan", "q3", "en_buyuk", "iqr", "alt_sinir", "ust_sinir", "alt_biyik", "ust_biyik",
    "aykiri_sayisi",
)
"""Kutu grafiği özetinin satırları: beş sayı özeti (çeyrekler ders kuralıyla), IQR, Q₁ − 1,5·IQR ve
Q₃ + 1,5·IQR sınırları, sınırların içindeki en uç gözlemler (bıyık uçları) ve aykırı değer sayısı."""
PERCENT_KINDS = (None, "satir", "sutun")
CLASS_COLUMNS = (
    "orta_nokta", "frekans", "goreli", "yuzde", "kumulatif_frekans", "kumulatif_goreli", "kumulatif_yuzde",
)
"""Sınıf tablosunun seçilebilir sütunları; ``alt`` ve ``ust`` her zaman vardır."""
TOTALLED_CLASS_COLUMNS = ("frekans", "goreli", "yuzde")
PERCENTILE_METHODS = ("ders", "yazilim")
"""``ders``: L_p = (p/100)(n + 1) ve doğrusal ara değer (Hyndman–Fan tip 6); ``yazilim``: numpy ve R'nin
varsayılanı (tip 7), 1 + (n − 1)p/100 konumu."""
TOTAL = "Toplam"
"""Frekans ve çapraz tablolarda toplam satırının/sütununun adı (iki dilde aynı)."""


# --- Veri ------------------------------------------------------------------

@dataclass(frozen=True)
class InlineData:
    """Ders notlarında basılı küçük veri seti; kodda satır içinde yazılır.

    ``rows`` her gözlem için bir demettir. ``layout`` verilirse (tek sütunlu veride)
    kod ve ekran değerleri notlardaki tablo gibi satır başına ``layout`` değer dizer.
    """

    frame: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    comment: str
    layout: int | None = None


@dataclass(frozen=True)
class FromCounts:
    """Sayım tablosundan gözlem düzeyinde veri: sayım tablosunun her satırı ``sayı`` kez tekrarlanır.

    ``rows`` her satır için (kategoriler..., sayı) demetidir.
    """

    frame: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    comment: str


@dataclass(frozen=True)
class VariableTypes:
    """Değişkenlerin notlardaki istatistiksel türü ve yazılımın onları saklama biçimi.

    ``rows``: (değişken, istatistiksel tür, ayrıntı). Yazılımın saklama türü veriden okunur.
    """

    frame: str
    rows: tuple[tuple[str, str, str], ...]
    result: str


@dataclass(frozen=True)
class Outcomes:
    """Çok aşamalı bir deneyin bütün sonuçları: aşamalardaki seçeneklerin bütün bileşimleri.

    ``stages``: (sütun adı, seçenekler). Satırlar ağaç diyagramındaki sırayla gelir: ilk aşama en yavaş,
    son aşama en hızlı değişir. Satır sayısı çarpım kuralıyla n₁·n₂·…·n_k'dir.
    """

    frame: str
    stages: tuple[tuple[str, tuple[object, ...]], ...]
    comment: str


@dataclass(frozen=True)
class Selections:
    """``items`` içinden ``k`` öğenin bütün seçimleri, sözlük sırasıyla.

    ``ordered=False``: sıra önemsiz, kombinasyonlar (C(N, k) satır); ``ordered=True``: sıra veya görev önemli,
    permütasyonlar (P(N, k) satır). ``columns`` k sütunun adlarıdır (ör. başkan, raportör).
    """

    frame: str
    items: tuple[str, ...]
    k: int
    ordered: bool
    columns: tuple[str, ...]
    comment: str


@dataclass(frozen=True)
class Event:
    """Olay: örnek uzayın bir alt kümesi. ``column`` değeri ``values`` içinde olan satırlarda 1, diğerlerinde 0.

    Satırlar örnek noktalar (ya da simülasyonda tek tek denemeler) olduğunda yeni sütun olayın gösterge
    değişkenidir; olayın olasılığı bu göstergeyle seçilen örnek noktaların olasılıkları toplanarak bulunur.
    """

    frame: str
    name: str
    column: str
    values: tuple[object, ...]
    comment: str


@dataclass(frozen=True)
class ShowFrame:
    """Bir veri çerçevesinin seçili sütunlarını gösterir (ör. örnek uzay ve olayların gösterge sütunları).

    ``head``: yalnız ilk ``head`` satır (pandas ``head``). ``rows``: yalnız bu gözlemler (1'den başlayan sıra
    numarası; notlardaki "1, 2, 3, …, 526" gibi). ``where``: yalnız koşulu sağlayan satırlar (ör. bir kişinin
    panel gözlemleri). Üçü birlikte kullanılmaz. ``decimals``: kesirli sütunlar bu basamakla gösterilir (tam sayı
    değerli sütunlar tam sayı kalır); verilmezse basamak sütunun değerlerinden seçilir.
    """

    frame: str
    columns: tuple[str, ...]
    comment: str
    head: int = 0
    rows: tuple[int, ...] = ()
    where: tuple[str, object] | None = None
    decimals: int | None = None

    def __post_init__(self) -> None:
        if sum((bool(self.head), bool(self.rows), self.where is not None)) > 1:
            raise ValueError("ShowFrame: head, rows ve where birlikte kullanılamaz.")


@dataclass(frozen=True)
class LoadWooldridge:
    """Wooldridge (2020) veri seti: Python ``wooldridge`` paketinden (``wd.data``), R'de ``wooldridge``
    paketinden (``data(..., package = "wooldridge")``). İki paket aynı veriyi verir (testle denetlenir).

    ``columns`` verilirse yalnız bu sütunlar tutulur (sıra korunur). Çerçevenin adı veri setinin adıdır.
    """

    dataset: str
    comment: str
    columns: tuple[str, ...] = ()

    @property
    def frame(self) -> str:
        return self.dataset


@dataclass(frozen=True)
class SortRows:
    """Satırları bir değişkene göre küçükten büyüğe dizer (eşit değerlerde önceki sıra korunur) ve satır
    numarasını baştan verir. Zaman serisinde bu işlem zaman sırasını bozar."""

    frame: str
    by: str
    comment: str


@dataclass(frozen=True)
class MapCodes:
    """Kategori etiketlerini sayı kodlarına çevirir (ör. Kaldı = 0, Geçti = 1)."""

    frame: str
    source: str
    name: str
    mapping: tuple[tuple[str, float], ...]
    comment: str


@dataclass(frozen=True)
class Groups:
    """Gözlemleri sırayla gruplara ayırır: ilk ``sizes[0]`` gözlem ``labels[0]`` grubunda, ..."""

    frame: str
    name: str
    labels: tuple[str, ...]
    sizes: tuple[int, ...]
    comment: str


@dataclass(frozen=True)
class Support:
    """Kesikli bir rassal değişkenin olası değerleri: ``lower``, ``lower`` + 1, …, ``upper`` (tam sayılar).

    Her satır bir olası değerdir; olasılıklar ardından ``Derive`` ile (ör. ``E.dbinom``) eklenir.
    """

    frame: str
    name: str
    lower: int
    upper: int
    comment: str


@dataclass(frozen=True)
class RowSum:
    """Satır toplamı: ``columns`` sütunlarının her satırdaki toplamı (ör. bir dizideki Bernoulli başarılarının
    sayısı X = Y₁ + ⋯ + Yₙ)."""

    frame: str
    name: str
    columns: tuple[str, ...]
    comment: str


@dataclass(frozen=True)
class Rectangles:
    """Eğri altındaki alanı dikdörtgenlerle hesaplamak için [``lower``, ``upper``] aralığının bölünmesi.

    Aralık genişliği ``width`` olan k = (upper − lower)/width dikdörtgene ayrılır; ``name`` sütunu dikdörtgenlerin
    orta noktalarıdır: lower + width·(i − 0,5), i = 1, …, k. Dikdörtgenin alanı yükseklik (yoğunluk) × genişliktir.
    """

    frame: str
    name: str
    lower: float
    upper: float
    width: float
    comment: str

    @property
    def count(self) -> int:
        k = round((self.upper - self.lower) / self.width)
        if k < 1 or abs(k * self.width - (self.upper - self.lower)) > 1e-9:
            raise ValueError("Aralık genişliği dikdörtgen genişliğinin tam katı olmalıdır.")
        return k


@dataclass(frozen=True)
class Derive:
    """İfadeden yeni bir sayısal değişken türetir."""

    frame: str
    name: str
    expr: Expr
    comment: str


# --- Simülasyon ------------------------------------------------------------

@dataclass(frozen=True)
class NewSample:
    """Simülasyon için ``nobs`` satırlık boş örneklem.

    ``seed`` verilirse rastgele sayı üreteci bu tohumla (yeniden) başlatılır. ``seed=None`` yalnız Monte
    Carlo döngüsü içinde kullanılır: üreteç döngüden önce bir kez tohumlanır, her tekrar yeni çekiliş yapar.
    Deneyde tek bir üreteç vardır; bütün çekilişler işlem sırasıyla ondan yapılır.
    """

    frame: str
    nobs: int
    seed: int | None


@dataclass(frozen=True)
class Draw:
    """Sayısal rastgele değişken: normal(ortalama, std. sapma), uniform(alt, üst), beta(a, b) veya
    gamma(biçim, ölçek)."""

    frame: str
    name: str
    distribution: str
    first: float
    second: float
    comment: str


@dataclass(frozen=True)
class DrawCount:
    """Sayım rassal değişkeni (``COUNT_DISTRIBUTIONS``): binom (n, p) — n bağımsız Bernoulli denemesindeki başarı
    sayısı; Poisson (λ) — bir aralıktaki olay sayısı; hipergeometrik (N, r, n) — r başarı içeren N birimden yerine
    koymadan seçilen n birimdeki başarı sayısı."""

    frame: str
    name: str
    distribution: str
    parameters: tuple[float, ...]
    comment: str


@dataclass(frozen=True)
class DrawCategory:
    """Kategorik rastgele değişken.

    Her gözlem için u ~ Tek-düze(0, 1) çekilir; kategori, birikimli olasılığı u'yu ilk aşan kategoridir
    (birikimli olasılıkların sonuncusu 1 kabul edilir). ``by`` verilirse olasılıklar o değişkenlerin
    kategorilerine göre değişir: ``probabilities`` (koşul etiketleri, olasılıklar) çiftleridir;
    ``by`` boşsa tek çift vardır ve etiketler boş demettir.
    """

    frame: str
    name: str
    categories: tuple[str, ...]
    probabilities: tuple[tuple[tuple[str, ...], tuple[float, ...]], ...]
    comment: str
    by: tuple[str, ...] = ()


@dataclass(frozen=True)
class DrawDiscrete:
    """Kesikli rassal değişken: ``values`` değerlerini ``probabilities`` olasılıklarıyla alır.

    Ters dağılım fonksiyonu yöntemi: her gözlem için u ~ Tek-düze(0, 1) çekilir; X, birikimli olasılığı F(x) u'yu
    ilk aşan değerdir (birikimli olasılıkların sonuncusu 1 kabul edilir). ``DrawCategory`` ile aynı kural; sonuç
    sayıdır, kategori etiketi değildir.
    """

    frame: str
    name: str
    values: tuple[float, ...]
    probabilities: tuple[float, ...]
    comment: str


# --- Sayma ve özet ---------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """Gözlem sayısı ``n`` ve değişken sayısı ``k``; ``exclude`` sütunları (kimlik) sayılmaz."""

    frame: str
    observations: str
    variables: str
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class Count:
    """``column == value`` koşulunu sağlayan gözlem sayısı."""

    frame: str
    name: str
    column: str
    value: object
    comment: str


@dataclass(frozen=True)
class Statistic:
    """Bir değişkenin tek istatistiği, skaler olarak (isteğe bağlı olarak bir alt grupta)."""

    frame: str
    variable: str
    stat: str
    name: str
    comment: str
    where: tuple[str, object] | None = None
    decimals: int = 4


@dataclass(frozen=True)
class PairStatistic:
    """İki sayısal değişkenin örneklem kovaryansı (``cov``: payda n − 1) ya da Pearson korelasyonu (``corr``)."""

    frame: str
    x: str
    y: str
    stat: str
    name: str
    comment: str
    decimals: int = 4


@dataclass(frozen=True)
class Scalar:
    """Skalerlerden (``E.ref``) ve sabitlerden hesaplanan tek sayı.

    ``percent``: değer yüzde biriminde; ekranda yüzde işaretiyle gösterilir.
    """

    name: str
    expr: Expr
    comment: str
    decimals: int = 4
    percent: bool = False


@dataclass(frozen=True)
class ScalarTable:
    """Birkaç skaleri tek tabloda toplar (satırlar etiketler, tek sütun ``deger``)."""

    rows: tuple[tuple[str, Expr], ...]
    result: str
    decimals: int = 2


@dataclass(frozen=True)
class GroupSummary:
    """Bir değişkenin gruplarına göre özet: (sütun adı, değişken, istatistik). Gruplar ``order`` sırasıyla; grup
    değerleri sayı da olabilir (ör. başarı sayısı x = 0, 1, …). ``decimals``: ekranda gösterim basamağı (``count``
    sütunları tam sayı)."""

    frame: str
    by: str
    columns: tuple[tuple[str, str, str], ...]
    result: str
    order: tuple[object, ...]
    decimals: int = 3


@dataclass(frozen=True)
class FrequencyTable:
    """Kategorik değişkenin frekans dağılımı.

    Sütunlar: ``frekans`` (f), ``relative`` ise ``goreli`` (r = f/n) ve ``yuzde`` (p = 100 r).
    Kategoriler ``order`` sırasıyla; ``totals`` ise sonda ``Toplam`` satırı.
    """

    frame: str
    variable: str
    result: str
    order: tuple[str, ...]
    relative: bool = True
    totals: bool = False


@dataclass(frozen=True)
class CrossTab:
    """İki kategorik değişkenin çapraz tablosu.

    ``percent``: None sayılar; ``satir`` satır yüzdeleri (her satır 100'e toplanır); ``sutun`` sütun
    yüzdeleri. ``margins``: sayı tablosunda ``Toplam`` satırı ve sütunu; satır yüzdelerinde ``Toplam``
    sütunu, sütun yüzdelerinde ``Toplam`` satırı. ``where`` verilirse yalnız o alt gruptaki gözlemler.
    ``weights`` verilirse hücreler gözlem sayısı değil o sütunun toplamıdır (ör. ortak olasılık tablosu:
    her örnek noktanın olasılığı kendi hücresine yazılır); yalnız ``percent=None`` ile kullanılır.
    """

    frame: str
    row: str
    column: str
    result: str
    row_order: tuple[str, ...]
    column_order: tuple[str, ...]
    percent: str | None = None
    margins: bool = False
    where: tuple[str, object] | None = None
    decimals: int = 1
    weights: str | None = None


@dataclass(frozen=True)
class JoinColumns:
    """Aynı satır adlarına sahip tabloların sütunlarını tek tabloda yan yana toplar.

    ``columns``: (yeni sütun adı, tablo, sütun). Satırlar ilk tablonun sırasıyladır.
    """

    result: str
    columns: tuple[tuple[str, str, str], ...]
    decimals: int = 3
    percent: bool = False


@dataclass(frozen=True)
class BoxSummary:
    """Kutu grafiği özeti (``BOX_ROWS``): her seri için bir sütun.

    ``series``: (veri çerçevesi, değişken, etiket). Çeyrekler ders kuralıyla (L_p = (p/100)(n + 1)); bıyıklar
    Q₁ − 1,5·IQR ve Q₃ + 1,5·IQR sınırlarının içindeki en küçük ve en büyük gözleme kadar uzanır.
    """

    series: tuple[tuple[str, str, str], ...]
    result: str


@dataclass(frozen=True)
class ClassTable:
    """Nicel bir değişkenin eşit genişlikli sınıflarla frekans dağılımı.

    Sınıflar [a, a + h), [a + h, a + 2h), ...: alt sınır dahil, üst sınır hariç (notlardaki 10 ≤ x < 20
    yazımı). ``lower`` verilirse ``classes`` sınıf oradan başlar; verilmezse ilk sınıf en küçük değeri
    içeren h katından başlar ve sınıf sayısı en büyük değeri kapsayacak kadardır.

    Tabloda ``alt`` ve ``ust`` sütunları her zaman vardır; ``columns`` diğerlerini seçer (``CLASS_COLUMNS``):
    orta nokta m = (alt + üst)/2, frekans f, göreli frekans r = f/n, yüzde p = 100 r, kümülatif frekans
    F, F/n ve kümülatif yüzde. Satır adları "10 ≤ x < 20"; ``row_labels="ust"`` ise "x < 20" (kümülatif
    tablo). ``totals`` sonda ``Toplam`` satırı ekler (frekans, göreli ve yüzde sütunlarının toplamı).
    """

    frame: str
    variable: str
    result: str
    width: float
    columns: tuple[str, ...]
    lower: float | None = None
    classes: int | None = None
    totals: bool = False
    row_labels: str = "sinif"


@dataclass(frozen=True)
class StemLeaf:
    """Gövde–yaprak gösterimi: gövde onlar basamağı, yaprak birler basamağı (negatif olmayan tam sayılar).

    Sonuç tablosunun satırları gövdelerdir (en küçükten en büyüğe, boş gövdeler dahil); sütunlar
    ``yapraklar`` (küçükten büyüğe, boşlukla ayrılmış) ve ``yaprak_sayisi``.
    """

    frame: str
    variable: str
    result: str


@dataclass(frozen=True)
class Percentile:
    """p. yüzdelik ``name`` skalerine yazılır; ``location`` verilirse konum da skaler olur.

    ``method="ders"``: L_p = (p/100)(n + 1); L_p tam sayı değilse komşu iki gözlem arasında doğrusal ara
    değer; L_p ≤ 1 ise en küçük, L_p ≥ n ise en büyük gözlem (Hyndman–Fan tip 6). ``method="yazilim"``:
    numpy ve R'nin varsayılanı (tip 7), konum 1 + (n − 1)p/100.
    """

    frame: str
    variable: str
    p: float
    name: str
    comment: str
    location: str | None = None
    method: str = "ders"
    decimals: int = 2


# --- Grafikler -------------------------------------------------------------

@dataclass(frozen=True)
class BarChart:
    """Tek serili sütun grafiği.

    ``source`` bir sonuç tablosudur (kategoriler satır adları) ya da ``x`` verilirse bir veri
    çerçevesidir (kategoriler ``x`` sütunu). ``Toplam`` satırı çizilmez. ``sort="azalan"``
    sütunları yüksekten düşüğe dizer; ``y_range`` değer ekseninin sınırlarıdır (yanıltıcı
    eksen örneği için). ``percent``: değer etiketleri yüzde işaretiyle.
    """

    source: str
    y: str
    x_label: str
    y_label: str
    title: str
    x: str | None = None
    sort: str | None = None
    horizontal: bool = False
    y_range: tuple[float, float] | None = None
    percent: bool = False
    decimals: int = 0


@dataclass(frozen=True)
class GroupedBarChart:
    """Çapraz tablodan çok serili sütun grafiği: yan yana veya yığılmış (yüzde 100).

    ``series="satir"``: her tablo satırı bir seri, sütunlar yatay eksende; ``"sutun"``: tersi.
    ``Toplam`` satırı ve sütunu çizilmez. ``labels=False``: çok sayıda sütunda değer etiketleri yazılmaz (değerler
    tabloda gösterilir).
    """

    table: str
    x_label: str
    y_label: str
    title: str
    series: str = "satir"
    stacked: bool = False
    decimals: int = 0
    labels: bool = True


@dataclass(frozen=True)
class CompareBarChart:
    """Birkaç tablonun aynı sütununu yan yana karşılaştırır.

    ``tables``: (yatay eksendeki etiket, tablo adı); her tablonun satırları birer seridir.
    """

    tables: tuple[tuple[str, str], ...]
    column: str
    x_label: str
    y_label: str
    title: str
    decimals: int = 1


@dataclass(frozen=True)
class PieChart:
    """Dilim grafiği. Dilim açıları θ = 360°·r ``result`` tablosuna yazılır (sütun ``aci``).

    Dilimler ``order`` sırasıyla, 0°'den (saat 3 yönü) başlayarak saat yönünün tersine dizilir.
    """

    table: str
    column: str
    result: str
    title: str
    order: tuple[str, ...]


@dataclass(frozen=True)
class LineChart:
    """Bir veri çerçevesinde iki değişkenin çizgi grafiği (ör. zaman serisi).

    ``references``: (skaler adı, etiket) yatay çizgileri (ör. gerçek olasılık). ``markers``: noktalar
    işaretlenir; uzun serilerde (ör. birikimli oran) yalnız çizgi.
    """

    frame: str
    x: str
    y: str
    x_label: str
    y_label: str
    title: str
    references: tuple[tuple[str, str], ...] = ()
    markers: bool = True
    bands: tuple[tuple[str, str], ...] = ()
    """(sütun, etiket): aynı çerçeveden kesikli çizgiyle çizilen ek seriler (ör. μ ± 2σ/√n bandının iki kenarı);
    boş etiketli seri açıklamada gösterilmez."""
    series: tuple[tuple[str, str], ...] = ()
    """(sütun, etiket): aynı eksende düz çizgiyle çizilen başka seriler (ör. enflasyon ve işsizlik oranı). Verilirse
    ilk seri ``y`` sütunudur ve adı ``legend`` alanıdır; bütün seriler açıklamada görünür."""
    legend: str = ""


@dataclass(frozen=True)
class ScatterPlot:
    """Saçılım grafiği: her gözlem bir (x, y) noktası.

    ``fit_line``: en küçük kareler doğrusu da çizilir (y'nin x üzerine basit regresyonu; notlardaki
    "tahmin edilen ortalama ilişkinin doğrusal özeti"). ``size``: nokta büyüklüğü, ``opacity``: saydamlık
    (çok gözlemli veride üst üste binen noktalar görünür).
    """

    frame: str
    x: str
    y: str
    x_label: str
    y_label: str
    title: str
    fit_line: bool = False
    size: float = 11.0
    opacity: float = 1.0
    lines: tuple[tuple["Parameter", "Parameter", str], ...] = ()
    """(sabit, eğim, etiket): kesikli çizgiyle çizilen bilinen doğrular y = sabit + eğim·x (ör. simülasyonda gerçek
    ortalama ilişki); sabit ve eğim sayı ya da önceden hesaplanmış bir skalerin adıdır."""
    means: str = ""
    """Boş değilse aynı x değerindeki gözlemlerin y ortalamaları (koşullu ortalamanın örneklem karşılığı) bu etiketle
    ayrı noktalar olarak çizilir (ör. eğitim düzeylerine göre ortalama ücret)."""


@dataclass(frozen=True)
class BoxPlot:
    """Yatay kutu grafiği; ``series`` ve kurallar ``BoxSummary`` ile aynıdır.

    Kutu Q₁'den Q₃'e, kutu içindeki çizgi medyandır; bıyıklar sınırların içindeki en uç gözlemlere uzanır,
    sınırların dışındaki gözlemler (aykırı değer adayları) ayrı noktalardır.
    """

    series: tuple[tuple[str, str, str], ...]
    x_label: str
    y_label: str
    title: str


@dataclass(frozen=True)
class Histogram:
    """Monte Carlo sonuç tablosundaki ya da bir veri çerçevesindeki sütunların histogramı; ``[lower, upper]``
    aralığında ``bins`` kutu.

    ``references``: (değer, etiket) dikey çizgileri (ör. gerçek anakütle değeri). Değer bir sayı ya da daha önce
    hesaplanmış bir skalerin adıdır (ör. örneklemden bulunan x̄ + 2s). ``y_label``: Monte Carlo tablosunda tekrar
    sayısı, veri çerçevesinde gözlem sayısı.
    """

    table: str
    columns: tuple[tuple[str, str], ...]
    bins: int
    lower: float
    upper: float
    title: str
    x_label: str
    references: tuple[tuple[float | str, str], ...] = ()
    y_label: str = "Tekrar sayısı"
    curves: tuple[tuple[str, "Parameter", "Parameter", str], ...] = ()
    """(dağılım, birinci, ikinci parametre, etiket): beklenen sayı eğrileri. Kutudaki beklenen sayı, ilk serinin gözlem
    sayısı × kutu genişliği × f(x)'tir; histogramın yüksekliğiyle aynı ölçektedir (``DENSITIES``)."""


@dataclass(frozen=True)
class ClassHistogram:
    """Sınıf tablosunun histogramı: sınıflar sayısal eksende bitişik dikdörtgenler.

    Dikdörtgenin genişliği sınıf genişliği, yüksekliği ``y`` sütunudur (frekans veya yüzde). ``Toplam``
    satırı çizilmez. ``labels``: değerler dikdörtgenlerin üstüne yazılır (``percent`` ise yüzde işaretiyle).
    """

    table: str
    y: str
    x_label: str
    y_label: str
    title: str
    labels: bool = False
    percent: bool = False
    decimals: int = 0


@dataclass(frozen=True)
class DotPlot:
    """Nokta grafiği: her gözlem bir nokta; aynı değerdeki gözlemler üst üste dizilir.

    ``references``: (skaler adı, etiket) dikey çizgileri (ör. ortalama, medyan). ``x_range``: yatay eksenin
    sınırları; iki veri setinin yayılımı karşılaştırılırken iki grafikte aynı eksen için (notlardaki gibi).
    """

    frame: str
    variable: str
    x_label: str
    title: str
    y_label: str = "Aynı değerdeki gözlem sayısı"
    references: tuple[tuple[str, str], ...] = ()
    x_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class MosaicChart:
    """Mozaik grafiği: çapraz tablonun (sayılar ya da ortak olasılıklar) her satırı bir sütundur.

    Sütun genişliği satır toplamının genel toplama oranıdır (marjinal olasılık); sütun içindeki parçalar satırdaki
    hücrelerin satır toplamına oranıdır (koşullu olasılık). Böylece her parçanın alanı ortak olasılığa eşittir ve
    parçaya bu değer yazılır. ``Toplam`` satırı ve sütunu çizilmez.
    """

    table: str
    x_label: str
    y_label: str
    title: str
    decimals: int = 2


@dataclass(frozen=True)
class TreeDiagram:
    """İki aşamalı olasılık ağacı, soldan sağa: kök, ilk aşamanın dalları, ikinci aşamanın dalları.

    Veri çerçevesinin her satırı bir tam yoldur: ``first`` ve ``second`` aşamaların kategorileri, ``first_p`` ilk
    dalın olasılığı, ``second_p`` ikinci dalın koşullu olasılığıdır. Yol sonunda ortak olasılık
    ``first_p × second_p`` yazılır. Kategoriler veri sırasıyla, ilk yol en üstte çizilir. Eksen yoktur.
    """

    frame: str
    first: str
    second: str
    first_p: str
    second_p: str
    root: str
    title: str
    decimals: int = 3


@dataclass(frozen=True)
class HeatMap:
    """Isı haritası: tablonun her hücresi bir kare; renk değer arttıkça koyulaşır ve değer hücreye yazılır.

    Satırlar yukarıdan aşağıya tablo sırasıyla, sütunlar soldan sağa (notlardaki ortak dağılım tablosu gibi).
    ``Toplam`` satırı ve sütunu çizilmez.
    """

    table: str
    x_label: str
    y_label: str
    title: str
    decimals: int = 2


@dataclass(frozen=True)
class DensityPlot:
    """Sürekli bir dağılımın yoğunluk eğrisi f(x) (``DENSITIES``); ``shade`` aralıklarının altı boyanır: olasılık,
    eğri altındaki alandır.

    ``first`` ve ``second``: normalde μ ve σ, tek-düzede a ve b, üstelde μ ve σ = μ, gammada biçim ve oran.
    ``x_range`` yatay eksenin sınırlarıdır (kaydırıcılar değişince aynı eksen). ``references``: (değer, etiket) dikey
    çizgileri. ``y_max`` verilirse dikey eksen 0 ile ``y_max`` arasında sabittir: σ büyüyünce eğrinin basıklaştığı
    görülür.
    """

    distribution: str
    first: float
    second: float
    x_range: tuple[float, float]
    title: str
    x_label: str
    y_label: str = "f(x)"
    shade: tuple[tuple[float, float], ...] = ()
    references: tuple[tuple[float, str], ...] = ()
    y_max: float | None = None


@dataclass(frozen=True)
class DensityCompare:
    """Aynı eksende birden fazla yoğunluk eğrisi (ör. bireysel X ile farklı n'lerde X̄'in yoğunlukları).

    ``curves``: (dağılım, birinci, ikinci parametre, etiket); parametreler sayı ya da önceden hesaplanmış bir
    skalerin adıdır (``DENSITIES``). Eğriler ``PALETTE`` sırasıyla çizilir.
    """

    curves: tuple[tuple[str, Parameter, Parameter, str], ...]
    x_range: tuple[float, float]
    title: str
    x_label: str
    y_label: str = "f(x)"
    y_max: float | None = None


@dataclass(frozen=True)
class PmfWithDensity:
    """Kesikli olasılık fonksiyonu (``frame``'in ``x`` ve ``y`` sütunları; çubuklar) ve aynı eksende sürekli bir
    yaklaşım yoğunluğu (``distribution``, ``first``, ``second``; parametre sayı ya da skaler adı).

    ``shade`` aralıklarının altı boyanır: sürekli yaklaşımda bir tam sayı değerinin karşılığı olan alan (süreklilik
    düzeltmesi, ör. X = 12 → 11,5–12,5).
    """

    frame: str
    x: str
    y: str
    distribution: str
    first: Parameter
    second: Parameter
    x_label: str
    y_label: str
    title: str
    bar_label: str
    curve_label: str
    shade: tuple[tuple[float, float], ...] = ()


# --- Ekonometri: betimleme, grup özetleri, en küçük kareler ----------------------------

DESCRIBE_STATS = ("count", "mean", "std", "min", "max")
"""Betimsel özetin sütunları (pandas ``describe()`` adlarıyla): gözlem sayısı, ortalama, örneklem standart
sapması (payda n − 1), en küçük ve en büyük değer."""
GROUP_STATS = ("count", "mean", "std", "min", "max", "sum")
MODEL_QUANTITIES = ("r2", "adj_r2", "nobs", "f", "f_p", "ssr", "df_resid")
"""Modelin tek sayıları: R², düzeltilmiş R², gözlem sayısı, genel anlamlılık F istatistiği ve p-değeri,
artık kareler toplamı, artık serbestlik derecesi."""
COEF_QUANTITIES = ("coef", "se", "t", "p", "ci_low", "ci_high")
"""Katsayının sayıları: tahmin, klasik standart hata, t istatistiği, iki yönlü p-değeri (t dağılımı, n − k
serbestlik derecesi) ve %95 güven aralığının alt/üst sınırı."""
INTERCEPT = "Intercept"
"""Sabit terimin adı (statsmodels formül yazımı); R'de ``(Intercept)`` olarak yazılır."""


@dataclass(frozen=True)
class Describe:
    """Değişkenlerin betimsel özeti; satırlar değişkenler, sütunlar ``stats`` (pandas
    ``describe().T``'deki adlarla). ``decimals``: ekranda gösterim basamağı."""

    frame: str
    variables: tuple[str, ...]
    result: str
    comment: str
    stats: tuple[str, ...] = DESCRIBE_STATS
    decimals: int = 3


@dataclass(frozen=True)
class GroupStats:
    """Grupların özeti, notlardaki ``df.groupby(by)[variable].agg([...])`` gibi: satırlar grup değerleri (iki
    gruplama değişkeninde bileşimler), sütunlar ``stats``. Gruplar küçükten büyüğe dizilir."""

    frame: str
    by: tuple[str, ...]
    variable: str
    stats: tuple[str, ...]
    result: str
    comment: str
    decimals: int = 4


@dataclass(frozen=True)
class PanelSummary:
    """Panel yapısının özeti: gözlem sayısı, birim sayısı, ilk ve son dönem, birim başına gözlenen farklı dönem
    sayısının en küçük ve en büyük değeri (eşitse dengeli panel)."""

    frame: str
    unit: str
    time: str
    result: str
    comment: str


@dataclass(frozen=True)
class OLS:
    """En küçük kareler tahmini (sabit terimli): ``outcome ~ regressors``.

    Uygulama ve üretilen Python kodu aynı çağrıyı kullanır (statsmodels ``smf.ols(...).fit()``); R'de
    ``lm()``. Standart hatalar klasik (homoskedastik) standart hatalardır; t istatistiği ve p-değeri
    n − k serbestlik dereceli t dağılımından, güven aralığı aynı dağılımın kritik değeriyle hesaplanır.
    Tahmin örneklemi modeldeki değişkenlerde eksik değeri olmayan gözlemlerdir (iki dilde aynı).
    """

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    comment: str

    def __post_init__(self) -> None:
        if not self.regressors:
            raise ValueError("Modelde en az bir açıklayıcı değişken olmalıdır.")
        if len(set(self.regressors)) != len(self.regressors) or self.outcome in self.regressors:
            raise ValueError("Açıklayıcı değişkenler tekil olmalı ve bağımlı değişkeni içermemelidir.")

    @property
    def formula(self) -> str:
        return f"{self.outcome} ~ " + " + ".join(self.regressors)


@dataclass(frozen=True)
class ShowModel:
    """Tahmin edilmiş modelin yazılım çıktısı: katsayı tablosu ve model bilgisi.

    ``columns``: ekranda gösterilen katsayı sütunları (``COEF_QUANTITIES``'den); konu henüz standart hatayı
    işlemediyse yalnız ``("coef",)``. Üretilen kod yazılımın tam çıktısını yazdırır (notlardaki gibi).
    ``stats``: model bilgisi (``MODEL_QUANTITIES``'den). ``stars``: R özetindeki anlamlılık yıldızları; yıldızlar
    henüz işlenmediyse (ör. Konu 0) ``False`` ve R çıktısı yıldızsız yazdırılır (statsmodels özetinde yıldız yoktur)."""

    model: str
    comment: str
    columns: tuple[str, ...] = COEF_QUANTITIES
    stats: tuple[str, ...] = ("nobs", "r2", "f")
    stars: bool = True


@dataclass(frozen=True)
class ModelValue:
    """Modelden tek bir sayı, skaler olarak: bir katsayının niceliği (``term`` ve ``COEF_QUANTITIES``) ya da
    modelin niceliği (``term`` yok, ``MODEL_QUANTITIES``)."""

    name: str
    model: str
    quantity: str
    comment: str
    term: str | None = None
    decimals: int = 4

    def __post_init__(self) -> None:
        allowed = COEF_QUANTITIES if self.term is not None else MODEL_QUANTITIES
        if self.quantity not in allowed:
            raise ValueError(f"Desteklenmeyen model niceliği: {self.quantity}")


@dataclass(frozen=True)
class RegressionTable:
    """Makale tipi regresyon tablosu: her model bir sütun; her terim için katsayı ve altında parantez içinde
    standart hata; en altta gözlem sayısı ve R².

    ``models``: (sütun başlığı, model adı). ``terms``: tablodaki sırayla terimler (sabit terim ``INTERCEPT``).
    Sonuç tablosunun satırları ``terim`` (katsayı), ``terim_sh`` (standart hata), ``n`` ve ``r2``'dir; modelde
    olmayan terimin hücresi boştur. ``stars``: katsayının yanında p-değerine göre yıldız (*** p < 0,01;
    ** p < 0,05; * p < 0,10); eşikler tablonun altında yazılır.

    ``standard_errors=False``: standart hata satırları yoktur (konu standart hatayı henüz işlemediyse, ör. Konu 3–4);
    yıldızlar da gösterilmez. ``r2=False``: R² satırı yoktur (ör. Konu 3; R² Konu 4'te tanımlanır)."""

    models: tuple[tuple[str, str], ...]
    terms: tuple[str, ...]
    result: str
    comment: str
    stars: bool = True
    decimals: int = 3
    standard_errors: bool = True
    r2: bool = True

    def __post_init__(self) -> None:
        if self.stars and not self.standard_errors:
            raise ValueError("Yıldızlar standart hatalarla birlikte gösterilir (standard_errors=False ise stars=False).")


@dataclass(frozen=True)
class MonteCarlo:
    """``body`` işlemlerini ``reps`` kez tekrarlar; her tekrarda ``collect`` ifadelerini toplar.

    Rastgele sayı üreteci döngüden önce ``seed`` ile bir kez tohumlanır. Sonuç tablosunun her satırı
    bir tekrardır; sütunlar ``collect`` adlarıdır.
    """

    result: str
    reps: int
    seed: int
    body: tuple["Operation", ...]
    collect: tuple[tuple[str, Expr], ...]
    comment: str


Operation = Union[
    InlineData,
    FromCounts,
    LoadWooldridge,
    SortRows,
    Describe,
    GroupStats,
    PanelSummary,
    OLS,
    ShowModel,
    ModelValue,
    RegressionTable,
    Outcomes,
    Selections,
    VariableTypes,
    Event,
    ShowFrame,
    MapCodes,
    Groups,
    Support,
    Rectangles,
    RowSum,
    Derive,
    NewSample,
    Draw,
    DrawCount,
    DrawCategory,
    DrawDiscrete,
    Shape,
    Count,
    Statistic,
    PairStatistic,
    Scalar,
    ScalarTable,
    GroupSummary,
    FrequencyTable,
    CrossTab,
    JoinColumns,
    BoxSummary,
    ClassTable,
    StemLeaf,
    Percentile,
    BarChart,
    GroupedBarChart,
    CompareBarChart,
    PieChart,
    LineChart,
    ScatterPlot,
    BoxPlot,
    Histogram,
    ClassHistogram,
    DotPlot,
    MosaicChart,
    TreeDiagram,
    HeatMap,
    DensityPlot,
    DensityCompare,
    PmfWithDensity,
    MonteCarlo,
]

CHARTS = (
    BarChart, GroupedBarChart, CompareBarChart, PieChart, LineChart, ScatterPlot, BoxPlot, Histogram, ClassHistogram,
    DotPlot, MosaicChart, TreeDiagram, HeatMap, DensityPlot, DensityCompare, PmfWithDensity,
)
AXISLESS_CHARTS = (PieChart, TreeDiagram)
"""Ekseni olmayan grafikler: eksen adı gerekmez (pasta dilimleri, olasılık ağacı)."""


# --- Notlarla karşılaştırma -------------------------------------------------

@dataclass(frozen=True)
class StatTarget:
    """Bir değişkenin (isteğe bağlı olarak bir alt gruptaki) istatistiği."""

    frame: str
    variable: str
    stat: str
    where: tuple[str, object] | None = None


@dataclass(frozen=True)
class ScalarTarget:
    name: str


@dataclass(frozen=True)
class TableTarget:
    """Bir sonuç tablosunun hücresi: satır adı ve sütun adı."""

    table: str
    row: str
    column: str


@dataclass(frozen=True)
class CellTarget:
    """Bir veri çerçevesinin ``row``'uncu gözleminin (1'den başlar) ``column`` değeri: xᵢ."""

    frame: str
    column: str
    row: int


@dataclass(frozen=True)
class CoefTarget:
    """Bir modelin katsayı niceliği (``COEF_QUANTITIES``): ör. eğitim katsayısı, standart hatası, t, p, GA."""

    model: str
    term: str
    quantity: str = "coef"

    def __post_init__(self) -> None:
        if self.quantity not in COEF_QUANTITIES:
            raise ValueError(f"Desteklenmeyen katsayı niceliği: {self.quantity}")


@dataclass(frozen=True)
class ModelTarget:
    """Bir modelin niceliği (``MODEL_QUANTITIES``): ör. R², gözlem sayısı, F istatistiği."""

    model: str
    quantity: str

    def __post_init__(self) -> None:
        if self.quantity not in MODEL_QUANTITIES:
            raise ValueError(f"Desteklenmeyen model niceliği: {self.quantity}")


Target = Union[StatTarget, ScalarTarget, TableTarget, CellTarget, CoefTarget, ModelTarget]


@dataclass(frozen=True)
class Check:
    """Notlarda basılı bir sayı ve onu üreten hesap. Tolerans, notlarda basılı basamak sayısıdır."""

    label: str
    target: Target
    expected: float
    decimals: int = 4

    @property
    def tolerance(self) -> float:
        if self.decimals <= 0:
            return 0.5
        return 0.5 * 10 ** (-self.decimals) + 1e-12


# --- Etkileşimli spesifikasyon ------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """Tek seçim (ör. açıklayıcı değişken, test türü). ``options``: (değer, öğrenciye gösterilen etiket)."""

    key: str
    label: str
    options: tuple[tuple[str, str], ...]
    default: str
    help: str = ""

    def __post_init__(self) -> None:
        values = [value for value, _ in self.options]
        if len(set(values)) != len(values) or self.default not in values:
            raise ValueError(f"{self.key}: seçenekler tekil olmalı ve varsayılanı içermelidir.")

    def normalize(self, value: object) -> str:
        if value not in {item for item, _ in self.options}:
            raise ValueError(f"{self.label}: geçersiz seçim {value!r}.")
        return str(value)

    def option_label(self, value: str) -> str:
        return dict(self.options)[value]


@dataclass(frozen=True)
class MultiChoice:
    """Çoklu seçim (ör. açıklayıcı değişken kümesi). Seçim ``options`` sırasıyla tutulur; en az ``minimum``,
    en çok ``maximum`` öğe seçilir."""

    key: str
    label: str
    options: tuple[tuple[str, str], ...]
    default: tuple[str, ...]
    help: str = ""
    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        values = [value for value, _ in self.options]
        if len(set(values)) != len(values) or not set(self.default) <= set(values):
            raise ValueError(f"{self.key}: seçenekler tekil olmalı ve varsayılanı içermelidir.")
        if self.normalize(self.default) != tuple(self.default):
            raise ValueError(f"{self.key}: varsayılan seçim seçeneklerin sırasıyla yazılmalıdır.")

    def normalize(self, value: object) -> tuple[str, ...]:
        chosen = set(value) if isinstance(value, (list, tuple, set, frozenset)) else {value}
        unknown = chosen - {item for item, _ in self.options}
        if unknown:
            raise ValueError(f"{self.label}: geçersiz seçim {sorted(map(str, unknown))}.")
        ordered = tuple(item for item, _ in self.options if item in chosen)
        if len(ordered) < self.minimum:
            raise ValueError(f"{self.label}: en az {self.minimum} seçim yapınız.")
        if self.maximum is not None and len(ordered) > self.maximum:
            raise ValueError(f"{self.label}: en çok {self.maximum} seçim yapılabilir.")
        return ordered

    def option_label(self, value: str) -> str:
        return dict(self.options)[value]


@dataclass(frozen=True)
class NumberChoice:
    """Sayı seçimi (kaydırıcı), ör. tahmin noktası x₀ ya da anlamlılık düzeyi."""

    key: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    help: str = ""
    integer: bool = False
    decimals: int = 2

    def __post_init__(self) -> None:
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"{self.key}: varsayılan değer aralığın içinde olmalıdır.")

    def normalize(self, value: object) -> float | int:
        number = float(value)
        if not self.minimum <= number <= self.maximum:
            raise ValueError(f"{self.label}: değer {self.minimum}–{self.maximum} aralığında olmalıdır.")
        # Kesirli değerler koda iki basamakla geçer (0,30000000000000004 gibi yazımlar olmaz).
        return int(round(number)) if self.integer else round(number, self.decimals)


Control = Union[Choice, MultiChoice, NumberChoice]
Choices = Mapping[str, object]


def _names(value) -> set[str]:
    """Bir işlem alanındaki ad(lar): dizge, dizge demeti ya da (etiket, ad) çiftleri."""

    if isinstance(value, str):
        return {value}
    if isinstance(value, tuple):
        found: set[str] = set()
        for item in value:
            if isinstance(item, str):
                found.add(item)
            elif isinstance(item, tuple):
                found |= {part for part in item if isinstance(part, str)}
        return found
    return set()


_READ_FIELDS = ("frame", "source", "table", "model", "models", "tables", "series", "columns")
_WRITE_FIELDS = ("result", "name")
_FRAME_WRITERS = ("LoadWooldridge", "InlineData", "FromCounts", "Outcomes", "Selections", "Support", "Rectangles",
                  "NewSample", "Derive", "Event", "MapCodes", "Groups", "RowSum", "Draw", "DrawCount", "DrawCategory",
                  "DrawDiscrete", "SortRows")
_FRAME_CREATORS = ("LoadWooldridge", "InlineData", "FromCounts", "Outcomes", "Selections", "Support", "Rectangles",
                   "NewSample")
"""Çerçeveyi baştan kuran işlemler: eski çerçeveyi okumaz, yerine yenisini yazar (ör. veriyi yeniden yükleme)."""


def operation_reads(op) -> set[str]:
    """İşlemin kullandığı adlar (veri çerçevesi, tablo, model, skaler). Tutucu bir üst küme yeterlidir."""

    from core.labs import expr as E

    creator = type(op).__name__ in _FRAME_CREATORS
    found: set[str] = set()
    for item in fields(op):
        value = getattr(op, item.name)
        if item.name in _READ_FIELDS and not (creator and item.name in ("frame", "columns")):
            found |= _names(value)
        elif isinstance(value, (E.Var, E.Const, E.BinOp, E.Call, E.Ref)):
            found |= E.references(value)
    if hasattr(op, "frame") and not creator:
        found.add(op.frame)
    return found


def operation_writes(op) -> set[str]:
    """İşlemin yazdığı ya da değiştirdiği adlar."""

    found = {value for name in _WRITE_FIELDS if isinstance(value := getattr(op, name, None), str)}
    if type(op).__name__ in _FRAME_WRITERS:
        found.add(op.frame)
    return found


def tainted_writes(default: tuple, operations: tuple, tainted: set[str]) -> tuple[bool, set[str]]:
    """Bir adımın işlemlerinden sonra seçime bağlı adlar: ``(adımda seçime bağlı işlem var mı, yeni küme)``.

    Bir işlem, notlardaki (varsayılan) tanımda birebir yoksa ya da seçime bağlı bir adı okuyorsa seçime bağlıdır ve
    yazdığı adlar seçime bağlı olur. Seçime bağlı olmayan bir işlem yazdığı adları notlardaki gibi yeniden kurar.
    Varsayılan tanımın yazdığı ama seçilen tanımın yazmadığı adlar da seçime bağlıdır (artık aynı değildir).
    """

    tainted = set(tainted)
    known = set(default)
    dirty = False
    for op in operations:
        if op not in known or operation_reads(op) & tainted:
            dirty = True
            tainted |= operation_writes(op)
        else:
            tainted -= operation_writes(op)
    missing = set().union(set(), *(operation_writes(op) for op in default)) - set().union(
        set(), *(operation_writes(op) for op in operations))
    if missing:
        dirty = True
        tainted |= missing
    return dirty, tainted


# --- Adım ve uygulama --------------------------------------------------------

@dataclass(frozen=True)
class NoteRef:
    section: str
    step: int = 0
    objects: tuple[str, ...] = ()

    def label(self) -> str:
        parts = [f"Notlar §{self.section}"]
        if self.step:
            parts.append(f"Adım {self.step}")
        parts.extend(self.objects)
        return " · ".join(parts)


@dataclass(frozen=True)
class LabStep:
    """Uygulamanın bir adımı; notlardaki bir alt bölüme bağlıdır.

    ``controls`` ve ``build`` verilirse adım etkileşimlidir: ``build(seçimler)`` işlem listesini üretir ve
    ``operations`` varsayılan seçimlerin (notlardaki spesifikasyonun) listesidir. ``note_for``: seçimlere ve
    sonuçlara göre değişen açıklama metni (ör. seçilen değişkenle katsayının yorumu); verilmezse ``takeaway``.
    """

    number: int
    title: str
    note: NoteRef
    explanation: str
    operations: tuple[Operation, ...] = ()
    checks: tuple[Check, ...] = ()
    reproducibility: ReproClass = ReproClass.EXACT
    takeaway: str = ""
    code_note: str = ""
    controls: tuple[Control, ...] = ()
    uses: tuple[Control, ...] = ()
    """Önceki adımların denetimleri: bu adımın işlemleri onların seçimine de bağlıdır (ör. Adım 4'te seçilen
    açıklayıcı değişkenin katsayısını yorumlayan Adım 5)."""
    build: Callable[[Choices], tuple[Operation, ...]] | None = field(default=None, compare=False, repr=False)
    note_for: Callable[[object, Choices], str] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (self.controls or self.uses) and self.build is None:
            raise ValueError(f"Adım {self.number}: denetimi olan ya da denetim kullanan adımın build'i olmalıdır.")
        if self.build is not None and not (self.controls or self.uses):
            raise ValueError(f"Adım {self.number}: build yalnız denetimi olan adımlarda kullanılır.")
        if self.build is not None and tuple(self.build(self.defaults())) != tuple(self.operations):
            raise ValueError(f"Adım {self.number}: operations, varsayılan seçimlerin işlem listesi olmalıdır.")

    @property
    def key(self) -> str:
        return f"adim{self.number}"

    def defaults(self) -> dict[str, object]:
        return {control.key: control.default for control in (*self.uses, *self.controls)}


def interactive_step(*, number: int, title: str, note: NoteRef, explanation: str,
                     build: Callable[[Choices], tuple[Operation, ...]], controls: tuple[Control, ...] = (),
                     uses: tuple[Control, ...] = (), **options) -> LabStep:
    """Etkileşimli adım: ``operations`` varsayılan seçimlerden (notlardaki spesifikasyondan) üretilir."""

    defaults = {control.key: control.default for control in (*uses, *controls)}
    return LabStep(number=number, title=title, note=note, explanation=explanation, operations=tuple(build(defaults)),
                   controls=controls, uses=uses, build=build, **options)


@dataclass(frozen=True)
class LabSpec:
    """Bir konunun Uygulama sekmesi (``kind="uygulama"``) veya tek bir Sezgi deneyi (``kind="sezgi"``).

    ``variant``: ``resolve`` ile üretilmiş tanımda notlardakinden farklı sonuç veren adımların numaraları
    (seçimi değişen adımlar ve onların sonucunu kullanan sonraki adımlar); bu adımlarda notlarla
    karşılaştırma yapılmaz.
    """

    topic_key: str
    title: str
    note_section: str
    steps: tuple[LabStep, ...]
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    consistency_notes: tuple[str, ...] = field(default_factory=tuple)
    kind: str = "uygulama"
    variant: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        keys = [control.key for step in self.steps for control in step.controls]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.topic_key}: denetim anahtarları tekil olmalıdır.")
        earlier: set[object] = set()
        for step in self.steps:
            if not set(step.uses) <= earlier:
                raise ValueError(f"{self.topic_key} Adım {step.number}: kullanılan denetim önceki bir adımda olmalıdır.")
            earlier |= set(step.controls)

    def label(self, name: str) -> str:
        """Değişken, sütun veya tablo için öğrenciye gösterilecek Türkçe ad."""

        return dict(self.labels).get(name, name)

    def step(self, number: int) -> LabStep:
        for item in self.steps:
            if item.number == number:
                return item
        raise ValueError(f"Adım bulunamadı: {number}")

    def operations_through(self, number: int) -> tuple[Operation, ...]:
        """Bir adımı tek başına çalıştırmak için gereken bütün önceki işlemler."""

        collected: list[Operation] = []
        for item in self.steps:
            if item.number > number:
                break
            collected.extend(item.operations)
        return tuple(collected)

    # --- Etkileşimli spesifikasyon ---------------------------------------------
    @property
    def controls(self) -> tuple[Control, ...]:
        return tuple(control for step in self.steps for control in step.controls)

    def defaults(self) -> dict[str, object]:
        return {control.key: control.default for control in self.controls}

    def normalize(self, choices: Choices) -> dict[str, object]:
        """Bütün denetimlerin seçimi; verilmeyen denetim varsayılanını alır. Geçersiz seçim hatadır."""

        normalized = {}
        for control in self.controls:
            value = choices.get(control.key, control.default)
            normalized[control.key] = control.normalize(value)
        unknown = set(choices) - set(normalized)
        if unknown:
            raise ValueError(f"Tanımsız denetim: {sorted(unknown)}")
        return normalized

    def resolve(self, choices: Choices) -> "LabSpec":
        """Seçimlerle yeniden kurulan tanım.

        Seçimi varsayılandan farklı bir adım notlardan farklıdır; o adımın seçime bağlı bir çıktısını (veri çerçevesi,
        tablo, model, skaler) okuyan sonraki adımlar da farklıdır. Farklı adımların notlarla karşılaştırması kaldırılır.

        Bağımlılık işlem düzeyinde izlenir (``tainted_writes``): değişen bir adımda notlardaki tanımla birebir aynı
        olan ve seçime bağlı bir adı okumayan işlemler (ör. veriyi yeniden yükleme) çıktılarını notlardaki gibi kurar;
        yalnız farklı işlemlerin ve onlara bağlı işlemlerin yazdığı adlar sonraki adımlara geçer.
        """

        chosen = self.normalize(choices)
        defaults = {control.key: control.normalize(control.default) for control in self.controls}
        steps: list[LabStep] = []
        tainted: set[str] = set()
        variant: list[int] = []
        for step in self.steps:
            relevant = (*step.uses, *step.controls)
            own = {control.key: chosen[control.key] for control in relevant}
            changed = any(own[control.key] != defaults[control.key] for control in relevant)
            operations = tuple(step.build(own)) if step.build is not None and changed else step.operations
            dirty, tainted = tainted_writes(step.operations, operations, tainted)
            if changed or dirty:
                variant.append(step.number)
                # Denetimler özgün tanımda kalır (arayüz onları oradan çizer); bu adım artık seçimlerin sonucudur.
                steps.append(replace(step, operations=operations, checks=(), controls=(), uses=(), build=None))
            else:
                steps.append(step)
        return replace(self, steps=tuple(steps), variant=tuple(variant))
