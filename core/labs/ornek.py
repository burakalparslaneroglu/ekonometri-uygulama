"""Uygulama sekmesinin ek veri kaynakları: alternatif örnek ve öğrencinin kendi verisi.

Ders notlarının çözümlü örnekleri (``core.labs.konuNN``) değişmez. Her konu için ek olarak bir **genel uygulama**
yazılır (``core.labs.ornek_konuNN``): aynı adımlar, aynı numaralar ve aynı işlemler, fakat veri bir ``Case``'ten gelir.
Alternatif örnek, genel uygulamanın başka Wooldridge (2020) veri setleriyle (deney adımında kurgusal bir veriyle)
kurulmuş hâlidir; "Kendi verini yükle" seçeneğinde aynı genel uygulama öğrencinin dosyasıyla kurulur. Böylece iki ek
kaynak tek bir tanımı paylaşır. Etkileşimli adımların denetimleri de genel uygulamadadır: seçenekler verinin
değişkenlerinden kurulur, varsayılan seçim örneğin temel spesifikasyonudur.

Notlar dışındaki kaynaklarda kontrollerin beklenen değerleri uygulamanın kendi hesabıdır (``with_app_values``):
indirilen kod bu değerleri yeniden üretmelidir. Alternatif örneklerin değerleri ayrıca testlerde bağımsız bir hesapla
doğrulanır.

Öğrencinin sütun ve kategori adları metinlere ``md`` ile girer: Markdown ve KaTeX işaretleri kaçırılır, böylece bir ad
(ör. "Fiyat ($)", "a|b", "1. sınıf") sayfanın biçimini bozmaz. Ad hiçbir zaman matematik ifadesinin içine yazılmaz.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping

import pandas as pd

import numpy as np

from core.labs.runner import run_lab
from core.labs.spec import CellTarget, CoefTarget, LabSpec, ModelTarget, Operation, TableTarget

SOURCE_LABELS = {
    "notlar": "Notlardaki örnek",
    "alternatif": "Alternatif örnek",
    "kendi": "Kendi verini yükle",
}
POSITIVE_WORDS = ("1", "evet", "var", "geçti", "başarılı", "tuttu", "memnun", "dönüştü", "doğru", "olumlu", "kabul",
                  "program", "tedavi", "deney", "katıldı", "yes", "true")
"""İki kategorili bir değişkende varsayılan "1" kategorisi (ilk eşleşen; büyük-küçük harf Türkçe kuralıyla)."""


# --- Sayı yazımı (metinler için) --------------------------------------------------------

def sayi(value: float, decimals: int = 0) -> str:
    """Türkçe sayı: ondalık virgül, tipografik eksi (0,625; −1,5). Yuvarlanınca sıfır olan değer "−0" yazılmaz."""

    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f"{value:.{decimals}f}".replace(".", ",").replace("-", "−")


def yuzde(value: float, decimals: int = 1) -> str:
    """Yüzde işareti sayıdan önce (%62,5)."""

    return "%" + sayi(value, decimals)


def kisa(value: float, decimals: int = 3) -> str:
    """Gereksiz sıfırları atılmış Türkçe sayı (0,320 → 0,32; 15,0 → 15). Sıfırlar yalnız ondalık kısımdan atılır."""

    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text.replace(".", ",").replace("-", "−")


def sayim(value: float) -> str:
    """Sayım: binlik ayırıcı nokta (4.360)."""

    return f"{int(round(value)):,}".replace(",", ".")


def liste(items: list[str]) -> str:
    """Türkçe sıralama: "A", "A ve B", "A, B ve C"."""

    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " ve " + items[-1]


_MARKDOWN = re.compile(r"([\\`*_\[\]<>#|$~&])")


def md(text: object) -> str:
    """Markdown metnine girecek kullanıcı metni (sütun ya da kategori adı): biçim işaretleri kaçırılır, satır sonları
    boşluk olur, baştaki "1." numaralı liste sanılmaz."""

    value = re.sub(r"\s*[\r\n]+\s*", " ", str(text))
    value = _MARKDOWN.sub(r"\\\1", value)
    return re.sub(r"^(\d+)([.)])", r"\1\\\2", value)


def free_name(base: str, taken: Iterable[str]) -> str:
    """Türetilen sütun için kullanılmayan ad (``base``, ``base_2``, …)."""

    used = set(taken)
    candidate, index = base, 2
    while candidate in used:
        candidate, index = f"{base}_{index}", index + 1
    return candidate


def default_pick(categories: Iterable[str]) -> str:
    """İki kategorili değişkende 1 ile kodlanacak varsayılan kategori: {0, 1} kodunda 1, aksi hâlde Evet, Var, Program
    gibi olumlu bir kategori; yoksa sıradaki son kategori (alfabetik sırada ikincisi)."""

    from core.labs.kendi_veri import fold

    items = list(categories)
    folded = {fold(item): item for item in items}
    for word in POSITIVE_WORDS:
        if word in folded:
            return folded[word]
    return items[-1]


# --- Örnek (vaka) ---------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Case:
    """Genel uygulamanın verisi ve değişkenlerin rolleri.

    ``load``: veriyi kuran işlemler (alternatif örnekte Wooldridge veri seti, kendi verinde ``ReadFile``). ``data``: aynı
    verinin temizlenmiş hâli (seçenekleri, metinleri ve kontrolleri kurmak için). ``roles``: rol → sütun (koddaki ad);
    ``extras``: ek sayısal sütunlar (koddaki adlar); ``labels``: sütun → ekranda görünen ad; ``units``: sütun → birim
    (metinlerde; kendi verinde boş); ``levels``: rol → 1 ile kodlanan kategori; ``orders``: sütun → kategori sırası;
    ``unit``: gözlem biriminin adı (ör. "çalışan"); ``extra``: konuya özgü ayarlar ve metinler.
    """

    source: str
    load: tuple[Operation, ...]
    frame: str
    data: pd.DataFrame
    roles: Mapping[str, str]
    labels: Mapping[str, str]
    extras: tuple[str, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    levels: Mapping[str, str] = field(default_factory=dict)
    orders: Mapping[str, tuple] = field(default_factory=dict)
    unit: str = "gözlem"
    extra: Mapping[str, object] = field(default_factory=dict)

    def column(self, role: str) -> str:
        return self.roles[role]

    def label(self, role: str) -> str:
        return self.name(self.roles[role])

    def name(self, column: str) -> str:
        """Sütunun ekranda görünen adı."""

        return self.labels.get(column, column)

    def md(self, role: str) -> str:
        """Rolün sütun adı, Markdown metnine girecek biçimde."""

        return md(self.label(role))

    def has(self, role: str) -> bool:
        return role in self.roles

    @property
    def own(self) -> bool:
        return self.source == "kendi"


def with_app_values(spec: LabSpec) -> LabSpec:
    """Kontrollerin beklenen değerlerini uygulamanın kendi hesabıyla doldurur (notlar dışındaki kaynaklar).

    Kendi verinde tablodaki boş bir hücre (``CellTarget``) kontrol edilmez; başka bir değerin hesaplanamaması bir
    hatadır (doğrulama bunu önlemeliydi).
    """

    run = run_lab(spec)
    steps = []
    for step in spec.steps:
        results = run.step_checks(step.number)
        checks = []
        for result in results:
            if not math.isfinite(result.value):
                if spec.source == "kendi" and isinstance(result.check.target, CellTarget):
                    continue
                raise ValueError(f"{result.check.label}: değer hesaplanamadı.")
            # Gösterim basamağında sıfır olan değer (ör. 1e-17) kodda "-0.000" yazılmasın diye sıfır alınır.
            value = 0.0 if abs(result.value) < 0.5 * 10 ** (-result.check.decimals) else result.value
            checks.append(replace(result.check, expected=value))
        steps.append(replace(step, checks=tuple(checks)))
    return replace(spec, steps=tuple(steps))


def usable_pair(data: pd.DataFrame, first: str, second: str, minimum: int = 3) -> bool:
    """İki sütun birlikte kullanılabilir mi: ikisinin de değeri olan en az ``minimum`` gözlem ve bu gözlemlerde iki
    sütunun da en az iki farklı değeri (sabit bir sütunla korelasyon ve eğim tanımsızdır)."""

    complete = data[[first, second]].dropna()
    return len(complete) >= minimum and complete[first].nunique() > 1 and complete[second].nunique() > 1


EXACT_FIT = 1e-9
"""Artık kareler toplamı, toplam kareler toplamının bu oranından küçükse uyum (neredeyse) tamdır."""


def exact_fit(data: pd.DataFrame, y: str, x: str) -> bool:
    """Noktalar (neredeyse) tam bir doğrunun üzerinde mi (R² ≈ 1). Böyle bir veride standart hata sıfıra çok yakındır;
    t, p-değeri, güven aralığı ve F yuvarlama hatasına duyarlıdır, Python ile R'de aynı çıkmaz."""

    complete = data[[y, x]].dropna().astype(float)
    slope, intercept = np.polyfit(complete[x], complete[y], 1)
    residual = complete[y] - (intercept + slope * complete[x])
    total = float(((complete[y] - complete[y].mean()) ** 2).sum())
    return float((residual ** 2).sum()) <= EXACT_FIT * total


_FRAGILE = ("se", "t", "p", "ci_low", "ci_high")


def stable_checks(checks, exact: bool, table: str | None = None):
    """Uyum tamsa standart hataya bağlı kontroller (standart hata, t, p, güven aralığı, F) çıkarılır.
    ``table``: makale tablosu; parantez içindeki standart hata satırı (``…_sh``) da çıkarılır."""

    if not exact:
        return tuple(checks)

    def fragile(check) -> bool:
        target = check.target
        if isinstance(target, CoefTarget):
            return target.quantity in _FRAGILE
        if isinstance(target, ModelTarget):
            return target.quantity in ("f", "f_p")
        return isinstance(target, TableTarget) and target.table == table and str(target.row).endswith("_sh")

    return tuple(check for check in checks if not fragile(check))


EXACT_FIT_NOTE = (" Noktaların hepsi neredeyse tam bir doğrunun üzerindedir (R² ≈ 1). Böyle bir veride çıktıdaki standart "
                  "hata sıfıra çok yakındır; t, p-değeri, güven aralığı ve F yuvarlama hatasına duyarlıdır ve indirilen "
                  "kodda karşılaştırılmaz.")


# --- Kendi verin: roller ----------------------------------------------------------------

@dataclass(frozen=True)
class Role:
    """Kendi verinde öğrencinin bir sütun seçtiği rol.

    ``use``: ``kategorik``, ``sayisal`` ya da ``serbest`` (olduğu gibi; ör. kimlik sütunu). ``required``: rol zorunludur
    ve bu sütunda değeri olmayan satırlar analizden çıkarılır. ``levels``: kategorik rolün kategori sayısı aralığı.
    ``pick``: verilirse öğrenci bu rolün kategorilerinden birini de seçer (ör. programa atanan grup, kod 1). ``group``:
    aynı adımda birlikte gereken rollerin adı (ör. atama ve sonuç); grubun bir rolü seçilirse hepsi seçilmelidir.
    """

    key: str
    label: str
    use: str
    required: bool
    steps: tuple[int, ...]
    help: str
    levels: tuple[int, int] = (2, 15)
    pick: str | None = None
    group: str | None = None
    suggest: bool = False
    """İsteğe bağlı rol için de dosyadan bir sütun önerilir (öğrenci kaldırabilir)."""
    complete: bool = False
    """Seçilirse sütunda boş hücre olamaz (ör. kimlik ya da dönem sütunu)."""
    unique: bool = False
    """Seçilirse sütundaki değerler birbirinden farklı olmalı (kimlik sütunu)."""


@dataclass(frozen=True)
class CustomLab:
    """Bir konunun "Kendi verini yükle" tanımı: roller, genel uygulamayı kuran fonksiyon ve örnek dosya.

    ``extra_columns``: öğrenci rollerin dışında ek sütunlar da seçebilir (``extra_use`` kullanımıyla; ör. ek sayısal
    değişkenler), en çok ``max_extra`` sütun.
    """

    roles: tuple[Role, ...]
    build: Callable[[Case], LabSpec]
    sample: Callable[[], pd.DataFrame]
    intro: str
    order_roles: tuple[str, ...] = ()
    """Kategori sırası seçeneğinin uygulandığı roller."""
    min_rows: int = 5
    extra_columns: bool = False
    extra_use: str = "sayisal"
    extra_label: str = "Ek sayısal değişkenler (isteğe bağlı)"
    extra_help: str = ""
    max_extra: int = 6
    validate: Callable[[Case], None] | None = None
    """Konuya özgü ek denetim (ör. dönem sütunu); kullanılamıyorsa ``UploadError``."""
    suggest: Callable[[object], Mapping[str, str]] | None = None
    """Dosyanın ilk açılışında rollere önerilen sütunlar (``UploadedTable`` → rol → sütun; ör. panel yapısı); öneri
    yoksa zorunlu roller ve ``Role.suggest`` rolleri için genel kural kullanılır."""


@dataclass(frozen=True)
class TopicVariants:
    """Bir konunun ek veri kaynakları."""

    alternative: Callable[[], LabSpec]
    story: str
    """Alternatif örneğin kısa tanımı (sekmenin üstünde gösterilir)."""
    custom: CustomLab | None = None


# --- Kendi verin: seçimlerden örneğe -----------------------------------------------------

ORDER_TEXT = {
    "alfabetik": "alfabetik sırayla",
    "dosya": "dosyadaki ilk görülme sırasıyla",
    "frekans": "frekansa göre (çoktan aza)",
}


@dataclass(frozen=True)
class CustomChoices:
    """Öğrencinin veri panelindeki seçimleri.

    ``roles``: rol → dosyadaki sütun adı (seçilmediyse ``None``); ``extra``: ek sütunlar (dosyadaki adlar); ``order``:
    kategori sırası kuralı (``kendi_veri.ORDER_RULES``); ``picks``: rol → seçilen kategori.
    """

    roles: Mapping[str, str | None]
    extra: tuple[str, ...] = ()
    order: str = "alfabetik"
    picks: Mapping[str, str] = field(default_factory=dict)


def _selections(custom: CustomLab, table, choices: CustomChoices):
    from core.labs import kendi_veri as K

    for role in custom.roles:
        if role.required and not choices.roles.get(role.key):
            raise K.UploadError(f"“{role.label}” için bir sütun seçin.")
    groups: dict[str, list[Role]] = {}
    for role in custom.roles:
        if role.group:
            groups.setdefault(role.group, []).append(role)
    for members in groups.values():
        chosen = [role for role in members if choices.roles.get(role.key)]
        if chosen and len(chosen) < len(members):
            missing = [f"“{role.label}”" for role in members if not choices.roles.get(role.key)]
            raise K.UploadError("Bu roller birlikte çalışır; şunlar için de sütun seçin: " + liste(missing) + ".")
    if len(choices.extra) > custom.max_extra:
        raise K.UploadError(f"En çok {custom.max_extra} ek sütun seçilebilir.")
    uses: dict[str, str] = {}
    required: dict[str, bool] = {}
    for role in custom.roles:
        original = choices.roles.get(role.key)
        if not original:
            continue
        if original not in table.columns:
            raise K.UploadError(f"“{original}” sütunu dosyada yok.")
        previous = uses.get(original)
        if previous is not None and previous != role.use:
            raise K.UploadError(f"“{original}” sütunu farklı türde iki rol için seçildi; her rol için uygun bir sütun "
                                "seçin.")
        uses[original] = role.use
        required[original] = required.get(original, False) or role.required
    for original in choices.extra:
        if original not in table.columns:
            raise K.UploadError(f"“{original}” sütunu dosyada yok.")
        if original in uses:
            raise K.UploadError(f"“{original}” sütunu hem bir rol için hem ek sütun olarak seçildi.")
        uses[original] = custom.extra_use
    # Kod adları dosyadaki bütün sütunlar için dosya sırasıyla verilir: bir sütunun adı seçimlere bağlı değildir
    # ("Gelir (TL)" ile "Gelir TL" hangi rollere seçilirse seçilsin gelir_tl ve gelir_tl_2 olur); saklanan adım
    # seçimleri rol değişince başka bir sütunu göstermez.
    taken: set[str] = set()
    codes: dict[str, str] = {}
    for original in table.columns:
        codes[original] = K.code_name(original, taken)
        taken.add(codes[original])
    return [K.Selection(codes[original], original, use, required=required.get(original, False))
            for original, use in uses.items()]


def custom_case(custom: CustomLab, table, choices: CustomChoices) -> tuple[Case, tuple[str, ...]]:
    """Öğrencinin dosyası ve seçimlerinden genel uygulamanın örneğini kurar; kullanılamıyorsa ``UploadError``."""

    from core.labs import kendi_veri as K

    selections = _selections(custom, table, choices)
    prepared = K.prepare(table, selections, frame="veri", comment=f"Yüklediğiniz veri dosyası: {table.file_name}")
    data = prepared.frame
    if len(data) < custom.min_rows:
        raise K.UploadError(f"Analiz için en az {custom.min_rows} gözlem gerekir; seçilen sütunlarda {len(data)} "
                            "gözlem var.")
    names = {item.original: item.name for item in selections}
    roles = {role.key: names[choices.roles[role.key]] for role in custom.roles if choices.roles.get(role.key)}
    extras = tuple(names[original] for original in choices.extra)
    labels = {item.name: item.original for item in selections}
    for role in custom.roles:
        if role.key not in roles:
            continue
        column = roles[role.key]
        values = data[column]
        blanks = int(values.isna().sum())
        if role.complete and blanks:
            raise K.UploadError(f"“{labels[column]}” sütununda {blanks} boş hücre var. “{role.label}” rolündeki "
                                "sütunda her gözlemin değeri olmalı; boş hücreleri doldurun ya da başka bir sütun "
                                "seçin.")
        if role.unique and values.duplicated().any():
            repeated = values[values.duplicated()].iloc[0]
            raise K.UploadError(f"“{labels[column]}” sütununda tekrar eden değerler var (ör. “{repeated}”). "
                                f"“{role.label}” rolündeki sütunda her gözlemin değeri farklı olmalı.")
    orders: dict[str, tuple] = {}
    levels: dict[str, str] = {}
    for role in custom.roles:
        if role.key not in roles or role.use != "kategorik":
            continue
        column = roles[role.key]
        K.check_levels(data[column], labels[column], *role.levels)
        orders[column] = K.category_order(data[column], choices.order)
        if role.pick:
            pick = choices.picks.get(role.key)
            levels[role.key] = pick if pick in orders[column] else default_pick(orders[column])
    case = Case(
        source="kendi",
        load=(prepared.read,),
        frame="veri",
        data=data,
        roles=roles,
        labels=labels,
        extras=extras,
        levels=levels,
        orders=orders,
        unit="gözlem",
        extra={"order_text": ORDER_TEXT[choices.order], "file_name": table.file_name},
    )
    if custom.validate is not None:
        custom.validate(case)
    return case, prepared.notes
