"""Konu 3–7 genel uygulamalarının ortak parçaları: veri, roller, adlar ve sayı denetimleri.

Alternatif örneklerin verisi:

* ücret adımları: WAGE2 (Wooldridge, 2020; 935 erkek çalışan, 1980; ``wage`` aylık kazançtır, ABD doları);
* notlarda HPRICE1 kullanılan konut adımları: KIELMC'nin 1978 satışları (179 konut, fiyat 1978 dolarıyla). 1978 tek
  dönemlik bir yatay kesittir; 1981 satışları çöp yakma tesisinden etkilenir (Konu 2'deki havuzlanmış yatay kesit);
* Konu 3'ün sıfır–bir değişken adımı: Konu 2'nin kurgusal iş arama programı (``core.labs.kurgusal_veri``; kurayla
  atama, veri üretim süreci modül belgesinde). Wooldridge paketinde JTRAIN2 dışında rastgele atamalı veri yoktur.

"Kendi verini yükle" seçeneğinde bütün adımlar öğrencinin dosyasıyla kurulur: notlarda ikinci bir veri setiyle yapılan
adımlar (konut adımları) aynı dosyanın sonuç ve açıklayıcı değişkenini kullanır. Seçilen bütün sütunlarda (sonuç,
temel açıklayıcı ve ek değişkenler) boş hücresi olan satırlar çıkarılır: bütün modeller aynı gözlemlerle kurulur
(kısmi regresyon, eksik değişken ayrıştırması ve R² karşılaştırması aynı örneklemi ister).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from core import wooldridge_data as W
from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.kurgusal_veri import PROGRAM_COLUMNS, PROGRAM_LABELS, PROGRAM_ROWS
from core.labs.ornek import EXACT_FIT, Case, CustomLab, Role, free_name, md, sayi, usable_pair
from core.labs.spec import (
    Derive,
    InlineData,
    LabSpec,
    LoadWooldridge,
    NumberChoice,
    Operation,
    ReadFile,
    TakeRows,
)
from core.labs.wording import tr_lower

SONUC, ACIKLAYICI, GOSTERGE = "sonuc", "aciklayici", "gosterge"
MAX_EXTRA = 3
"""Kendi verinde en çok ek sayısal değişken: notlardaki seçenek kümeleri üç ya da dört değişkenlidir."""
MAX_LEVELS = 25
"""Düzey ortalamaları ve koşullu ortalama için açıklayıcı değişkenin en çok farklı değer sayısı."""

WAGE2, KIELMC, KONUT, PROGRAM = "wage2", "kielmc", "konut", "program"
HOUSE_YEAR = 1978
HOUSE_COLUMNS = ("price", "lprice", "area", "larea", "rooms", "baths", "land")
"""KIELMC'den 1978 konutlarında tutulan sütunlar (yaş WAGE2'deki yaşla aynı adı taşıdığı için alınmaz)."""

# --- WAGE2 ve KIELMC'nin cümle içindeki adları ------------------------------------------------------------

WAGE2_PHRASES = {
    "wage": "aylık kazanç", "lwage": "log aylık kazanç", "educ": "eğitim", "exper": "iş deneyimi",
    "tenure": "kıdem", "IQ": "IQ puanı", "KWW": "iş dünyası bilgisi puanı", "sibs": "kardeş sayısı",
}
WAGE2_STEPS = {
    "educ": "eğitim süresi bir yıl daha uzun olan",
    "exper": "iş deneyimi bir yıl daha fazla olan",
    "tenure": "mevcut işverendeki kıdemi bir yıl daha fazla olan",
    "IQ": "IQ puanı bir puan daha yüksek olan",
    "KWW": "iş dünyası bilgisi puanı bir puan daha yüksek olan",
    "sibs": "kardeş sayısı bir kişi daha fazla olan",
}
"""Bir birimlik farkın cümle içindeki yazımı (… olan çalışanların)."""
WAGE2_UNITS = {"wage": "dolar", "educ": "yıl", "exper": "yıl", "tenure": "yıl", "IQ": "puan", "KWW": "puan",
               "sibs": "kişi"}
WAGE2_LABELS = {
    "wage": ("Aylık kazanç", "ABD doları/ay"), "lwage": ("Aylık kazancın logaritması", "log"),
    "educ": ("Eğitim", "yıl"), "exper": ("İş deneyimi", "yıl"), "tenure": ("Mevcut işverendeki kıdem", "yıl"),
    "IQ": ("IQ puanı", "puan"), "KWW": ("İş dünyası bilgisi testi puanı", "puan"), "sibs": ("Kardeş sayısı", "kişi"),
}

HOUSE_PHRASES = {
    "price": "satış fiyatı", "lprice": "log satış fiyatı", "area": "konut büyüklüğü", "larea": "log konut büyüklüğü",
    "rooms": "oda sayısı", "baths": "banyo sayısı", "land": "arsa büyüklüğü",
}
HOUSE_STEPS = {
    "area": "büyüklüğü bir fit² daha fazla olan",
    "rooms": "bir odası daha fazla olan",
    "baths": "bir banyosu daha fazla olan",
    "land": "arsası bir fit² daha büyük olan",
}
HOUSE_UNITS = {"price": "dolar", "area": "fit²", "rooms": "oda", "baths": "banyo", "land": "fit²"}
HOUSE_LABELS = {
    "price": ("Satış fiyatı", "ABD doları, 1978"), "lprice": ("Satış fiyatının logaritması", "log"),
    "area": ("Konut büyüklüğü", "fit²"), "larea": ("Konut büyüklüğünün logaritması", "log"),
    "rooms": ("Oda sayısı", "adet"), "baths": ("Banyo sayısı", "adet"), "land": ("Arsa büyüklüğü", "fit²"),
}


# --- Adlar ----------------------------------------------------------------------------------------------

def display(case: Case, column: str) -> str:
    """Sütunun ekrandaki adı, birimiyle: "Eğitim (yıl)"; kendi verinde dosyadaki ad."""

    unit = case.units.get(column, "")
    return f"{case.name(column)} ({unit})" if unit else case.name(column)


def phrase(case: Case, column: str) -> str:
    """Cümle içindeki ad: alternatif örnekte küçük harfle ("aylık kazanç"), kendi verinde tırnak içinde."""

    if case.own:
        return f"“{md(case.name(column))}”"
    return dict(case.extra.get("phrases", {})).get(column) or tr_lower(case.name(column))


def capital(text: str) -> str:
    """İlk harf büyük (Türkçe i → İ, ı → I); tırnakla başlayan metin olduğu gibi kalır."""

    if not text:
        return text
    return {"i": "İ", "ı": "I"}.get(text[0], text[0].upper()) + text[1:]


def short_unit(case: Case, column: str) -> str:
    """Metindeki birim ("dolar", "yıl"); kendi verinde boş."""

    return dict(case.extra.get("short_units", {})).get(column, "")


def amount(case: Case, column: str, value: float, decimals: int) -> str:
    """Bir farkın büyüklüğü birimiyle: "60,21 dolar"; birimi bilinmiyorsa "60,21 birim"."""

    unit = short_unit(case, column)
    return f"{sayi(value, decimals)} {unit}" if unit else f"{sayi(value, decimals)} birim"


def digits_for(value: float, minimum: int) -> int:
    """Metindeki ondalık basamak: en az ``minimum``; mutlak değeri 10^(2 − minimum)'den küçük sayıda en az üç anlamlı
    basamak (ör. 0,0000123; ekrandaki ``topics.lab_ui.coefficient_number`` ile aynı kural), en çok 12."""

    if value != 0 and math.isfinite(value) and abs(value) < 10 ** (2 - minimum):
        return int(min(12, max(minimum, 2 - math.floor(math.log10(abs(value))))))
    return minimum


def series(case: Case, column: str) -> pd.Series:
    """Sütunun değerleri; kendi verinde türetilen log sütunu (``ln_...``) sonuçtan hesaplanır."""

    if column in case.data.columns:
        return case.data[column].astype(float)
    for source in case.data.columns:
        if log_column(case, source) == column:
            return np.log(case.data[source].astype(float))
    raise KeyError(column)


def negligible(case: Case, value: float, outcome: str, regressor: str) -> bool:
    """Eğim hesap hassasiyetinde sıfır mı: |β̂| · s_X ≤ 10⁻⁹ · s_Y (ilişkinin gücü yuvarlama gürültüsü düzeyinde).
    Küçük ama gerçek bir eğim (ör. TL cinsinden bir açıklayıcının 0,0000123'lük eğimi) sıfır sayılmaz."""

    if value == 0 or not math.isfinite(value):
        return True
    spread_y, spread_x = float(series(case, outcome).std()), float(series(case, regressor).std())
    return abs(value) * spread_x <= 1e-9 * max(spread_y, np.finfo(float).tiny)


def change(value: float, text: str, zero: bool) -> str:
    """Farkın yüklemi: "yaklaşık 60,214 dolar daha yüksektir"; fark sıfırsa "aynıdır" ("daha aynıdır" yazılmaz).
    ``text`` farkın mutlak değerinin yazımıdır."""

    if zero:
        return "aynıdır"
    return f"yaklaşık {text} daha {'yüksektir' if value > 0 else 'düşüktür'}"


def rough(log_change: float) -> str:
    """Log farkı büyükse (|Δ ln Y| > 0,1) 100 · β̂ yaklaşımının kaba olduğunu söyleyen cümle; tam dönüşüm
    100 · [exp(β̂) − 1] Konu 9'un konusudur (ör. β̂ = 1,36 için yaklaşım %136, tam değer %290)."""

    if not math.isfinite(log_change) or abs(log_change) <= 0.1:
        return ""
    return " Fark küçük olmadığı için 100 · β̂ yaklaşımı kabadır; tam dönüşüm Konu 9'da."


def step_words(case: Case, column: str) -> str:
    """Bir birimlik farkın yazımı: "eğitim süresi bir yıl daha uzun olan"; kendi verinde "“X” değeri bir birim daha
    yüksek olan"."""

    known = dict(case.extra.get("steps", {}))
    if column in known and not case.own:
        return known[column]
    return f"{phrase(case, column)} değeri bir birim daha yüksek olan"


def plural(case: Case) -> str:
    """Gözlem biriminin çoğul tamlayanı: "çalışanların", "konutların", "gözlemlerin"."""

    return str(case.extra.get("plural", "gözlemlerin"))


def outcome_words(case: Case) -> str:
    """Sonuç değişkeninin iyelik biçimi: "aylık kazancı"; kendi verinde "“Y” değeri"."""

    y = case.roles[SONUC]
    known = dict(case.extra.get("possessive", {}))
    if y in known and not case.own:
        return known[y]
    return f"{phrase(case, y)} değeri"


def listing(items: list[str]) -> str:
    """Türkçe sıralama: "a", "a ve b", "a, b ve c"."""

    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " ve " + items[-1]


def options(case: Case, columns: Iterable[str]) -> tuple[tuple[str, str], ...]:
    return tuple((column, display(case, column)) for column in columns)


def candidates(case: Case) -> tuple[str, ...]:
    """Açıklayıcı değişken seçenekleri (temel açıklayıcı önce). Kendi verinde yalnız sonuçla birlikte kullanılabilen
    sütunlar: ikisi de değişmeli (sabit bir değişkenle eğim tanımsızdır)."""

    if "regressors" in case.extra:
        return tuple(case.extra["regressors"])
    y = case.roles[SONUC]
    return tuple(column for column in dict.fromkeys((case.roles[ACIKLAYICI], *case.extras))
                 if column != y and usable_pair(case.data, y, column))


def levels(case: Case, column: str) -> tuple[float, ...]:
    """Bir sayısal değişkenin farklı değerleri, küçükten büyüğe."""

    return tuple(float(value) for value in np.sort(case.data[column].dropna().unique()))


def discrete(case: Case, column: str) -> bool:
    """Değişkenin en çok ``MAX_LEVELS`` farklı değeri var mı (düzey ortalamaları anlamlı mı)."""

    return case.data[column].nunique() <= MAX_LEVELS


def level_value(case: Case, column: str, value: float):
    """Koşul değeri koddaki türüyle: tam sayı değerli sütunda tam sayı, diğerlerinde ondalıklı sayı."""

    number = float(value)
    values = case.data[column].dropna().to_numpy(dtype=float)
    return int(number) if np.all(values == np.round(values)) and number.is_integer() else number


def level_text(value: float) -> str:
    """Değerin kısa yazımı, üslü gösterim olmadan: 12; 2,5; 1573,25."""

    number = float(value)
    if number.is_integer():
        return sayi(number, 0)
    text = f"{number:.10f}".rstrip("0").rstrip(".")
    return text.replace(".", ",").replace("-", "−")


def positive(case: Case, column: str) -> bool:
    """Bütün değerler pozitif mi (logaritma alınabilir mi)."""

    values = case.data[column].dropna()
    return len(values) > 0 and float(values.min()) > 0


def log_column(case: Case, column: str) -> str:
    """Değişkenin doğal logaritmasının sütun adı: alternatif örnekte verideki sütun (``lwage``), kendi verinde türetilen
    sütun (``ln_...``)."""

    known = dict(case.extra.get("logs", {}))
    if column in known:
        return known[column]
    return free_name(f"ln_{column}", set(case.data.columns))


def log_operations(case: Case, frame: str, columns: Iterable[str]) -> tuple[Operation, ...]:
    """Logaritması veride olmayan sütunlar için türetme işlemleri."""

    known = dict(case.extra.get("logs", {}))
    return tuple(Derive(frame, log_column(case, column), E.log(E.var(column)), f"ln({case.name(column)})")
                 for column in columns if column not in known)


def log_label(case: Case, column: str) -> str:
    return f"ln({case.name(column)})"


def reload(case: Case) -> tuple[Operation, ...]:
    """Veriyi önceki adımlarda türetilen sütunlar olmadan yeniden kuran işlemler (notlardaki "yeniden yüklenir")."""

    if case.own:
        read = case.load[0]
        assert isinstance(read, ReadFile)
        return (replace(read, comment=f"{read.comment}: önceki adımlarda türetilen sütunlar olmadan yeniden okunur"),)
    first, *rest = case.load
    assert isinstance(first, LoadWooldridge)
    return (replace(first, comment=f"{first.comment}; önceki adımlarda türetilen sütunlar olmadan yeniden yüklenir"),
            *rest)


SECOND = "ikinci"
"""Kendi verinde notlarda ikinci bir veri setiyle yapılan adımların çerçevesi."""


def second(case: Case) -> Case:
    """Notlarda ikinci bir veri setiyle yapılan adımların verisi. Alternatif örnekte ``case.extra["house"]``; kendi
    verinde aynı dosya, ayrı bir çerçeveye okunur: bu adımların türettiği sütunlar ana çerçeveyi değiştirmez (notlarda
    da ikinci veri ayrı bir çerçevedir)."""

    if "house" in case.extra:
        return case.extra["house"]
    read = case.load[0]
    assert isinstance(read, ReadFile)
    return replace(case, frame=SECOND, load=(replace(read, frame=SECOND,
                                                      comment=f"{read.comment}: ikinci modeller için ayrı bir çerçeve"),))


def labels_of(case: Case) -> dict[str, str]:
    """``LabSpec.labels`` için sütun adları, birimleriyle."""

    return {column: display(case, column) for column in case.labels}


# --- Sayısal hassasiyet ------------------------------------------------------------------------------------

def noise_decimals(case: Case, column: str | None = None, regressors: Iterable[str] = ()) -> int:
    """Artıkların toplamının gösterim basamağı: hesap hassasiyeti n · ölçek düzeyindedir; ölçek max|Y| ile tahmin
    edilen değerin terimlerinin büyüklüğünün (|β̂₀| + Σ|β̂ⱼXᵢⱼ|) en büyüğüdür. Büyük açıklayıcı değerleri birbirini
    götüren terimlerle çarpılınca gürültü |Y|'den çok büyük olabilir. İki yazılımın yuvarlama gürültüsü bu basamakta
    görünmez."""

    y = column or case.roles[SONUC]
    values = case.data[y].astype(float)
    scale = float(values.abs().max())
    regressors = tuple(regressors)
    if regressors:
        design = np.column_stack([np.ones(len(values)), case.data[list(regressors)].to_numpy(dtype=float)])
        coefficients, *_ = np.linalg.lstsq(design, values.to_numpy(), rcond=None)
        scale = max(scale, float((np.abs(design) @ np.abs(coefficients)).max()))
    exponent = math.log10(max(scale * len(case.data), 1.0))
    return int(max(0, min(10, 12 - math.ceil(exponent))))


def exact_multi(case: Case, outcome: str, regressors) -> bool:
    """Çoklu modelin uyumu (neredeyse) tam mı (R² ≈ 1): standart hata sıfıra çok yakındır; t, p ve güven aralığı
    yuvarlama hatasına duyarlıdır (``core.labs.ornek.exact_fit``'in çoklu modeldeki karşılığı)."""

    if outcome not in case.data.columns:
        return False
    data = case.data[[outcome, *regressors]].astype(float)
    design = np.column_stack([np.ones(len(data)), data[list(regressors)].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, data[outcome].to_numpy(), rcond=None)
    residual = data[outcome].to_numpy() - design @ coefficients
    total = float(((data[outcome] - data[outcome].mean()) ** 2).sum())
    return float((residual ** 2).sum()) <= EXACT_FIT * total


def coefficients(case: Case, outcome: str, regressors: Iterable[str]) -> dict[str, float]:
    """EKK katsayıları (metinlerin ve gösterim basamaklarının hesabı için; uygulamanın hesabı işlemlerle yapılır)."""

    regressors = tuple(regressors)
    values = series(case, outcome).to_numpy()
    design = np.column_stack([np.ones(len(values)), *(series(case, name).to_numpy() for name in regressors)])
    estimates, *_ = np.linalg.lstsq(design, values, rcond=None)
    return dict(zip(("Intercept", *regressors), (float(value) for value in estimates)))


EXACT_MULTI_NOTE = (" Model veriye neredeyse tam uyuyor (R² ≈ 1). Böyle bir veride standart hatalar sıfıra çok yakındır; t, "
                    "p-değeri, güven aralığı ve F yuvarlama hatasına duyarlıdır ve indirilen kodda karşılaştırılmaz.")
"""``core.labs.ornek.EXACT_FIT_NOTE``'in çoklu modeldeki karşılığı (noktalar bir doğrunun değil düzlemin üzerindedir)."""


# --- Sayı denetimleri --------------------------------------------------------------------------------------

def _nice_step(span: float, points: int) -> float:
    """Kaydırıcı adımı: 10'un kuvveti; aralık en çok ``points`` adıma bölünür."""

    span = abs(span) or 1.0
    return 10.0 ** math.ceil(math.log10(span / points))


def number_control(key: str, label: str, value: float, low: float, high: float, help: str = "",
                   step: float | None = None, points: int = 1000) -> NumberChoice:
    """Verinin ölçeğine uyan sayı denetimi: aralık ve varsayılan değer adımın katına yuvarlanır. Adım 1 ya da daha
    büyükse tam sayı kaydırıcısıdır; değilse basamak adımdan gelir."""

    low, high = float(min(low, value)), float(max(high, value))
    step = float(step) if step is not None else _nice_step(high - low, points)
    minimum = math.floor(round(low / step, 9)) * step
    maximum = math.ceil(round(high / step, 9)) * step
    if maximum <= minimum:
        maximum = minimum + step
    default = min(max(round(value / step) * step, minimum), maximum)
    if step >= 1 and float(step).is_integer():
        return NumberChoice(key, label, int(minimum), int(maximum), int(default), int(step), help=help, integer=True,
                            decimals=0)
    decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    return NumberChoice(key, label, round(minimum, decimals), round(maximum, decimals), round(default, decimals),
                        round(step, decimals), help=help, decimals=decimals)


def gap_control(case: Case, key: str, label: str, help: str) -> NumberChoice:
    """Açıklayıcıdaki bir farkın denetimi: varsayılan yaklaşık bir standart sapma, en küçük değer bir birim (tam sayı
    veride 1, kesirli veride kaydırıcı adımı), en büyük değer verideki aralık (en çok altı standart sapma)."""

    values = case.data[case.roles[ACIKLAYICI]].astype(float)
    spread, width = float(values.std()), float(values.max() - values.min())
    integral = bool(np.all(values == np.round(values)))
    default = nice(spread, 1) or 1.0
    if integral:
        default = max(1.0, float(round(default)))
    high = max(min(6 * spread, width), default)
    control = number_control(key, label, default, 1.0 if integral else 0.0, high, help=help,
                             step=1.0 if integral else None)
    if control.minimum <= 0:  # sıfır fark iki gözlemi aynı yapar
        control = replace(control, minimum=control.step)
    return control


def nice(value: float, digits: int = 3) -> float:
    """Değerin ``digits`` anlamlı basamağa yuvarlanmışı (ör. 60,2143 → 60,2; 0,012345 → 0,0123)."""

    if value == 0 or not math.isfinite(value):
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, digits - 1 - exponent)


def shown_decimals(value: float, digits: int = 3) -> int:
    """``nice`` ile yuvarlanmış değerin yazımındaki ondalık basamak."""

    if value == 0 or not math.isfinite(value):
        return 0
    return max(0, digits - 1 - math.floor(math.log10(abs(value))))


# --- Alternatif örneklerin verisi -------------------------------------------------------------------------

def _labels(table: dict[str, tuple[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    return {name: label for name, (label, _) in table.items()}, {name: unit for name, (_, unit) in table.items()}


def wage2_case(regressors: tuple[str, ...], extras: tuple[str, ...] = (), **extra) -> Case:
    """WAGE2: sonuç aylık kazanç, temel açıklayıcı eğitim; ``regressors`` seçeneklerdir."""

    data = W.load(WAGE2)
    labels, units = _labels(WAGE2_LABELS)
    settings = {
        "phrases": WAGE2_PHRASES, "steps": WAGE2_STEPS, "short_units": WAGE2_UNITS, "plural": "çalışanların",
        "possessive": {"wage": "aylık kazancı"}, "data_name": "WAGE2", "regressors": regressors, "logs": {"wage": "lwage"},
        "rounded_coefficients": True,
    }
    settings.update(extra)
    return Case(
        source="alternatif",
        load=(LoadWooldridge(WAGE2, "WAGE2 veri seti (Wooldridge, 2020): 935 erkek çalışan, 1980"),),
        frame=WAGE2,
        data=data,
        roles={SONUC: "wage", ACIKLAYICI: "educ"},
        labels=labels,
        extras=extras or tuple(name for name in regressors if name != "educ"),
        units=units,
        unit="çalışan",
        extra=settings,
    )


def house_frame() -> pd.DataFrame:
    data = W.load(KIELMC)
    return data[data["year"] == HOUSE_YEAR][list(HOUSE_COLUMNS)].reset_index(drop=True)


def house_case(regressors: tuple[str, ...] = ("area", "rooms", "baths", "land"), **extra) -> Case:
    """KIELMC'nin 1978 satışları: sonuç satış fiyatı, temel açıklayıcı konut büyüklüğü."""

    labels, units = _labels(HOUSE_LABELS)
    settings = {
        "phrases": HOUSE_PHRASES, "steps": HOUSE_STEPS, "short_units": HOUSE_UNITS, "plural": "konutların",
        "possessive": {"price": "satış fiyatı"}, "data_name": "KIELMC (1978)", "regressors": regressors,
        "logs": {"price": "lprice", "area": "larea"},
    }
    settings.update(extra)
    return Case(
        source="alternatif",
        load=(
            LoadWooldridge(KIELMC, "KIELMC veri seti (Wooldridge, 2020): 1978 ve 1981'de satılan konutlar"),
            TakeRows(KONUT, KIELMC, f"Yalnız {HOUSE_YEAR}'de satılan 179 konut: tek dönemlik yatay kesit (fiyat "
                                    f"{HOUSE_YEAR} dolarıyla)", where=("year", HOUSE_YEAR), columns=HOUSE_COLUMNS),
        ),
        frame=KONUT,
        data=house_frame(),
        roles={SONUC: "price", ACIKLAYICI: "area"},
        labels=labels,
        extras=tuple(name for name in regressors if name != "area"),
        units=units,
        unit="konut",
        extra=settings,
    )


def program_frame() -> pd.DataFrame:
    return pd.DataFrame(list(PROGRAM_ROWS), columns=list(PROGRAM_COLUMNS))


def program_case() -> Case:
    """Kurgusal iş arama programı (Konu 2): sonuç program sonrası yıllık kazanç, sıfır–bir değişkenler program, kadın,
    evli."""

    labels = {name: label.split(" (")[0] for name, label in PROGRAM_LABELS.items()}
    units = {name: label.split(" (")[1].rstrip(")") for name, label in PROGRAM_LABELS.items() if " (" in label}
    return Case(
        source="alternatif",
        load=(InlineData(PROGRAM, PROGRAM_COLUMNS, PROGRAM_ROWS,
                         "Kurgusal iş arama programı (Konu 2): 200 kişi, kurayla 100 program ve 100 kontrol"),),
        frame=PROGRAM,
        data=program_frame(),
        roles={SONUC: "kazanc", GOSTERGE: "program"},
        labels=labels,
        extras=("kadin", "evli"),
        units=units,
        unit="kişi",
        extra={"plural": "kişilerin", "short_units": {"kazanc": "bin TL"}},
    )


# --- Kendi verin ------------------------------------------------------------------------------------------

def validate(case: Case) -> None:
    """Sonuç ve açıklayıcı farklı sütunlar; seçilen her sayısal sütunda en az iki farklı değer; en büyük model (temel
    açıklayıcı ve bütün ek değişkenler, k açıklayıcı) için en az k + 2 gözlem (artık serbestlik derecesi en az 1); ek
    değişkenler temel açıklayıcıyla birlikte tam doğrusal bağlantı kurmaz (çoklu modeller kurulamazdı)."""

    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    if y == x:
        raise K.UploadError("Sonuç ve açıklayıcı değişken için farklı sütunlar seçin.")
    if y in case.extras:
        raise K.UploadError("Sonuç değişkeni ek değişkenler arasında seçilemez.")
    for column in (y, x, *case.extras):
        values = case.data[column]
        if values.nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    if not usable_pair(case.data, y, x):
        raise K.UploadError("Sonuç ve açıklayıcı değişkenin birlikte en az üç gözlemi olmalı.")
    columns = candidates(case)
    if len(case.data) < len(columns) + 2:
        raise K.UploadError(f"Temel açıklayıcı ve ek değişkenlerle ({len(columns)} açıklayıcı) model için en az "
                            f"{len(columns) + 2} gözlem gerekir; analizde {len(case.data)} gözlem var. Daha az ek "
                            "değişken seçin.")
    if len(columns) > 1:
        design = np.column_stack([np.ones(len(case.data)), case.data[list(columns)].to_numpy(dtype=float)])
        scaled = design / np.linalg.norm(design, axis=0)
        if np.linalg.matrix_rank(scaled, tol=1e-10) < design.shape[1]:
            raise K.UploadError("Seçilen açıklayıcılar arasında tam doğrusal bağlantı var (biri ötekilerin doğrusal "
                                "birleşimi; ör. aynı değişkenin iki ayrı birimi). Bu sütunlardan birini ek "
                                "değişkenlerden çıkarın.")


def roles(steps: tuple[int, ...], extra: tuple[Role, ...] = ()) -> tuple[Role, ...]:
    return (
        Role(SONUC, "Sonuç (bağımlı) değişken", "sayisal", True, steps,
             "Açıklanan sayısal değişken (ör. kazanç, fiyat, sınav puanı)."),
        Role(ACIKLAYICI, "Temel açıklayıcı değişken", "sayisal", True, steps,
             "Araştırma sorusundaki açıklayıcı değişken (ör. eğitim yılı, konut büyüklüğü)."),
        *extra,
    )


ROW_RULE = ("Seçilen bütün sütunlarda (sonuç, açıklayıcı ve ek değişkenler) boş hücresi olan satırlar analizden "
            "çıkarılır: bütün modeller aynı gözlemlerle kurulur.")


def custom_lab(build: Callable[[Case], LabSpec], sample: Callable[[], pd.DataFrame], intro: str,
               role_set: tuple[Role, ...], extra_help: str, **options) -> CustomLab:
    return CustomLab(
        roles=role_set,
        build=build,
        sample=sample,
        intro=intro,
        min_rows=5,
        extra_columns=True,
        extra_use="sayisal",
        extra_label=f"Ek sayısal değişkenler (isteğe bağlı, en çok {MAX_EXTRA})",
        extra_help=extra_help,
        max_extra=MAX_EXTRA,
        extra_required=True,
        validate=options.pop("validate", validate),
        **options,
    )


def sample_people() -> pd.DataFrame:
    """Örnek dosya: kurgusal iş arama programının 200 kişisi (işsiz kalanların kazancı 0); öğrenci verisi değil."""

    frame = program_frame()
    return pd.DataFrame({
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Eğitim yılı": frame["egitim"],
        "Yaş": frame["yas"],
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
        "Grup": np.where(frame["program"] == 1, "Program", "Kontrol"),
    })


def sample_employed() -> pd.DataFrame:
    """Örnek dosya: kurgusal iş arama programının program sonrası çalışan kişileri (kazanç > 0); öğrenci verisi
    değil."""

    frame = program_frame()
    frame = frame[frame["issiz"] == 0].reset_index(drop=True)
    return pd.DataFrame({
        "Yıllık kazanç (bin TL)": frame["kazanc"],
        "Eğitim yılı": frame["egitim"],
        "Yaş": frame["yas"],
        "Önceki yıllık kazanç (bin TL)": frame["onceki_kazanc"],
    })
