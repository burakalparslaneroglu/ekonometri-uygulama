"""Wooldridge (2020) veri setleri: yükleme ve öğrenciye gösterilen değişken adları.

Veri depoya kopyalanmaz. Uygulama ve üretilen Python kodu veriyi ``wooldridge`` Python paketinden
(T. Haruyama, 0.5.0), üretilen R kodu ``wooldridge`` R paketinden (J. M. Shea) okur. İki paket kitabın
7. baskısının verisini verir; aynı veriyi verdikleri testle denetlenir (satır, sütun, sütun toplamları).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd


@dataclass(frozen=True)
class Variable:
    label: str
    unit: str = ""

    @property
    def text(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    unit: str
    structure: str
    generation: str
    variables: dict[str, Variable]


_INDICATOR = "0/1 gösterge"

DATASETS: dict[str, Dataset] = {
    "wage1": Dataset(
        "wage1", "WAGE1: ücret ve çalışan özellikleri", "çalışan", "yatay kesit verisi", "gözlemsel",
        {
            "wage": Variable("Saatlik ücret", "ABD doları/saat"),
            "educ": Variable("Eğitim", "yıl"),
            "exper": Variable("Potansiyel deneyim", "yıl"),
            "tenure": Variable("Mevcut işverendeki kıdem", "yıl"),
            "nonwhite": Variable("Beyaz olmayan", _INDICATOR),
            "female": Variable("Kadın", _INDICATOR),
            "married": Variable("Evli", _INDICATOR),
            "numdep": Variable("Bakmakla yükümlü kişi sayısı", "kişi"),
            "smsa": Variable("Büyükşehir alanında yaşıyor", _INDICATOR),
            "northcen": Variable("Kuzey merkez bölgesi", _INDICATOR),
            "south": Variable("Güney bölgesi", _INDICATOR),
            "west": Variable("Batı bölgesi", _INDICATOR),
            "construc": Variable("İnşaat sektörü", _INDICATOR),
            "ndurman": Variable("Dayanıksız mal imalatı", _INDICATOR),
            "trcommpu": Variable("Ulaştırma, iletişim, kamu hizmetleri", _INDICATOR),
            "trade": Variable("Toptan ve perakende ticaret", _INDICATOR),
            "services": Variable("Hizmetler sektörü", _INDICATOR),
            "profserv": Variable("Profesyonel hizmetler sektörü", _INDICATOR),
            "profocc": Variable("Profesyonel meslek", _INDICATOR),
            "clerocc": Variable("Büro mesleği", _INDICATOR),
            "servocc": Variable("Hizmet mesleği", _INDICATOR),
            "lwage": Variable("Saatlik ücretin logaritması", "log"),
            "expersq": Variable("Deneyimin karesi", "yıl²"),
            "tenursq": Variable("Kıdemin karesi", "yıl²"),
        },
    ),
    "hprice1": Dataset(
        "hprice1", "HPRICE1: konut fiyatları ve özellikleri", "konut", "yatay kesit verisi", "gözlemsel",
        {
            "price": Variable("Konut fiyatı", "bin ABD doları"),
            "assess": Variable("Vergi değeri", "bin ABD doları"),
            "bdrms": Variable("Yatak odası sayısı", "adet"),
            "lotsize": Variable("Arsa büyüklüğü", "fit²"),
            "sqrft": Variable("Konut büyüklüğü", "fit²"),
            "colonial": Variable("Kolonyal mimari tarz", _INDICATOR),
            "lprice": Variable("Fiyatın logaritması", "log"),
            "lassess": Variable("Vergi değerinin logaritması", "log"),
            "llotsize": Variable("Arsa büyüklüğünün logaritması", "log"),
            "lsqrft": Variable("Konut büyüklüğünün logaritması", "log"),
        },
    ),
    "phillips": Dataset(
        "phillips", "PHILLIPS: enflasyon ve işsizlik", "yıl", "zaman serisi verisi", "gözlemsel",
        {
            "year": Variable("Yıl"),
            "unem": Variable("İşsizlik oranı", "%"),
            "inf": Variable("Enflasyon oranı", "%"),
            "inf_1": Variable("Bir önceki yılın enflasyon oranı", "%"),
            "unem_1": Variable("Bir önceki yılın işsizlik oranı", "%"),
            "cinf": Variable("Enflasyon oranındaki değişim", "yüzde puan"),
            "cunem": Variable("İşsizlik oranındaki değişim", "yüzde puan"),
        },
    ),
    "cps78_85": Dataset(
        "cps78_85", "CPS78_85: 1978 ve 1985 çalışan örneklemleri", "çalışan", "havuzlanmış yatay kesit verisi",
        "gözlemsel",
        {
            "educ": Variable("Eğitim", "yıl"),
            "south": Variable("Güneyde yaşıyor", _INDICATOR),
            "nonwhite": Variable("Beyaz olmayan", _INDICATOR),
            "female": Variable("Kadın", _INDICATOR),
            "married": Variable("Evli", _INDICATOR),
            "exper": Variable("Potansiyel deneyim", "yıl"),
            "expersq": Variable("Deneyimin karesi", "yıl²"),
            "union": Variable("Sendika üyesi", _INDICATOR),
            "lwage": Variable("Saatlik ücretin logaritması", "log"),
            "wage": Variable("Saatlik ücret", "ABD doları/saat, cari fiyatlarla"),
            "age": Variable("Yaş", "yıl"),
            "year": Variable("Yıl", "78 ya da 85"),
            "y85": Variable("1985 yılı", _INDICATOR),
            "y85fem": Variable("1985 × kadın", _INDICATOR),
            "y85educ": Variable("1985 × eğitim", "yıl"),
            "y85union": Variable("1985 × sendika", _INDICATOR),
        },
    ),
    "wagepan": Dataset(
        "wagepan", "WAGEPAN: 1980–1987 çalışan paneli", "kişi–yıl", "panel veri", "gözlemsel",
        {
            "nr": Variable("Kişi kimliği"),
            "year": Variable("Yıl"),
            "agric": Variable("Tarım sektörü", _INDICATOR),
            "black": Variable("Siyah", _INDICATOR),
            "bus": Variable("İş hizmetleri sektörü", _INDICATOR),
            "construc": Variable("İnşaat sektörü", _INDICATOR),
            "ent": Variable("Eğlence sektörü", _INDICATOR),
            "exper": Variable("Deneyim", "yıl"),
            "fin": Variable("Finans sektörü", _INDICATOR),
            "hisp": Variable("Hispanik", _INDICATOR),
            "poorhlth": Variable("Sağlık sorunu var", _INDICATOR),
            "hours": Variable("Yıllık çalışma saati", "saat"),
            "manuf": Variable("İmalat sektörü", _INDICATOR),
            "married": Variable("Evli", _INDICATOR),
            "min": Variable("Madencilik sektörü", _INDICATOR),
            "nrthcen": Variable("Kuzey merkez bölgesi", _INDICATOR),
            "nrtheast": Variable("Kuzeydoğu bölgesi", _INDICATOR),
            **{f"occ{number}": Variable(f"Meslek grubu {number}", _INDICATOR) for number in range(1, 10)},
            "per": Variable("Kişisel hizmetler sektörü", _INDICATOR),
            "pro": Variable("Profesyonel hizmetler sektörü", _INDICATOR),
            "pub": Variable("Kamu yönetimi", _INDICATOR),
            "rur": Variable("Kırsal alanda yaşıyor", _INDICATOR),
            "south": Variable("Güney bölgesi", _INDICATOR),
            "educ": Variable("Eğitim", "yıl"),
            "tra": Variable("Ulaştırma sektörü", _INDICATOR),
            "trad": Variable("Ticaret sektörü", _INDICATOR),
            "union": Variable("Sendika üyesi", _INDICATOR),
            "lwage": Variable("Saatlik ücretin logaritması", "log"),
            **{f"d8{number}": Variable(f"198{number} yılı", _INDICATOR) for number in range(1, 8)},
            "expersq": Variable("Deneyimin karesi", "yıl²"),
        },
    ),
    "jtrain2": Dataset(
        "jtrain2", "JTRAIN2: rastgele atamalı iş eğitimi programı", "kişi", "yatay kesit verisi", "deneysel",
        {
            "train": Variable("Eğitim programına atandı", _INDICATOR),
            "age": Variable("1977'deki yaş", "yıl"),
            "educ": Variable("Öğrenim süresi", "yıl"),
            "black": Variable("Siyah", _INDICATOR),
            "hisp": Variable("Hispanik", _INDICATOR),
            "married": Variable("Evli", _INDICATOR),
            "nodegree": Variable("Lise diploması yok", _INDICATOR),
            "mosinex": Variable("Deneydeki ay sayısı (1/78 öncesi)", "ay"),
            "re74": Variable("1974 reel kazancı", "bin ABD doları"),
            "re75": Variable("1975 reel kazancı", "bin ABD doları"),
            "re78": Variable("1978 reel kazancı", "bin ABD doları"),
            "unem74": Variable("1974'ün tamamında işsiz", _INDICATOR),
            "unem75": Variable("1975'in tamamında işsiz", _INDICATOR),
            "unem78": Variable("1978'in tamamında işsiz", _INDICATOR),
            "lre74": Variable("1974 kazancının logaritması (sıfırsa 0)", "log"),
            "lre75": Variable("1975 kazancının logaritması (sıfırsa 0)", "log"),
            "lre78": Variable("1978 kazancının logaritması (sıfırsa 0)", "log"),
            "agesq": Variable("Yaşın karesi", "yıl²"),
            "mostrn": Variable("Eğitim programında geçen ay", "ay"),
        },
    ),
}


@lru_cache(maxsize=None)
def _raw(name: str) -> pd.DataFrame:
    import wooldridge

    if name not in DATASETS:
        raise KeyError(f"Tanımsız veri seti: {name}")
    return wooldridge.data(name)


def load(name: str, columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Veri setinin kopyası (işlemler çerçeveyi değiştirebilir; önbellekteki özgün veri değişmez)."""

    frame = _raw(name)
    if columns:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise KeyError(f"{name}: sütun yok: {', '.join(missing)}")
        frame = frame[list(columns)]
    return frame.copy()


def variable(dataset: str, name: str) -> Variable:
    return DATASETS[dataset].variables[name]


def labels(*datasets: str) -> tuple[tuple[str, str], ...]:
    """``LabSpec.labels`` için (değişken, "Etiket (birim)") çiftleri; aynı ad birden çok veri setinde geçerse ilk
    veri setinin etiketi kullanılır."""

    found: dict[str, str] = {}
    for dataset in datasets:
        for name, item in DATASETS[dataset].variables.items():
            found.setdefault(name, item.text)
    return tuple(found.items())


def options(dataset: str, names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Denetim seçenekleri: (değişken, "Etiket (birim)")."""

    return tuple((name, DATASETS[dataset].variables[name].text) for name in names)
