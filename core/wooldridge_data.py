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
    # Uygulama sekmesinin alternatif örnekleri (Konu 0–2; ``core.labs.ornek_konuNN``)
    "wage2": Dataset(
        "wage2", "WAGE2: aylık kazanç ve çalışan özellikleri (935 erkek çalışan, 1980)", "çalışan",
        "yatay kesit verisi", "gözlemsel",
        {
            "wage": Variable("Aylık kazanç", "ABD doları/ay"),
            "hours": Variable("Ortalama haftalık çalışma saati", "saat"),
            "IQ": Variable("IQ puanı", "puan"),
            "KWW": Variable("İş dünyası bilgisi testi puanı", "puan"),
            "educ": Variable("Eğitim", "yıl"),
            "exper": Variable("İş deneyimi", "yıl"),
            "tenure": Variable("Mevcut işverendeki kıdem", "yıl"),
            "age": Variable("Yaş", "yıl"),
            "married": Variable("Evli", _INDICATOR),
            "black": Variable("Siyah", _INDICATOR),
            "south": Variable("Güneyde yaşıyor", _INDICATOR),
            "urban": Variable("Büyükşehir alanında yaşıyor", _INDICATOR),
            "sibs": Variable("Kardeş sayısı", "kişi"),
            "brthord": Variable("Doğum sırası", "sıra"),
            "meduc": Variable("Annenin eğitimi", "yıl"),
            "feduc": Variable("Babanın eğitimi", "yıl"),
            "lwage": Variable("Aylık kazancın logaritması", "log"),
        },
    ),
    "okun": Dataset(
        "okun", "OKUN: ABD'de büyüme ve işsizlik, 1959–2005", "yıl", "zaman serisi verisi", "gözlemsel",
        {
            "year": Variable("Yıl"),
            "pcrgdp": Variable("Reel GSYH büyümesi", "%"),
            "unem": Variable("İşsizlik oranı", "%"),
            "cunem": Variable("İşsizlik oranındaki değişim", "yüzde puan"),
        },
    ),
    "kielmc": Dataset(
        "kielmc", "KIELMC: çöp yakma tesisi ve konut fiyatları (1978 ve 1981)", "konut",
        "havuzlanmış yatay kesit verisi", "gözlemsel",
        {
            "year": Variable("Satış yılı", "1978 ya da 1981"),
            "age": Variable("Konutun yaşı", "yıl"),
            "agesq": Variable("Konut yaşının karesi", "yıl²"),
            "nbh": Variable("Mahalle kodu", "0–6"),
            "cbd": Variable("Merkezi iş bölgesine uzaklık", "fit"),
            "intst": Variable("Otoyola uzaklık", "fit"),
            "lintst": Variable("Otoyola uzaklığın logaritması", "log"),
            "price": Variable("Satış fiyatı", "ABD doları, cari fiyatlarla"),
            "rooms": Variable("Oda sayısı", "adet"),
            "area": Variable("Konut büyüklüğü", "fit²"),
            "land": Variable("Arsa büyüklüğü", "fit²"),
            "baths": Variable("Banyo sayısı", "adet"),
            "dist": Variable("Çöp yakma tesisine uzaklık", "fit"),
            "ldist": Variable("Tesise uzaklığın logaritması", "log"),
            "wind": Variable("Rüzgârın tesisten konuta estiği zaman", "%"),
            "lprice": Variable("Satış fiyatının logaritması", "log"),
            "y81": Variable("1981 yılı", _INDICATOR),
            "larea": Variable("Konut büyüklüğünün logaritması", "log"),
            "lland": Variable("Arsa büyüklüğünün logaritması", "log"),
            "y81ldist": Variable("1981 × tesise uzaklığın logaritması", "log"),
            "lintstsq": Variable("Otoyola uzaklığın logaritmasının karesi", "log²"),
            "nearinc": Variable("Tesise 3 mil ya da daha yakın", _INDICATOR),
            "y81nrinc": Variable("1981 × tesise yakın", _INDICATOR),
            "rprice": Variable("Satış fiyatı, 1978 fiyatlarıyla", "ABD doları"),
            "lrprice": Variable("1978 fiyatlarıyla satış fiyatının logaritması", "log"),
        },
    ),
    "crime4": Dataset(
        "crime4", "CRIME4: Kuzey Karolina ilçelerinde suç, 1981–1987", "ilçe–yıl", "panel veri", "gözlemsel",
        {
            "county": Variable("İlçe kimliği"),
            "year": Variable("Yıl", "81–87"),
            "crmrte": Variable("Kişi başına suç sayısı", "suç/kişi"),
            "prbarr": Variable("Tutuklama oranı (tutuklama / suç)", "oran"),
            "prbconv": Variable("Mahkûmiyet oranı (mahkûmiyet / tutuklama)", "oran"),
            "prbpris": Variable("Hapis cezası oranı (hapis / mahkûmiyet)", "oran"),
            "avgsen": Variable("Ortalama ceza süresi", "gün"),
            "polpc": Variable("Kişi başına polis sayısı", "polis/kişi"),
            "density": Variable("Nüfus yoğunluğu"),
            "taxpc": Variable("Kişi başına vergi geliri", "ABD doları/kişi"),
            "west": Variable("Batı Kuzey Karolina", _INDICATOR),
            "central": Variable("Orta Kuzey Karolina", _INDICATOR),
            "urban": Variable("Büyükşehir alanında", _INDICATOR),
            "pctmin80": Variable("Azınlık nüfus payı, 1980", "%"),
            "wcon": Variable("Haftalık ücret: inşaat", "ABD doları/hafta"),
            "wtuc": Variable("Haftalık ücret: ulaştırma, kamu hizmetleri, iletişim", "ABD doları/hafta"),
            "wtrd": Variable("Haftalık ücret: toptan ve perakende ticaret", "ABD doları/hafta"),
            "wfir": Variable("Haftalık ücret: finans, sigorta, gayrimenkul", "ABD doları/hafta"),
            "wser": Variable("Haftalık ücret: hizmetler", "ABD doları/hafta"),
            "wmfg": Variable("Haftalık ücret: imalat", "ABD doları/hafta"),
            "wfed": Variable("Haftalık ücret: federal çalışanlar", "ABD doları/hafta"),
            "wsta": Variable("Haftalık ücret: eyalet çalışanları", "ABD doları/hafta"),
            "wloc": Variable("Haftalık ücret: yerel yönetim çalışanları", "ABD doları/hafta"),
            "mix": Variable("Suç bileşimi: yüz yüze / diğer suçlar", "oran"),
            "pctymle": Variable("Genç erkeklerin nüfus payı", "pay"),
            **{f"d8{number}": Variable(f"198{number} yılı", _INDICATOR) for number in range(2, 8)},
            "lcrmrte": Variable("Kişi başına suçun logaritması", "log"),
            "lprbarr": Variable("Tutuklama oranının logaritması", "log"),
            "lprbconv": Variable("Mahkûmiyet oranının logaritması", "log"),
            "lprbpris": Variable("Hapis cezası oranının logaritması", "log"),
            "lavgsen": Variable("Ortalama ceza süresinin logaritması", "log"),
            "lpolpc": Variable("Kişi başına polisin logaritması", "log"),
            "ldensity": Variable("Nüfus yoğunluğunun logaritması", "log"),
            "ltaxpc": Variable("Kişi başına vergi gelirinin logaritması", "log"),
            "lwcon": Variable("İnşaat ücretinin logaritması", "log"),
            "lwtuc": Variable("Ulaştırma ve kamu hizmetleri ücretinin logaritması", "log"),
            "lwtrd": Variable("Ticaret ücretinin logaritması", "log"),
            "lwfir": Variable("Finans ücretinin logaritması", "log"),
            "lwser": Variable("Hizmetler ücretinin logaritması", "log"),
            "lwmfg": Variable("İmalat ücretinin logaritması", "log"),
            "lwfed": Variable("Federal çalışan ücretinin logaritması", "log"),
            "lwsta": Variable("Eyalet çalışanı ücretinin logaritması", "log"),
            "lwloc": Variable("Yerel yönetim ücretinin logaritması", "log"),
            "lmix": Variable("Suç bileşiminin logaritması", "log"),
            "lpctymle": Variable("Genç erkek payının logaritması", "log"),
            "lpctmin": Variable("Azınlık payının logaritması", "log"),
            "clcrmrte": Variable("Log suç oranında yıllık değişim", "log farkı"),
            "clprbarr": Variable("Log tutuklama oranında yıllık değişim", "log farkı"),
            "clprbcon": Variable("Log mahkûmiyet oranında yıllık değişim", "log farkı"),
            "clprbpri": Variable("Log hapis oranında yıllık değişim", "log farkı"),
            "clavgsen": Variable("Log ceza süresinde yıllık değişim", "log farkı"),
            "clpolpc": Variable("Log kişi başına polisteki yıllık değişim", "log farkı"),
            "cltaxpc": Variable("Log kişi başına vergide yıllık değişim", "log farkı"),
            "clmix": Variable("Log suç bileşiminde yıllık değişim", "log farkı"),
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
