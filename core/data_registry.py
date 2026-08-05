"""Wooldridge veri kataloğu ve kontrollü değişken eşleşmeleri."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import wooldridge as woo


@dataclass(frozen=True)
class VariableMetadata:
    """Bir değişkenin öğrenciye gösterilecek açıklamasını tutar."""

    name: str
    label: str
    description: str
    unit: str


@dataclass(frozen=True)
class DatasetMetadata:
    """Bir Wooldridge veri setinin katalog bilgisini tutar."""

    key: str
    title: str
    source: str
    description: str
    observation_unit: str
    variables: dict[str, VariableMetadata]
    allowed_pairs: dict[str, tuple[str, ...]]
    data_structure: str = "yatay kesit verisi"
    time_variable: str | None = None
    time_coverage: str | None = None
    frequency: str | None = None
    identifier_variable: str | None = None
    repeated_units: bool = False
    collection_method: str = "gözlemsel"
    classification_reason: str = ""


DATASETS: dict[str, DatasetMetadata] = {
    "wage1": DatasetMetadata(
        key="wage1",
        title="WAGE1 — Ücret ve bireysel özellikler",
        source="Wooldridge, J. M. (2020), Introductory Econometrics: A Modern Approach, WAGE1.",
        description="ABD'de çalışan bireylere ait yatay kesit verisi. Ücret ile eğitim ve iş deneyimi gibi özelliklerin birlikte değişimini incelemek için kullanılır.",
        observation_unit="Çalışan",
        variables={
            "wage": VariableMetadata("wage", "Saatlik ücret", "Bireyin saatlik ücretidir.", "ABD doları/saat"),
            "educ": VariableMetadata("educ", "Eğitim", "Tamamlanan eğitim yılıdır.", "yıl"),
            "exper": VariableMetadata("exper", "İş deneyimi", "İş piyasasındaki potansiyel deneyim yılıdır.", "yıl"),
            "tenure": VariableMetadata("tenure", "Mevcut işyerindeki kıdem", "Mevcut işyerinde çalışma süresidir.", "yıl"),
        },
        allowed_pairs={"wage": ("educ", "exper", "tenure")},
        data_structure="yatay kesit verisi",
        time_coverage="Tek dönem bağlamı",
        frequency="Uygulanmaz",
        collection_method="gözlemsel",
        classification_reason="Her satır farklı bir çalışanı temsil eder; aynı çalışan zaman içinde tekrar izlenmez.",
    ),
    "hprice1": DatasetMetadata(
        key="hprice1",
        title="HPRICE1 — Konut fiyatları ve özellikleri",
        source="Wooldridge, J. M. (2020), Introductory Econometrics: A Modern Approach, HPRICE1.",
        description="Konutların satış fiyatı ve fiziksel özelliklerine ait yatay kesit verisi. Fiyat ile büyüklük göstergeleri arasındaki ilişkiyi betimlemek için kullanılır.",
        observation_unit="Konut",
        variables={
            "price": VariableMetadata("price", "Konut fiyatı", "Konutun satış fiyatıdır.", "bin ABD doları"),
            "sqrft": VariableMetadata("sqrft", "Konut alanı", "Konutun kapalı alanıdır.", "kare fit"),
            "lotsize": VariableMetadata("lotsize", "Arsa alanı", "Konutun bulunduğu arsanın alanıdır.", "kare fit"),
            "bdrms": VariableMetadata("bdrms", "Yatak odası sayısı", "Konuttaki yatak odası sayısıdır.", "adet"),
        },
        allowed_pairs={"price": ("sqrft", "lotsize", "bdrms")},
    ),
    "phillips": DatasetMetadata(
        key="phillips",
        title="PHILLIPS — Enflasyon ve işsizlik",
        source="Wooldridge, J. M. (2020), Introductory Econometrics: A Modern Approach, PHILLIPS.",
        description="1948–1996 yılları için yıllık enflasyon ve işsizlik oranları.",
        observation_unit="Yıl",
        variables={
            "year": VariableMetadata("year", "Yıl", "Takvim yılıdır.", "yıl"),
            "unem": VariableMetadata("unem", "İşsizlik oranı", "Yıllık işsizlik oranıdır.", "yüzde"),
            "inf": VariableMetadata("inf", "Enflasyon oranı", "Yıllık enflasyon oranıdır.", "yüzde"),
        },
        allowed_pairs={},
        data_structure="zaman serisi verisi",
        time_variable="year",
        time_coverage="1948–1996",
        frequency="yıllık",
        collection_method="gözlemsel",
        classification_reason="Her satır ardışık bir yılı temsil eder; yıl sırası ekonomik bilgi taşır.",
    ),
    "cps78_85": DatasetMetadata(
        key="cps78_85",
        title="CPS78_85 — İki çalışan örneklemi",
        source="Wooldridge, J. M. (2020), Introductory Econometrics: A Modern Approach, CPS78_85.",
        description="1978 ve 1985 yıllarında seçilen iki çalışan örnekleminin birleştirilmiş verisi.",
        observation_unit="Çalışan",
        variables={
            "year": VariableMetadata("year", "Örneklem yılı", "Örneklemin alındığı yıldır.", "yıl"),
            "educ": VariableMetadata("educ", "Eğitim", "Tamamlanan eğitim yılıdır.", "yıl"),
            "lwage": VariableMetadata("lwage", "Ücretin logaritması", "Saatlik ücretin doğal logaritmasıdır.", "log birim"),
            "age": VariableMetadata("age", "Yaş", "Çalışanın yaşıdır.", "yıl"),
        },
        allowed_pairs={},
        data_structure="havuzlanmış yatay kesit verisi",
        time_variable="year",
        time_coverage="1978 ve 1985",
        frequency="iki ayrı kesit dönemi",
        collection_method="gözlemsel",
        classification_reason="İki ayrı yıldan çalışan örneklemleri birleştirilmiştir; aynı kişilerin izlendiğini gösteren kimlik bilgisi yoktur.",
    ),
    "wagepan": DatasetMetadata(
        key="wagepan",
        title="WAGEPAN — Çalışan kişi-yıl gözlemleri",
        source="Wooldridge, J. M. (2020), Introductory Econometrics: A Modern Approach, WAGEPAN.",
        description="Aynı çalışanların 1980–1987 yılları arasındaki kişi-yıl gözlemleri.",
        observation_unit="Çalışan-yıl",
        variables={
            "nr": VariableMetadata("nr", "Çalışan kimliği", "Çalışanı tanımlayan kimlik numarasıdır.", "kimlik"),
            "year": VariableMetadata("year", "Yıl", "Gözlemin takvim yılıdır.", "yıl"),
            "educ": VariableMetadata("educ", "Eğitim", "Tamamlanan eğitim yılıdır.", "yıl"),
            "lwage": VariableMetadata("lwage", "Ücretin logaritması", "Saatlik ücretin doğal logaritmasıdır.", "log birim"),
        },
        allowed_pairs={},
        data_structure="panel veri",
        time_variable="year",
        time_coverage="1980–1987",
        frequency="yıllık",
        identifier_variable="nr",
        repeated_units=True,
        collection_method="gözlemsel",
        classification_reason="Aynı çalışan kimlikleri sekiz farklı yılda tekrar gözlenir; her satır bir çalışan-yıl birleşimidir.",
    ),
    "jtrain2": DatasetMetadata(
        key="jtrain2",
        title="JTRAIN2 — İş eğitimi programı",
        source="Wooldridge, J. M. (2020), Introductory Econometrics: A Modern Approach, JTRAIN2.",
        description="İş eğitimi programına atananlar ile kontrol grubunun karşılaştırılması için deneysel veri.",
        observation_unit="Katılımcı",
        variables={
            "train": VariableMetadata("train", "Grup ataması", "1 eğitim grubunu, 0 kontrol grubunu gösterir.", "gösterge"),
            "re78": VariableMetadata("re78", "1978 reel kazancı", "1978 yılındaki reel kazançtır.", "bin ABD doları"),
            "educ": VariableMetadata("educ", "Eğitim", "Tamamlanan eğitim yılıdır.", "yıl"),
        },
        allowed_pairs={},
        data_structure="yatay kesit verisi",
        time_coverage="1978 sonuç ölçümü",
        frequency="tek sonuç dönemi",
        collection_method="deneysel",
        classification_reason="Katılımcılar eğitim ve kontrol gruplarına atanmıştır; veri üretim biçimi deneysel olarak tanımlanır.",
    ),
}


def list_datasets(dataset_keys: tuple[str, ...] = ("wage1", "hprice1")) -> tuple[DatasetMetadata, ...]:
    """Katalogdaki veri setlerini kararlı sırada döndürür."""
    return tuple(get_dataset_metadata(key) for key in dataset_keys)


def get_dataset_metadata(dataset_key: str) -> DatasetMetadata:
    """Veri seti anahtarını doğrular ve katalog bilgisini döndürür."""
    try:
        return DATASETS[dataset_key]
    except KeyError as error:
        raise ValueError(f"Desteklenmeyen veri seti: {dataset_key}") from error


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """Wooldridge paketinden bir veri setini yükler."""
    metadata = get_dataset_metadata(dataset_key)
    try:
        frame = woo.data(metadata.key)
    except Exception as error:  # Paket kaynaklı hata kullanıcı arayüzünde gösterilir.
        raise RuntimeError(f"{metadata.title} veri seti yüklenemedi: {error}") from error
    if frame.empty:
        raise RuntimeError(f"{metadata.title} veri seti boş döndü.")
    missing = sorted(set(metadata.variables).difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"{metadata.title} beklenen sütunları içermiyor: {', '.join(missing)}. "
            f"Paket sürümünü ve veri seti anahtarını kontrol edin."
        )
    return frame.copy()


def allowed_explanatory_variables(dataset_key: str, dependent: str) -> tuple[str, ...]:
    """Bağımlı değişken için pedagojik olarak seçilmiş açıklayıcıları döndürür."""
    metadata = get_dataset_metadata(dataset_key)
    try:
        return metadata.allowed_pairs[dependent]
    except KeyError as error:
        raise ValueError(
            f"{dependent!r}, {metadata.title} için desteklenen bağımlı değişken değildir."
        ) from error


def variable_metadata(dataset_key: str, variable: str) -> VariableMetadata:
    """Bir değişkenin katalog bilgisini döndürür."""
    metadata = get_dataset_metadata(dataset_key)
    try:
        return metadata.variables[variable]
    except KeyError as error:
        raise ValueError(f"{variable!r}, {metadata.title} kataloğunda tanımlı değil.") from error
