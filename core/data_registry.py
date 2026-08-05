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
}


def list_datasets() -> tuple[DatasetMetadata, ...]:
    """Katalogdaki veri setlerini kararlı sırada döndürür."""
    return tuple(DATASETS[key] for key in ("wage1", "hprice1"))


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
