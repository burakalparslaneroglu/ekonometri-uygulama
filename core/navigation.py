"""Sunum bağlantıları: Streamlit'ten bağımsız URL doğrulama ve başlangıç durumu."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from math import isclose, isfinite
from typing import Mapping, Sequence

from core.labs.registry import LABS
from core.labs.sezgi import SimExperiment
from core.topic_registry import get_topic

TAB_LABELS = {"uygulama": "Uygulama", "sezgi": "Sezgi", "sinama": "Kendini sına"}


@lru_cache(maxsize=13)
def experiments(topic_key: str) -> tuple[SimExperiment, ...]:
    """Deney numaraları ve ayarlar doğrudan mevcut tanımlardan gelir."""
    module = import_module(f"core.labs.sezgi_{topic_key}")
    return getattr(module, f"{topic_key.upper()}_EXPERIMENTS")


@dataclass(frozen=True)
class Route:
    topic_key: str
    tab: str
    experiment: int | None = None
    step: int | None = None
    source: str = "notlar"
    parameters: tuple[tuple[str, float | int], ...] = ()


def _integer(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"{label} bir negatif olmayan tam sayı olmalı.")
    return int(value)


def parse_route(query: Mapping[str, Sequence[str]]) -> Route | None:
    """Tekrarlı, geçersiz veya çelişen parametreleri sessizce düzeltmeden reddeder."""
    if not query:
        return None
    values = {}
    for key, entries in query.items():
        if len(entries) != 1:
            raise ValueError(f"{key} parametresi yalnız bir kez verilebilir.")
        values[key] = entries[0]
    if "konu" not in values:
        raise ValueError("Sunum bağlantısında konu parametresi gerekli (00–12).")
    number = _integer(values["konu"], "Konu")
    topic = f"konu{number:02d}"
    if topic not in LABS:
        raise ValueError("Konu 00–12 arasında olmalı.")
    tab = values.get("sekme", "uygulama")
    if tab not in TAB_LABELS:
        raise ValueError("Sekme uygulama, sezgi veya sinama olmalı.")
    allowed = {"konu", "sekme"}
    experiment, step, parameters, source = None, None, (), "notlar"
    if tab == "sezgi":
        allowed.add("deney")
        if "deney" in values:
            experiment = _integer(values["deney"], "Deney")
            matches = [e for e in experiments(topic) if e.number == experiment]
            if not matches:
                raise ValueError(f"Konu {number:02d} için Deney {experiment} bulunmuyor.")
            settings = {p.key: int(p.default) if p.integer else float(p.default)
                        for p in matches[0].parameters}
            for item in matches[0].parameters:
                key = f"ayar_{item.key}"
                allowed.add(key)
                if key not in values:
                    continue
                try:
                    value = float(values[key])
                except ValueError:
                    raise ValueError(f"{key} sayısal olmalı.") from None
                if not isfinite(value) or not item.minimum <= value <= item.maximum:
                    raise ValueError(f"{key}: değer {item.minimum}–{item.maximum} aralığında olmalı.")
                if item.integer and not value.is_integer():
                    raise ValueError(f"{key} tam sayı olmalı.")
                position = (value - item.minimum) / item.step
                if not isclose(position, round(position), rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"{key}: kaydırıcının {item.step} adımına uymalı.")
                settings[item.key] = int(value) if item.integer else value
            parameters = tuple(sorted(settings.items()))
    elif tab == "uygulama":
        allowed.update({"adim", "kaynak"})
        source = values.get("kaynak", "notlar")
        if source not in {"notlar", "alternatif", "kendi"}:
            raise ValueError("Veri kaynağı notlar, alternatif veya kendi olmalı.")
        if "adim" in values:
            step = _integer(values["adim"], "Adım")
            if step not in {s.number for s in LABS[topic].steps}:
                raise ValueError(f"Konu {number:02d} için Adım {step} bulunmuyor.")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError("Bu hedef için geçersiz parametre: " + ", ".join(sorted(unknown)))
    return Route(topic, tab, experiment, step, source, parameters)


def initial_state(route: Route) -> dict[str, object]:
    """Geçerli URL'nin yalnız ilk açılışta uygulanacak atomik widget değerleri."""
    topic = route.topic_key
    state: dict[str, object] = {"selected_topic": get_topic(topic).label,
                                f"{topic}_tab": TAB_LABELS[route.tab]}
    if route.experiment is not None:
        state[f"{topic}_sezgi_deney"] = route.experiment
        state.update({f"{topic}_sezgi{route.experiment}_{name}": value
                      for name, value in route.parameters})
    if route.tab == "uygulama":
        state[f"{topic}_lab_kaynak"] = route.source
        if route.step is not None:
            state[f"{topic}_lab_step"] = route.step
        if route.source == "notlar":
            state.update({f"{topic}_secim_{c.key}": list(c.default) if isinstance(c.default, tuple) else c.default
                          for c in LABS[topic].controls})
    return state
