from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _r_version(path: str) -> tuple[int, ...]:
    """``.../R/R-4.4.1/bin/Rscript.exe`` → (4, 4, 1); en yeni kurulumu seçmek için."""

    match = re.search(r"R-(\d+(?:\.\d+)*)", path)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


@lru_cache(maxsize=1)
def find_rscript() -> str | None:
    """Rscript yolu: ``RSCRIPT`` ortam değişkeni, sonra PATH, Windows'ta sonra standart R kurulum klasörleri."""

    configured = os.environ.get("RSCRIPT")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("Rscript")
    if found:
        return found
    if sys.platform == "win32":
        candidates: list[str] = []
        for variable, parts in (("ProgramFiles", ("R",)), ("ProgramW6432", ("R",)),
                                ("LOCALAPPDATA", ("Programs", "R"))):
            root = os.environ.get(variable)
            if root:
                candidates += glob.glob(os.path.join(root, *parts, "R-*", "bin", "Rscript.exe"))
        if candidates:
            return max(candidates, key=_r_version)
    return None


def _r_environment() -> dict[str, str]:
    """Rscript alt süreçlerinin ortam değişkenleri.

    R iletileri İngilizcedir (``LANGUAGE=en``): testlerin "warning" denetimi R'nin ileti çevirisine bağlı kalmaz
    (Türkçe yerel ayarda uyarı ve hata metinleri Türkçe yazılır). Linux'ta ``C.UTF-8``, macOS'ta ``en_US.UTF-8``
    yerel ayarı kullanılır. Windows'ta R 4.2'den beri yerel kodlama UTF-8'dir ve ``C.UTF-8`` adlı yerel ayar yoktur:
    ``LC_ALL=C.UTF-8`` verilirse R açılışta "Setting LC_... failed" uyarıları yazar ve C yerel ayarına düşer.
    Windows'ta ``LC_*`` ve ``LANG`` ortamdan çıkarılır; R sistemin yerel ayarıyla (öğrencinin koşulu) çalışır.
    Üretilen R kodu yerel ayara bağlı sıralama ve harf dönüşümü kullanmaz: çıktısı C, Türkçe ve İngilizce yerel
    ayarda aynıdır."""

    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LC_") and key not in ("LANG", "LANGUAGE")}
    environment["LANGUAGE"] = "en"
    if sys.platform.startswith("linux"):
        environment.update(LANG="C.UTF-8", LC_ALL="C.UTF-8")
    elif sys.platform == "darwin":
        environment.update(LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8")
    return environment


@lru_cache(maxsize=1)
def _r_missing() -> str | None:
    """Üretilen R kodunu çalıştırmanın önündeki engel; ``None`` ise Rscript ve R ``wooldridge`` paketi hazırdır."""

    rscript = find_rscript()
    if rscript is None:
        return "Rscript bulunamadı (PATH, RSCRIPT ortam değişkeni ve standart R kurulum klasörleri)."
    check = 'quit(status = if (requireNamespace("wooldridge", quietly = TRUE)) 0 else 1)'
    try:
        result = subprocess.run([rscript, "-e", check], capture_output=True, timeout=120, env=_r_environment())
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Rscript çalıştırılamadı: {error}"
    if result.returncode != 0:
        return f'R "wooldridge" paketi kurulu değil ({rscript}): install.packages("wooldridge")'
    return None


@pytest.fixture(scope="session")
def rscript() -> str:
    """R testleri için Rscript yolu. Rscript ya da R ``wooldridge`` paketi yoksa test başarısız olmaz, atlanır."""

    reason = _r_missing()
    if reason is not None:
        pytest.skip(reason)
    return find_rscript()


@pytest.fixture(scope="session")
def r_environment() -> dict[str, str]:
    """Rscript alt süreçlerinin ortam değişkenleri (``_r_environment``)."""

    return _r_environment()
