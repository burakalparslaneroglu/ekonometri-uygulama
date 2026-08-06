"""Sürüm adayı dokümantasyonu ve bağımlılık ayrımını koruyan testler."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _nonempty_requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_readme_documents_scope_setup_tests_data_and_deployment() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Kapsam",
        "## Teknik yapı",
        "## Kurulum",
        "## Uygulamayı çalıştırma",
        "## Test ve doğrulama",
        "## Veri kaynakları",
        "## Dağıtım",
        "## Yorumlama ilkeleri",
        "## Kullanım notu",
    ):
        assert heading in readme

    topic_titles = (
        "Ekonometri ve Ampirik Araştırma",
        "Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus",
        "Basit Doğrusal Regresyon",
        "EKK Tahminini Değerlendirme",
        "Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu",
        "EKK Varsayımları, Yansızlık ve Model Sorunları",
        "Tek Katsayı İçin Hipotez Testleri",
        "Birden Fazla Kısıtın Sınanması",
        "Ölçekleme, Logaritmik Modeller, Karesel Terimler ve Model Seçimi",
        "Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler",
        "Etkileşim Terimleri ve Grup Farkları",
        "Heteroskedastisite ve Dayanıklı Çıkarım",
    )
    for title in topic_titles:
        assert title in readme

    for command in (
        "-m streamlit run app.py",
        "-m pytest -q",
        "-m compileall app.py core topics tests",
    ):
        assert command in readme


def test_runtime_and_development_requirements_are_separated() -> None:
    runtime = _nonempty_requirement_lines(ROOT / "requirements.txt")
    development = _nonempty_requirement_lines(ROOT / "requirements-dev.txt")

    assert not any(line.casefold().startswith("pytest") for line in runtime)
    assert "-r requirements.txt" in development
    assert any(line.casefold().startswith("pytest") for line in development)
    assert any(line.casefold().startswith("streamlit") for line in runtime)
    assert "wooldridge==0.5.0" in runtime


def test_local_review_and_patch_artifacts_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    patterns = {line.strip() for line in ignore if line.strip() and not line.startswith("#")}
    expected = {
        "review_bundle/",
        "REVIEW/",
        "audit_*.txt",
        "app_code*.zip",
        "codebase_dump.txt",
        "manifest.json",
        "PAKET_*.patch",
        "PAKET_*_TALIMATI.md",
        "*.mp4",
        "references_private/",
        "tools/generate_review_bundle.py",
    }
    assert expected.issubset(patterns)
