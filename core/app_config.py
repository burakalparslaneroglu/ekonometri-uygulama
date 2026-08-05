"""Uygulamanın kurumsal kimlik yapılandırması."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Uygulama genelinde kullanılan sabit kurumsal metinler."""

    course_name: str
    application_subtitle: str
    institution_name: str


APP_CONFIG = AppConfig(
    course_name="İKT 305 Ekonometri I",
    application_subtitle="Etkileşimli Uygulama",
    institution_name="İzmir Bakırçay Üniversitesi",
)
