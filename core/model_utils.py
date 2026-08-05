"""Basit EKK hesapları için Streamlit'ten bağımsız yardımcılar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class SimpleOLSResult:
    """Basit EKK tahmininin öğretim için gerekli çıktıları."""

    dependent: str
    explanatory: str
    intercept: float
    slope: float
    fitted_values: pd.Series
    residuals: pd.Series
    observed_values: pd.Series
    explanatory_values: pd.Series
    r_squared: float
    nobs: int

    @property
    def equation(self) -> str:
        """Tahmin denklemini açık bir metin olarak döndürür."""
        sign = "+" if self.slope >= 0 else "−"
        return f"ŷ = {self.intercept:.4f} {sign} {abs(self.slope):.4f} × {self.explanatory}"


FunctionalForm = Literal["düzey-düzey", "log-düzey", "düzey-log", "log-log"]
MIN_UNIT_SCALE = 1e-6
MAX_UNIT_SCALE = 1e6


@dataclass(frozen=True)
class ObservationDecomposition:
    """Bir gözlemdeki toplam sapmanın iki parçalı ayrıştırması."""

    observed: float
    mean_observed: float
    fitted: float
    residual: float
    total_deviation: float
    model_deviation: float
    residual_deviation: float


@dataclass(frozen=True)
class SumOfSquares:
    """Sabit terimli EKK modelinin TKT, MKT ve HKT değerleri."""

    total: float
    model: float
    error: float


@dataclass(frozen=True)
class FunctionalFormData:
    """Fonksiyonel biçim için dönüştürülmüş, tam örneklem verisi."""

    form: FunctionalForm
    data: pd.DataFrame
    dependent_name: str
    explanatory_name: str
    logged_dependent: bool
    logged_explanatory: bool


def prepare_model_data(frame: pd.DataFrame, dependent: str, explanatory: str) -> pd.DataFrame:
    """İki değişkenli modeli eksik ve geçersiz gözlemlerden arındırır."""
    if dependent == explanatory:
        raise ValueError("Bağımlı ve açıklayıcı değişkenler farklı olmalıdır.")
    missing = [column for column in (dependent, explanatory) if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    prepared = frame.loc[:, [dependent, explanatory]].apply(pd.to_numeric, errors="coerce").dropna()
    prepared = prepared[np.isfinite(prepared[dependent]) & np.isfinite(prepared[explanatory])].copy()
    if len(prepared) < 3:
        raise ValueError("Basit doğrusal regresyon için en az üç geçerli gözlem gerekir.")
    if prepared[explanatory].nunique() < 2:
        raise ValueError("Açıklayıcı değişken en az iki farklı değer içermelidir.")
    return prepared


def fit_simple_ols(frame: pd.DataFrame, dependent: str, explanatory: str) -> SimpleOLSResult:
    """Sabit terimli basit EKK modelini statsmodels ile tahmin eder."""
    prepared = prepare_model_data(frame, dependent, explanatory)
    observed = prepared[dependent].to_numpy(dtype=float)
    total_variation = float(np.square(observed - observed.mean()).sum())
    if np.isclose(total_variation, 0.0, atol=1e-12):
        raise ValueError("Bağımlı değişkende örneklem değişimi olmadığı için R-kare hesaplanamaz.")
    design = sm.add_constant(prepared[explanatory], has_constant="add")
    fitted_model = sm.OLS(prepared[dependent], design).fit()
    fitted_values = pd.Series(fitted_model.fittedvalues, index=prepared.index, name="tahmin")
    residuals = pd.Series(fitted_model.resid, index=prepared.index, name="artık")
    return SimpleOLSResult(
        dependent=dependent,
        explanatory=explanatory,
        intercept=float(fitted_model.params["const"]),
        slope=float(fitted_model.params[explanatory]),
        fitted_values=fitted_values,
        residuals=residuals,
        observed_values=prepared[dependent].rename(dependent),
        explanatory_values=prepared[explanatory].rename(explanatory),
        r_squared=float(fitted_model.rsquared),
        nobs=int(fitted_model.nobs),
    )


def descriptive_statistics(frame: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    """Seçili değişkenler için temel tanımlayıcı istatistikleri hesaplar."""
    missing = [column for column in variables if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde bulunamayan değişken: {', '.join(missing)}")
    summary = frame.loc[:, list(variables)].apply(pd.to_numeric, errors="coerce").describe().T
    return summary.loc[:, ["count", "mean", "std", "min", "max"]].rename(
        columns={"count": "Gözlem", "mean": "Ortalama", "std": "Std. sapma", "min": "En küçük", "max": "En büyük"}
    )


def predict_value(result: SimpleOLSResult, x_value: float) -> float:
    """Geçerli bir X değeri için tahmin edilen Y değerini hesaplar."""
    intercept, slope = _model_coefficients(result)
    return fitted_value(intercept, slope, x_value)


def _model_coefficients(result: SimpleOLSResult) -> tuple[float, float]:
    """Tahmin için gerekli model katsayılarını doğrular."""
    try:
        return result.intercept, result.slope
    except AttributeError as error:
        raise ValueError("Model tahmin için sabit terim ve eğim katsayısı içermelidir.") from error


def fitted_value(intercept: float, slope: float, x_value: float) -> float:
    """Sabit, eğim ve X değeriyle tahmin edilen değeri hesaplar."""
    values = {"Sabit terim": intercept, "Eğim katsayısı": slope, "X değeri": x_value}
    converted: dict[str, float] = {}
    for label, value in values.items():
        try:
            converted[label] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} sayısal olmalıdır.") from error
        if not np.isfinite(converted[label]):
            raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return converted["Sabit terim"] + converted["Eğim katsayısı"] * converted["X değeri"]


def residual_value(observed_value: float, predicted_value: float) -> float:
    """Gerçekleşen ve tahmin edilen değerlerden artığı hesaplar."""
    values = {"Y değeri": observed_value, "Tahmin edilen Y değeri": predicted_value}
    converted: dict[str, float] = {}
    for label, value in values.items():
        try:
            converted[label] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} sayısal olmalıdır.") from error
        if not np.isfinite(converted[label]):
            raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    return converted["Y değeri"] - converted["Tahmin edilen Y değeri"]


def observation_result(result: SimpleOLSResult, position: int) -> dict[str, float | int]:
    """Sıfırdan başlayan sıra numarasıyla seçilen gözlemin tahmin ve artığını döndürür."""
    if not isinstance(position, (int, np.integer)):
        raise ValueError("Gözlem sırası tam sayı olmalıdır.")
    if position < 0 or position >= result.nobs:
        raise IndexError(f"Gözlem sırası 0 ile {result.nobs - 1} arasında olmalıdır.")
    x_value = result.explanatory_values.iloc[position]
    observed_value = result.observed_values.iloc[position]
    if pd.isna(x_value) or pd.isna(observed_value):
        raise ValueError("Seçili gözlemde X ve Y değerleri eksik olamaz.")
    intercept, slope = _model_coefficients(result)
    predicted_value = fitted_value(intercept, slope, x_value)
    residual = residual_value(observed_value, predicted_value)
    index = result.observed_values.index[position]
    return {
        "index": int(index),
        "x": float(x_value),
        "observed": float(observed_value),
        "predicted": predicted_value,
        "residual": residual,
    }


def observation_decomposition(result: SimpleOLSResult, position: int) -> ObservationDecomposition:
    """Seçili gözlem için Y−Ȳ=(Ŷ−Ȳ)+û eşitliğini hesaplar."""
    item = observation_result(result, position)
    mean_observed = float(result.observed_values.mean())
    total_deviation = float(item["observed"]) - mean_observed
    model_deviation = float(item["predicted"]) - mean_observed
    residual = float(item["residual"])
    return ObservationDecomposition(
        observed=float(item["observed"]), mean_observed=mean_observed, fitted=float(item["predicted"]),
        residual=residual, total_deviation=total_deviation, model_deviation=model_deviation,
        residual_deviation=residual,
    )


def sum_of_squares(result: SimpleOLSResult) -> SumOfSquares:
    """TKT, MKT ve HKT'yi model sonucundan hesaplar."""
    observed = result.observed_values.to_numpy(dtype=float)
    fitted = result.fitted_values.to_numpy(dtype=float)
    residuals = result.residuals.to_numpy(dtype=float)
    if not (np.isfinite(observed).all() and np.isfinite(fitted).all() and np.isfinite(residuals).all()):
        raise ValueError("Kareler toplamları için tüm model değerleri sonlu olmalıdır.")
    mean_observed = float(observed.mean())
    total = float(np.square(observed - mean_observed).sum())
    if np.isclose(total, 0.0, atol=1e-12):
        raise ValueError("TKT sıfır olduğunda R-kare tanımlı değildir.")
    return SumOfSquares(
        total=total,
        model=float(np.square(fitted - mean_observed).sum()),
        error=float(np.square(residuals).sum()),
    )


def validate_sum_of_squares(sums: SumOfSquares, *, tolerance: float = 1e-8) -> bool:
    """TKT=MKT+HKT ayrıştırmasını uygun sayısal toleransla doğrular."""
    if tolerance < 0:
        raise ValueError("Tolerans negatif olamaz.")
    return bool(np.isclose(sums.total, sums.model + sums.error, rtol=tolerance, atol=tolerance))


def r_squared_from_model_sum(sums: SumOfSquares) -> float:
    """R-kareyi MKT/TKT biçiminde hesaplar."""
    if np.isclose(sums.total, 0.0, atol=1e-12):
        raise ValueError("TKT sıfır olduğunda R-kare tanımlı değildir.")
    return sums.model / sums.total


def r_squared_from_error_sum(sums: SumOfSquares) -> float:
    """R-kareyi 1−HKT/TKT biçiminde hesaplar."""
    if np.isclose(sums.total, 0.0, atol=1e-12):
        raise ValueError("TKT sıfır olduğunda R-kare tanımlı değildir.")
    return 1.0 - sums.error / sums.total


def rescaled_coefficients(result: SimpleOLSResult, *, dependent_scale: float = 1.0, explanatory_scale: float = 1.0) -> tuple[float, float]:
    """Y'=aY ve X'=bX için yeni sabit ve eğimi cebirsel olarak verir."""
    a = validate_unit_scale(dependent_scale, label="Y çarpanı")
    b = validate_unit_scale(explanatory_scale, label="X çarpanı")
    return result.intercept * a, result.slope * a / b


def validate_unit_scale(value: float, *, label: str = "Ölçü birimi çarpanı") -> float:
    """Ölçü birimi dönüşüm çarpanının pozitif ve belirlenen aralıkta olduğunu doğrular."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} sayısal olmalıdır.") from error
    if not np.isfinite(scale):
        raise ValueError(f"{label} sonlu bir sayı olmalıdır.")
    if scale <= 0:
        raise ValueError(f"{label} pozitif olmalıdır; negatif değer ölçü birimi değil işaret dönüşümüdür.")
    if not MIN_UNIT_SCALE <= scale <= MAX_UNIT_SCALE:
        raise ValueError(f"{label} {MIN_UNIT_SCALE:g} ile {MAX_UNIT_SCALE:g} arasında olmalıdır.")
    return scale


def format_number(value: float, *, decimals: int = 4) -> str:
    """Sonlu sayıyı büyüklüğüne göre kısa ondalık veya bilimsel gösterimle biçimler."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Biçimlendirilecek değer sayısal olmalıdır.") from error
    if not np.isfinite(number):
        raise ValueError("Biçimlendirilecek değer sonlu olmalıdır.")
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude < 10 ** (-decimals) or magnitude >= 10 ** (decimals + 2):
        return f"{number:.{decimals}e}"
    return f"{number:.{decimals}f}".rstrip("0").rstrip(".")


def transform_functional_form(
    frame: pd.DataFrame, dependent: str, explanatory: str, form: FunctionalForm
) -> FunctionalFormData:
    """Dört temel biçim için dönüşümü, log uygunluğunu açıkça doğrulayarak uygular."""
    if form not in {"düzey-düzey", "log-düzey", "düzey-log", "log-log"}:
        raise ValueError("Desteklenmeyen fonksiyonel biçim.")
    raw = prepare_model_data(frame, dependent, explanatory)
    logged_dependent = form in {"log-düzey", "log-log"}
    logged_explanatory = form in {"düzey-log", "log-log"}
    for column, logged in ((dependent, logged_dependent), (explanatory, logged_explanatory)):
        if logged:
            invalid_count = int((raw[column] <= 0).sum())
            if invalid_count:
                raise ValueError(
                    f"{column} için log dönüşümü yapılamaz: {invalid_count} gözlem sıfır veya negatiftir; örneklem değiştirilmedi."
                )
    transformed = pd.DataFrame(index=raw.index)
    dependent_name = f"ln({dependent})" if logged_dependent else dependent
    explanatory_name = f"ln({explanatory})" if logged_explanatory else explanatory
    transformed[dependent_name] = np.log(raw[dependent]) if logged_dependent else raw[dependent]
    transformed[explanatory_name] = np.log(raw[explanatory]) if logged_explanatory else raw[explanatory]
    return FunctionalFormData(
        form=form, data=transformed, dependent_name=dependent_name, explanatory_name=explanatory_name,
        logged_dependent=logged_dependent, logged_explanatory=logged_explanatory,
    )


def functional_form_interpretation(form: FunctionalForm, slope: float) -> str:
    """Ders notundaki yaklaşık ve tam yüzde yorum bileşenini üretir."""
    if form == "düzey-düzey":
        return f"X bir birim arttığında Y'nin tahmin edilen değeri {format_number(slope)} birim değişir."
    if form == "log-düzey":
        return f"X bir birim arttığında Y yaklaşık %{format_number(100 * slope)} değişir; tam yüzde karşılığı %{format_number(100 * (np.exp(slope) - 1))}'dir."
    if form == "düzey-log":
        return f"X yüzde 1 arttığında Y yaklaşık {format_number(slope / 100)} birim değişir."
    return f"X yüzde 1 arttığında Y yaklaşık %{format_number(slope)} değişir; eğim katsayısı esnekliktir."
