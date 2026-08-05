"""Model sonuçlarından yeniden üretilebilir öğretim soruları üretir."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from core.model_utils import SimpleOLSResult, observation_result, predict_value


KONU03_QUESTION_TYPES = (
    "slope",
    "intercept",
    "prediction",
    "direction",
    "line_meaning",
    "residual",
    "residual_sign",
    "relative_to_line",
    "absolute_residual",
)


@dataclass(frozen=True)
class GeneratedQuestion:
    """Bir sorunun öğrenciye gösterilecek metnini ve gizli çözümünü tutar."""

    question_type: str
    prompt: str
    answer: str
    index: int


def _stable_number(model_id: str, question_index: int, salt: str = "") -> int:
    """Kimlikten süreçler arası sabit bir tamsayı üretir."""
    source = f"{model_id}|{question_index}|{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big")


def cycle_question_type(
    model_id: str,
    question_index: int,
    question_types: tuple[str, ...],
    *,
    namespace: str,
) -> str:
    """Bir konuya ait soru türlerini deterministik ve ardışık döndürür."""
    if question_index < 0:
        raise ValueError("Soru sırası negatif olamaz.")
    if not question_types:
        raise ValueError("En az bir soru türü tanımlanmalıdır.")
    start = _stable_number(model_id, 0, f"{namespace}:type") % len(question_types)
    return question_types[(start + question_index) % len(question_types)]


def question_type_for(model_id: str, question_index: int) -> str:
    """Konu 03 için ardışık sıralarda farklı soru türü seçer."""
    return cycle_question_type(model_id, question_index, KONU03_QUESTION_TYPES, namespace="konu03")


def generate_question(
    result: SimpleOLSResult,
    model_id: str,
    question_index: int,
    *,
    dependent_label: str | None = None,
    explanatory_label: str | None = None,
    dependent_unit: str | None = None,
    explanatory_unit: str | None = None,
    observation_position: int | None = None,
) -> GeneratedQuestion:
    """Model sonucuna bağlı, deterministik ve özgün bir soru üretir."""
    question_type = question_type_for(model_id, question_index)
    x_name = explanatory_label or result.explanatory
    y_name = dependent_label or result.dependent
    x_unit = explanatory_unit or "birim"
    y_unit = dependent_unit or "birim"
    selected_position = (
        observation_position
        if observation_position is not None
        else _stable_number(model_id, question_index, "observation") % result.nobs
    )
    selected_observation = observation_result(result, selected_position)
    observation_number = selected_position + 1
    if question_type == "slope":
        direction = "artar" if result.slope >= 0 else "azalır"
        prompt = f"Tahmin edilen modelde {x_name} bir birim arttığında {y_name} için beklenen değişimi nasıl yorumlarsınız?"
        answer = (
            f"Eğim katsayısı {result.slope:.4f}'tür. {x_name} {x_unit} cinsinden bir birim arttığında, "
            f"{y_name}'nin tahmin edilen değeri ortalama olarak {abs(result.slope):.4f} {y_unit} {direction}. "
            "Bu, tek başına nedensel bir yorum değildir."
        )
    elif question_type == "intercept":
        zero_count = int((result.explanatory_values == 0).sum())
        prompt = f"Modelin sabit terimi {x_name}=0 iken neyi ifade eder? X=0 örneklem açısından anlamlı bir değeri temsil eder mi?"
        if zero_count:
            sample_statement = f"X=0 örneklemde {zero_count} kez gözlenir; bu nedenle veri aralığındadır."
        else:
            sample_statement = (
                f"X=0 örneklemde gözlenmez; açıklayıcı değişkenin gözlenen aralığı "
                f"{result.explanatory_values.min():.2f} ile {result.explanatory_values.max():.2f} arasındadır."
            )
        answer = (
            f"Sabit terim {result.intercept:.4f}, {x_name}=0 iken {y_name}'nin tahmin edilen değeridir. "
            f"{sample_statement} Bu nedenle sabitin iktisadi yorumu dikkatle yapılmalıdır; sabit çoğu zaman doğrunun konumunu belirleyen bir bileşendir."
        )
    elif question_type == "prediction":
        position = _stable_number(model_id, question_index, "prediction") % result.nobs
        x_value = float(result.explanatory_values.iloc[position])
        predicted = predict_value(result, x_value)
        prompt = f"{x_name} = {x_value:.2f} {x_unit} olduğunda modelin {y_name} için tahmini nedir?"
        answer = f"ŷ = {result.intercept:.4f} + ({result.slope:.4f} × {x_value:.2f}) = {predicted:.4f} {y_unit}."
    elif question_type == "direction":
        direction = "pozitif; doğru yukarı eğimlidir" if result.slope > 0 else "negatif; doğru aşağı eğimlidir"
        if result.slope == 0:
            direction = "sıfırdır; tahmin edilen doğru yataydır"
        prompt = f"Eğim katsayısının işaretine göre {x_name} ile {y_name} arasındaki doğrusal ilişkinin yönü nedir?"
        answer = (
            f"Eğim katsayısı {result.slope:.4f} olduğundan ilişkinin yönü {direction}. "
            "Bu yön, örneklemdeki doğrusal ilişkiyi özetler; tek başına nedensel etki göstermez."
        )
    elif question_type == "line_meaning":
        prompt = "Tahmin edilen regresyon doğrusu neyi betimler? Neden bu doğru tek başına nedensellik göstermez?"
        answer = (
            f"Tahmin edilen doğru, örneklemde {x_name}'nin farklı değerlerinde {y_name}'nin ortalama davranışını doğrusal olarak özetler. "
            "Gözlemsel yatay kesit verisinde değişkenlerin birlikte hareket etmesi, tek başına birinin diğerine neden olduğunu göstermez."
        )
    elif question_type == "residual":
        prompt = (
            f"{observation_number}. gözlemde {y_name}={selected_observation['observed']:.4f} ve "
            f"tahmin edilen değer {selected_observation['predicted']:.4f} {y_unit}. Artık nedir?"
        )
        answer = (
            f"û = y − ŷ = {selected_observation['observed']:.4f} − "
            f"{selected_observation['predicted']:.4f} = {selected_observation['residual']:.4f} {y_unit}."
        )
    elif question_type == "residual_sign":
        sign = "pozitiftir" if selected_observation["residual"] > 0 else "negatiftir"
        relation = "üzerindedir" if selected_observation["residual"] > 0 else "altındadır"
        if abs(selected_observation["residual"]) < 1e-10:
            sign = "sıfıra yakındır"
            relation = "yaklaşık olarak aynıdır"
        prompt = f"{observation_number}. gözlemin artığının işaretini nasıl yorumlarsınız?"
        answer = (
            f"Artık {selected_observation['residual']:.4f} olduğu için {sign}; gerçekleşen {y_name}, "
            f"tahmin edilen değerle {relation}."
        )
    elif question_type == "relative_to_line":
        prompt = f"{observation_number}. gözlem tahmin doğrusunun üstünde mi, altında mı yer alır?"
        if abs(selected_observation["residual"]) < 1e-10:
            answer = "Artık sıfıra yakın olduğundan gözlem tahmin doğrusu üzerinde yer alır."
        elif selected_observation["residual"] > 0:
            answer = "Pozitif artık nedeniyle gözlem tahmin doğrusunun üstünde yer alır."
        else:
            answer = "Negatif artık nedeniyle gözlem tahmin doğrusunun altında yer alır."
    else:
        prompt = f"{observation_number}. gözlem için büyük bir mutlak artık neyi gösterir; tek başına neyi kanıtlamaz?"
        answer = (
            f"Bu gözlemde mutlak artık |û|={abs(selected_observation['residual']):.4f} {y_unit}; bu değer, "
            "gözleme özgü tahmin hatasının büyüklüğünü gösterir. Tek başına veri giriş hatasını veya nedensel bir sonucu kanıtlamaz."
        )
    return GeneratedQuestion(question_type=question_type, prompt=prompt, answer=answer, index=question_index)
