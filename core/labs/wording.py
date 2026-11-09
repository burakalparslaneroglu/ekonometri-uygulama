"""Uygulama metinlerinde sayıya bağlı Türkçe ifadeler.

Kural: değişken bir sayıya Türkçe ek getirilmez ("0,54'lük" değil "0,54 birim"); yön sözcükleri sayıdan
ayrı yazılır.
"""

from __future__ import annotations

LOG_MEAN_NOTE = (
    "\n\n**Ortalama hedefi:** Sonuç düzeydeyse koşullu aritmetik ortalama; sonuç logaritmikse "
    "exp(E[ln(Y) | X]) ile tanımlanan koşullu geometrik ortalama yorumlanır. Bu yorum, doğru koşullu ortalama "
    "modeline ve E(u | X) = 0 koşuluna dayanır. Log modelde aynı yüzde farkı aritmetik ortalamaya taşımak için "
    "E[exp(u) | X] çarpanının karşılaştırılan değerlerde aynı olması gerekir."
)

MULTIPLE_POPULATION_NOTE = (
    "\n\n**Anakütle ve örneklem.** "
    r"$\mathbb{E}(u_i\mid X_{1i},\ldots,X_{ki})=0$ varsayımı altında "
    r"$\mathbb{E}(Y_i\mid X_{1i},\ldots,X_{ki})=\beta_0+\sum_{j=1}^{k}\beta_jX_{ji}$ (Denklem 5.3). "
    "Örneklem uyum denkleminde bilinmeyen parametrelerin yerine EKK tahminleri konur (Denklem 5.4)."
)

MULTIPLE_NORMAL_NOTE = (
    "\n\n**Normal denklemler.** Türevleri çözümde sıfıra eşitleyince "
    r"$\sum_i\widehat u_i=0$ ve her açıklayıcı değişken için $\sum_iX_{ji}\widehat u_i=0$ elde edilir (Denklem 5.7). "
    r"Bu yüzden $\overline{\widehat Y}=\bar Y$. Tabloda artık değerlerin toplamını ve iki ortalamayı karşılaştırın. "
    "Bu eşitlikler örneklemdeki EKK hesabından doğar."
)

FWL_NOTE = (
    "**Frisch–Waugh–Lovell teoremi.** Aynı gözlemlerde sonucu ve incelenen değişkeni ayrı ayrı kesim terimi "
    "ve aynı kontroller üzerine regresyona alın. İki yardımcı modelin artık değerlerini karşılaştıran eğim, "
    r"$\widehat\beta_j=\sum_i\widetilde X_{ji}\widetilde Y_i/\sum_i\widetilde X_{ji}^2$, tam modeldeki "
    "kısmi eğimdir (Denklem 5.8). Bu hesap için incelenen değişkenin kontrollerden arta kalan değişkenliği "
    "pozitif olmalıdır. Yardımcı modellerin artık değerleri sıfır ortalamalı olduğu için son regresyonun "
    "kesim terimi eğimi değiştirmez."
)

MULTIPLE_SLOPE_DIFFERENCE_NOTE = (
    "\n\n**İki açıklayıcı değişkenle katsayı farkı.** Aynı gözlemler ve kesim terimi kullanıldığında "
    r"$\widehat b_{1,\mathrm{basit}}=\widehat\beta_1+\widehat\beta_2\widehat\delta_1$ (Denklem 5.9). "
    r"$\widehat\delta_1$, $X_2$'nin $X_1$ üzerine kesim terimli regresyonundaki eğimdir. "
    "Dört gözlemli çalışma saati örneğinde basit eğim 3, kısmi eğim 2 ve kontrol katkısı 1'dir."
)


def tr_lower(text: str) -> str:
    """Türkçe küçük harf: I → ı, İ → i (``str.lower`` "İ"yi noktalı iki karaktere çevirir)."""

    return text.replace("I", "ı").replace("İ", "i").lower()


def at_zero(value: float, decimals: int) -> bool:
    """Değer gösterilen basamakta sıfır mı (ör. dört basamakta |değer| < 0,00005)."""

    return abs(value) < 0.5 * 10 ** (-decimals)


def directions(start: str, end: str) -> tuple[str, str]:
    """İleri ve geri yönün etiketi ("12 → 16", "16 → 12").

    Başlangıç ile yeni değer aynıysa yön x₀/x₁ ile ayrılır; aynı etiketli iki satır üretilen Python kodunda
    (sözlükle kurulan tablo) birleşirdi.
    """

    if start == end:
        return f"x₀ → x₁ ({start} → {end})", f"x₁ → x₀ ({end} → {start})"
    return f"{start} → {end}", f"{end} → {start}"


def signed_difference(value: float, tolerance: float = 1e-12) -> str:
    """Bir farkın yönü, yüklemle: "yüksektir", "düşüktür" ya da "aynıdır"."""

    if value > tolerance:
        return "yüksektir"
    if value < -tolerance:
        return "düşüktür"
    return "aynıdır"
