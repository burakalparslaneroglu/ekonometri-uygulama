"""İKT 305 Ekonometri I konu sırası. Başlıklar ders notlarındaki bölüm adlarıdır.

Yönlendirici soru konu başlığının altında gösterilir; metni notların bölüm girişindeki bağlantı kutusundan gelir.
"""

from __future__ import annotations

from core.types import TopicMetadata

TOPICS: tuple[TopicMetadata, ...] = (
    TopicMetadata(
        "konu00", 0, "Başlangıç Araç Kutusu: Veri, Notasyon ve Temel İstatistik", "Başlangıç Araç Kutusu",
        "Verinin nasıl düzenlendiğini, değişkenliğin nasıl ölçüldüğünü ve örneklemden anakütle hakkında nasıl "
        "konuşulduğunu hangi ortak dille anlatırız?",
    ),
    TopicMetadata(
        "konu01", 1, "Ekonometri ve Ampirik Araştırmanın Mantığı", "Ekonometri ve Ampirik Araştırma",
        "Bir iktisadi düşünce, verilerle incelenebilecek açık bir ekonometrik çalışmaya nasıl dönüştürülür?",
    ),
    TopicMetadata(
        "konu02", 2, "Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus", "Veri Türleri ve Nedensellik",
        "Elimizdeki verinin yapısı nedir ve bu veriyle hangi karşılaştırmalar anlamlıdır; gözlenen bir ilişki ne "
        "zaman nedensel bir etki olarak okunabilir?",
    ),
    TopicMetadata(
        "konu03", 3, "Basit Doğrusal Regresyon Modeli", "Basit Doğrusal Regresyon",
        "Bir sonuç değişkeninin ortalama düzeyi, tek bir açıklayıcı değişkenle nasıl özetlenebilir?",
    ),
    TopicMetadata(
        "konu04", 4, "EKK Tahminini Değerlendirme: Uyum, Ölçü Birimleri ve Temel Fonksiyonel Biçimler",
        "EKK Çıktısı, Uyum ve Fonksiyonel Biçimler",
        "Tahmin edilen doğru örneklemdeki gözlemlere ne ölçüde uyuyor; değişkenleri farklı ölçülerde yazdığımızda "
        "katsayıyı nasıl yorumlamalıyız?",
    ),
    TopicMetadata(
        "konu05", 5, "Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu", "Çoklu Regresyon",
        "Bir değişkenin sonuçla ilişkisi, modeldeki diğer gözlenen faktörler hesaba katıldıktan sonra nasıl "
        "incelenir?",
    ),
    TopicMetadata(
        "konu06", 6, "EKK Varsayımları, Yansızlık, Eksik Değişken Yanlılığı ve Çoklu Doğrusal Bağlantı",
        "EKK Varsayımları ve Yansızlık",
        "Bir yazılımın ürettiği katsayı, hangi varsayımlar altında anakütledeki ilişkiyi güvenilir biçimde ölçer?",
    ),
    TopicMetadata(
        "konu07", 7, "Tek Katsayı İçin Hipotez Testleri: Standart Hata, t İstatistiği, p-Değeri ve Güven Aralığı",
        "Tek Katsayı İçin Hipotez Testleri",
        "Tek bir örneklemde elde edilen katsayı ne kadar belirsizdir ve bu belirsizliği hesaba katarak anakütle "
        "parametresi hakkında hangi sınanabilir sonuçlara ulaşabiliriz?",
    ),
    TopicMetadata(
        "konu08", 8, "Birden Fazla Kısıtın Sınanması: F Testi ve Büyük Örneklem Mantığı",
        "F Testi ve Büyük Örneklem",
        "Birden fazla katsayıyı ilgilendiren bir soru, örneğin deneyim ve kıdemin birlikte ücret açıklamasına katkı "
        "sağlayıp sağlamadığı, tek bir hipotez altında nasıl sınanır?",
    ),
    TopicMetadata(
        "konu09", 9, "Ölçekleme, Logaritmik Modeller, Karesel Terimler ve Model Seçimi",
        "Ölçekleme, Log ve Karesel Terimler",
        "İlişki hangi biçimdedir: bir değişkenin etkisi kendi düzeyine göre değişiyor mu ve daha esnek bir model "
        "eklediği karmaşıklığa değer mi?",
    ),
    TopicMetadata(
        "konu10", 10, "Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler", "Kukla Değişkenler",
        "Bölge, sektör ya da medeni durum gibi sayıyla ölçülmeyen bilgiler regresyon modeline nasıl eklenir ve "
        "katsayıları hangi karşılaştırmayı anlatır?",
    ),
    TopicMetadata(
        "konu11", 11, "Etkileşim Terimleri ve Gruplar Arasında Sabit ile Eğim Farklılıkları",
        "Etkileşimler ve Grup Farkları",
        "Bir nicel değişkenin sonuçla ilişkisi, ait olunan gruba göre değişebilir mi?",
    ),
    TopicMetadata(
        "konu12", 12, "Heteroskedastisite ve Heteroskedastisiteye Dayanıklı Çıkarım",
        "Heteroskedastisite ve Dayanıklı Çıkarım",
        "Regresyon doğrusunun çevresindeki belirsizlik bütün gözlemler için aynı büyüklükte değilse ne olur?",
    ),
)

_BY_KEY = {topic.key: topic for topic in TOPICS}


def list_topics() -> tuple[TopicMetadata, ...]:
    return TOPICS


def get_topic(key: str) -> TopicMetadata:
    try:
        return _BY_KEY[key]
    except KeyError as error:
        raise KeyError(f"Tanımsız konu: {key}") from error
