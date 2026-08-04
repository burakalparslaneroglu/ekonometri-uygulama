# Mimari

Uygulama, Streamlit arayüzü ile hesaplama mantığını ayıran küçük modüllerden oluşur.

- `app.py`: Ortak sayfa yapılandırması, PAÜ görsel iskeleti ve konu yönlendirmesi.
- `topics/`: Her ders konusunun Streamlit görünümü. `konu03_basit_regresyon.py`, çekirdek fonksiyonları bir araya getirir.
- `core/data_registry.py`: Wooldridge kataloğu, değişken açıklamaları ve izin verilen pedagojik eşleşmeler.
- `core/model_utils.py`: Veri hazırlama, basit EKK, belirli bir X değeri için tahmin ve tanımlayıcı istatistikler. Streamlit bağımlılığı yoktur.
- `core/question_engine.py`: Model kimliği ve soru sırasından deterministik soru/çözüm üretir.
- `core/session_utils.py`: Soru sırası ve cevap görünürlüğünün `st.session_state` içindeki anahtarlarını yönetir.
- `assets/styles.css`: PAÜ renk paleti ve duyarlı görsel düzen.

## Veri akışı

1. Konu modülü, `data_registry` üzerinden Wooldridge paketindeki veri setini önbellekli olarak yükler.
2. Öğrencinin seçtiği izinli Y–X çifti `model_utils.fit_simple_ols` fonksiyonuna verilir.
3. Model çıktısı grafik, tablo, belirli bir X değeri için tahmin paneli ve soru motoru tarafından kullanılır.
4. Veri seti veya değişkenler değiştiğinde model kimliği değişir; `session_utils` önceki soru durumunu sıfırlar.

Model sonuçları betimseldir. Bu pilotta istatistiksel çıkarım sütunları ve yorumları bilinçli olarak sunulmaz.
