# İKT 305 Ekonometri I — Etkileşimli Streamlit Uygulaması

İzmir Bakırçay Üniversitesi İKT 305 Ekonometri I dersi için hazırlanmış, Türkçe ve etkileşimli bir öğretim uygulamasıdır. Uygulama; ekonometrik kavramları gerçek veri setleri, hesaplama laboratuvarları, grafikler, tablolar ve kısa uygulama sorularıyla destekler.

Uygulama çalışma zamanında bir büyük dil modeli veya dış API kullanmaz. Wooldridge veri setleri `wooldridge` Python paketi üzerinden yerel olarak yüklenir.

## Kapsam

1. Ekonometri ve Ampirik Araştırma
2. Ekonomik Veri Türleri, Nedensellik ve Ceteris Paribus
3. Basit Doğrusal Regresyon
4. EKK Tahminini Değerlendirme: Uyum, Ölçü Birimleri ve Temel Fonksiyonel Biçimler
5. Çoklu Regresyon Modeli ve Ceteris Paribus Yorumu
6. EKK Varsayımları, Yansızlık ve Model Sorunları
7. Tek Katsayı İçin Hipotez Testleri
8. Birden Fazla Kısıtın Sınanması: F Testi ve Büyük Örneklem Mantığı
9. Ölçekleme, Logaritmik Modeller, Karesel Terimler ve Model Seçimi
10. Kukla Değişkenler ve Kategorik Açıklayıcı Değişkenler
11. Etkileşim Terimleri ve Grup Farkları
12. Heteroskedastisite ve Dayanıklı Çıkarım

## Teknik yapı

- Python 3.12
- Streamlit
- pandas, NumPy, SciPy
- statsmodels
- Plotly
- Wooldridge veri paketi
- pytest ve Streamlit AppTest

Arayüz kodu `topics/`, ortak hesaplama ve soru altyapısı `core/`, görsel stiller `assets/`, otomatik kontroller `tests/` altında bulunur. Ayrıntılı mimari açıklama için [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) dosyasına bakın.

## Kurulum

Windows PowerShell:

```powershell
git clone https://github.com/burakalparslaneroglu/ekonometri-uygulama.git
cd ekonometri-uygulama
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Yalnız uygulamayı çalıştırmak için geliştirme bağımlılıkları yerine:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uygulamayı çalıştırma

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Test ve doğrulama

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app.py core topics tests
git diff --check
```

Sürüm adayı öncesinde otomatik testlere ek olarak Konu 01–12, metin ölçeği seçenekleri, soru düğmeleri, tablolar, metric kartları ve grafik eksenleri canlı Streamlit oturumunda kontrol edilmelidir.

## Veri kaynakları

Uygulama Wooldridge veri paketindeki başlıca `WAGE1`, `HPRICE1`, `PHILLIPS`, `CPS78_85`, `WAGEPAN` ve `JTRAIN2` veri setlerini kullanır. Veri açıklamaları ve uygulamada izin verilen model eşleşmeleri `core/data_registry.py` içinde tanımlıdır.

## Dağıtım

Streamlit Community Cloud veya eşdeğer bir Streamlit ortamında:

- ana dosya: `app.py`
- çalışma zamanı: Python 3.12
- bağımlılık dosyası: `requirements.txt`
- gizli anahtar gereksinimi: yok

Dağıtım ortamının Wooldridge paketini ve paket bağımlılıklarını internetten kurabilmesi gerekir.

## Yorumlama ilkeleri

Uygulamadaki regresyon sonuçları, aksi açıkça kurulmadıkça gözlemsel ve koşullu ilişkilerdir. İstatistiksel anlamlılık tek başına nedensellik, ekonomik önem veya doğru model spesifikasyonu anlamına gelmez. Heteroskedastisiteye dayanıklı standart hatalar nokta tahminini, eksik değişken yanlılığını veya nedensel tasarımı düzeltmez.

## Kullanım notu

Bu depo ders ve öğretim amacıyla geliştirilmiştir. Depoda şu anda ayrı bir `LICENSE` dosyası bulunmadığından yeniden kullanım ve dağıtım koşulları ayrıca tanımlanmamıştır.
