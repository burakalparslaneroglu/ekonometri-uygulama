# İKT 305 Ekonometri I — Etkileşimli Streamlit Uygulaması

İzmir Bakırçay Üniversitesi İKT 305 Ekonometri I dersi için hazırlanmış, Türkçe ve etkileşimli bir öğretim uygulamasıdır. Uygulama; ekonometrik kavramları gerçek veri setleri, hesaplama laboratuvarları, grafikler, tablolar ve kısa uygulama sorularıyla destekler.

Uygulama çalışma zamanında bir büyük dil modeli veya dış API kullanmaz. Wooldridge veri setleri `wooldridge` Python paketi üzerinden yerel olarak yüklenir.

Konu 0–10 yeni yapıdadır: her konu **Uygulama**, **Sezgi** ve **Kendini sına** sekmelerinden oluşur. Uygulama sekmesi notlardaki laboratuvarın adımlarını izler ve notlarda basılı her sayıyı gerçek veriyle yeniden üretir; öğrenci bir adımda farklı bir spesifikasyon seçebilir (ör. açıklayıcı değişken), seçim yalnız ona bağlı sonraki adımlara geçer ve notlardaki modelle yan yana gösterilir. Her adımın ve her Sezgi deneyinin Python ve R kodu aynı tanımdan üretilir. Konu 11–12 eski sayfalarıyla çalışır ve son blokta yeni yapıya taşınır.

## Kapsam

0. Başlangıç Araç Kutusu: Veri, Notasyon ve Temel İstatistik
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

| Konu | Yapı | Uygulama verisi | Sezgi | Kendini sına |
|---|---|---|---|---|
| 0 | Uygulama · Sezgi · Kendini sına | Tablo 0.1–0.3 (küçük örnekler), WAGE1 | 3 deney | 24 soru |
| 1 | Uygulama · Sezgi · Kendini sına | WAGE1 (§1.6) | 3 deney | 24 soru |
| 2 | Uygulama · Sezgi · Kendini sına | WAGE1, PHILLIPS, CPS78_85, WAGEPAN, JTRAIN2 | 3 deney | 24 soru |
| 3 | Uygulama · Sezgi · Kendini sına | WAGE1, Tablo 3.2 (küçük örnek), JTRAIN2 | 3 deney | 24 soru |
| 4 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | 3 deney | 24 soru |
| 5 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | 3 deney | 24 soru |
| 6 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | 3 deney | 24 soru |
| 7 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | 3 deney | 24 soru |
| 8 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | 3 deney | 24 soru |
| 9 | Uygulama · Sezgi · Kendini sına | WAGE1 | 3 deney | 24 soru |
| 10 | Uygulama · Sezgi · Kendini sına | WAGE1 | 3 deney | 24 soru |
| 11–12 | Eski sayfa (taşınacak) | — | — | — |

## Teknik yapı

- Python 3.12
- Streamlit
- pandas, NumPy, SciPy
- statsmodels
- Plotly
- Wooldridge veri paketi
- pytest ve Streamlit AppTest
- Üretilen kod: Python (`wooldridge`, pandas, statsmodels, matplotlib) ve temel R (tek paket: `wooldridge`)

İki dilde aynı sayı sözleşmesi: veri üzerindeki deterministik hesaplar (betimsel özet, EKK) iki dilde basılı basamakta aynı sayıyı verir; Sezgi deneylerinde Python kodu uygulamanın sayılarını birebir üretir, R farklı rastgele sayı üreteci kullandığı için yalnız dağılımda aynıdır.

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

`requirements-dev.txt` testlerin ihtiyaç duyduğu paketleri de kurar (üretilen Python kodu için matplotlib). Üretilen R kodunu çalıştıran testler `Rscript` ve `wooldridge` R paketini ister (`install.packages("wooldridge")`). `Rscript` önce `RSCRIPT` ortam değişkeninde, sonra PATH'te, Windows'ta sonra standart R kurulum klasörlerinde (`Program Files\R`, `AppData\Local\Programs\R`; en yeni sürüm) aranır. İkisinden biri yoksa R testleri atlanır; nedeni `pytest -rs` ile görülür.

Sürüm adayı öncesinde otomatik testlere ek olarak Konu 00–12, metin ölçeği seçenekleri, soru düğmeleri, tablolar, metric kartları ve grafik eksenleri canlı Streamlit oturumunda kontrol edilmelidir.

## Veri kaynakları

Uygulama Wooldridge veri paketindeki başlıca `WAGE1`, `HPRICE1`, `PHILLIPS`, `CPS78_85`, `WAGEPAN` ve `JTRAIN2` veri setlerini kullanır (kitabın 7. baskısının verisi). Yeni yapıdaki konular veriyi ve Türkçe değişken etiketlerini `core/wooldridge_data.py` üzerinden yükler; eski sayfalar `core/data_registry.py` kullanır. Veri dosyası depoda tutulmaz: uygulama ve üretilen Python kodu `wooldridge` Python paketinden, üretilen R kodu `wooldridge` R paketinden okur; iki paketin aynı veriyi verdiği testle denetlenir.

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
