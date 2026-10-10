# İKT 305 Ekonometri I — Etkileşimli Streamlit Uygulaması

İzmir Bakırçay Üniversitesi İKT 305 Ekonometri I dersi için hazırlanmış, Türkçe ve etkileşimli bir öğretim uygulamasıdır. Uygulama; ekonometrik kavramları gerçek veri setleri, hesaplama laboratuvarları, grafikler, tablolar ve kısa uygulama sorularıyla destekler.

Uygulama çalışma zamanında bir büyük dil modeli veya dış API kullanmaz. Wooldridge veri setleri `wooldridge` Python paketi üzerinden yerel olarak yüklenir.

Her konu **Uygulama**, **Sezgi** ve **Kendini sına** sekmelerinden oluşur. Uygulama sekmesi notlardaki laboratuvarın adımlarını izler ve notlarda basılı her sayıyı gerçek veriyle yeniden üretir; öğrenci bir adımda farklı bir spesifikasyon seçebilir (ör. açıklayıcı değişken), seçim yalnız ona bağlı sonraki adımlara geçer ve notlardaki modelle yan yana gösterilir. Her adımın ve her Sezgi deneyinin Python ve R kodu aynı tanımdan üretilir.

Sunumlardan doğru konu, sekme, deney veya laboratuvar adımı doğrudan URL ile açılabilir;
ör. `/?konu=03&sekme=sezgi&deney=2`. URL'nin başlangıç ayarları sonraki kullanıcı seçimlerini sıfırlamaz.
Parametreler ve veri kaynağı davranışı için [Sunum bağlantıları](docs/SUNUM_BAGLANTILARI.md) belgesine bakın.

Bütün konularda (0–12) Uygulama sekmesinin üstünde bir **veri kaynağı** seçimi vardır. Varsayılan **Notlardaki örnek**tir (yukarıdaki gibi). **Alternatif örnek** aynı adımları başka Wooldridge verileriyle yapar (WAGE2, OKUN, KIELMC, CRIME4, BEAUTY; Konu 2'nin deney adımında ve Konu 3'ün sıfır–bir değişken adımında veri üretim süreci belgelenmiş kurgusal bir iş arama programı). **Kendi verini yükle** aynı adımları öğrencinin Excel (.xlsx) ya da CSV dosyasıyla yapar: öğrenci sütunları rollere seçer (ör. sonuç ve açıklayıcı değişken), rolü seçilmeyen adım neye ihtiyacı olduğunu yazar. İki ek kaynakta kontrollerin beklenen değerleri uygulamanın kendi hesabıdır; indirilen Python ve R kodu bu değerleri yeniden üretir ve kendi verinde dosyayı betikle aynı klasörden okur. Yüklenen dosya yalnız oturumun belleğinde işlenir; diske, ortak önbelleğe ya da günlüğe yazılmaz.

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

| Konu | Yapı | Uygulama verisi (notlardaki örnek) | Alternatif örnek | Sezgi | Kendini sına |
|---|---|---|---|---|---|
| 0 | Uygulama · Sezgi · Kendini sına | Tablo 0.1–0.3 (küçük örnekler), WAGE1 | WAGE2, OKUN | 3 deney | 24 soru |
| 1 | Uygulama · Sezgi · Kendini sına | WAGE1 (§1.6) | WAGE2 | 3 deney | 24 soru |
| 2 | Uygulama · Sezgi · Kendini sına | WAGE1, PHILLIPS, CPS78_85, WAGEPAN, JTRAIN2 | WAGE2, OKUN, KIELMC, CRIME4, kurgusal deney | 3 deney | 24 soru |
| 3 | Uygulama · Sezgi · Kendini sına | WAGE1, Tablo 3.2 (küçük örnek), JTRAIN2 | WAGE2, kurgusal deney | 3 deney | 24 soru |
| 4 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | WAGE2, KIELMC (1978) | 3 deney | 24 soru |
| 5 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | WAGE2, KIELMC (1978) | 3 deney | 24 soru |
| 6 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | WAGE2, KIELMC (1978) | 3 deney | 24 soru |
| 7 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | WAGE2, KIELMC (1978) | 3 deney | 24 soru |
| 8 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | WAGE2, KIELMC (1978) | 3 deney | 24 soru |
| 9 | Uygulama · Sezgi · Kendini sına | WAGE1 | WAGE2 | 3 deney | 24 soru |
| 10 | Uygulama · Sezgi · Kendini sına | WAGE1 | BEAUTY | 3 deney | 24 soru |
| 11 | Uygulama · Sezgi · Kendini sına | WAGE1, HPRICE1 | BEAUTY, KIELMC (1978) | 3 deney | 24 soru |
| 12 | Uygulama · Sezgi · Kendini sına | HPRICE1, WAGE1 | KIELMC (1978), WAGE2 | 3 deney | 24 soru |

Alternatif örnek ve **Kendi verini yükle** bütün konularda vardır. Notlarda HPRICE1 kullanılan konut adımları (Konu 4–8, 11, 12) KIELMC'nin 1978 satışlarıyla (179 konut, tek dönemlik yatay kesit) yapılır; Konu 10–11'in kukla değişken adımları BEAUTY'dir (Hamermesh ve Biddle, 1994; 1260 çalışan, saatlik ücret). "Kurgusal deney" Konu 2'nin iş arama programıdır (kurayla atama; veri üretim süreci belgelidir, gerçek etki bilinir). Konu 3–12'de kendi verinde sonuç ve temel açıklayıcı değişken zorunludur, en çok üç ek sayısal değişken seçilir; seçilen sütunlardan birinde boş hücresi olan satırlar çıkarılır, böylece bütün modeller aynı gözlemlerle kurulur. Konu 9–11'de modeller notlardaki gibi log sonuçla kurulur: sonucun bütün değerleri pozitif olmalıdır. Konu 10'da iki kategorili değişken ile çok kategorili değişkenden (3–15 kategori) en az biri, Konu 11'de iki kategorili değişken zorunludur. Konu 12'de düzey modeli her zaman, log modeli sonuç ve temel açıklayıcı pozitifse kurulur.

## Teknik yapı

- Python 3.12
- Streamlit 1.56 veya üstü (`st.segmented_control(required=...)`, `st.file_uploader(max_upload_size=...)`)
- pandas, NumPy, SciPy
- statsmodels
- Plotly
- Wooldridge veri paketi
- openpyxl (yüklenen Excel dosyalarını okumak için)
- pytest ve Streamlit AppTest
- Üretilen kod: Python (`wooldridge`, pandas, statsmodels, matplotlib) ve temel R (tek paket: `wooldridge`). Kendi
  verinde Excel dosyası Python'da `openpyxl` ile, R'de `readxl` ile okunur; R'nin tek ek paketi budur ve yalnız Excel
  dosyası yüklendiğinde gerekir (CSV temel R ile okunur).

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

`requirements-dev.txt` testlerin ihtiyaç duyduğu paketleri de kurar (üretilen Python kodu için matplotlib). Üretilen R kodunu çalıştıran testler `Rscript` ve `wooldridge` R paketini ister (`install.packages("wooldridge")`). `Rscript` önce `RSCRIPT` ortam değişkeninde, sonra PATH'te, Windows'ta sonra standart R kurulum klasörlerinde (`Program Files\R`, `AppData\Local\Programs\R`; en yeni sürüm) aranır. İkisinden biri yoksa R testleri atlanır; nedeni `pytest -rs` ile görülür. R testleri R'yi İngilizce iletilerle çalıştırır (`LANGUAGE=en`; uyarı denetimi çeviriye bağlı kalmaz). Yerel ayar Linux'ta `C.UTF-8`, macOS'ta `en_US.UTF-8`, Windows'ta sistemin yerel ayarıdır (Windows'ta `C.UTF-8` yoktur). Üretilen R kodunun çıktısı C, Türkçe ve İngilizce yerel ayarda aynıdır. Kendi verinin Excel dosyasını R'de okuyan testler `readxl` paketini de ister (`install.packages("readxl")`); paket yoksa yalnız bu testler atlanır.

Sürüm adayı öncesinde otomatik testlere ek olarak Konu 00–12, metin ölçeği seçenekleri, adım ve deney düğmeleri, tablolar, metric kartları ve grafik eksenleri canlı Streamlit oturumunda kontrol edilmelidir.

## Veri kaynakları

Uygulama Wooldridge veri paketindeki başlıca `WAGE1`, `HPRICE1`, `PHILLIPS`, `CPS78_85`, `WAGEPAN` ve `JTRAIN2` veri setlerini, alternatif örneklerde ayrıca `WAGE2`, `OKUN`, `KIELMC` (Konu 4–12'de yalnız 1978 satışları), `CRIME4` ve `BEAUTY` veri setlerini kullanır (kitabın 7. baskısının verisi). Konu 2'nin alternatif deney adımındaki ve Konu 3'ün sıfır–bir değişken adımındaki iş arama programı kurgusaldır (`core/labs/kurgusal_veri.py`; veri üretim süreci modül belgesinde yazılıdır, tohum 305). Kendi veri seçeneğinin örnek dosyaları da bu kurgusal veriden üretilir. Konular veriyi ve Türkçe değişken etiketlerini `core/wooldridge_data.py` üzerinden yükler. Veri dosyası depoda tutulmaz: uygulama ve üretilen Python kodu `wooldridge` Python paketinden, üretilen R kodu `wooldridge` R paketinden okur; iki paketin aynı veriyi verdiği testle denetlenir.

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
