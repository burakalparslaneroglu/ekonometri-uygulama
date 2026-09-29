# Mimari

## Yeni yapı: tek tanım, iki dil (Konu 0–6)

Konu 0–6 üç sekmelidir: **Uygulama**, **Sezgi**, **Kendini sına**. Bir konunun bütün hesabı ve metni Streamlit'ten
bağımsız tanımlardan gelir; arayüz yalnız bu tanımları gösterir.

- `core/wooldridge_data.py`: Wooldridge veri setlerini `wooldridge` Python paketinden yükler (önbellekli, bağımsız
  kopya); her değişkenin Türkçe etiketi ve birimi, veri setinin gözlem birimi, yapısı ve üretim biçimi. Veri depoda
  tutulmaz.
- `core/labs/spec.py`: işlem türleri (`LoadWooldridge`, `Describe`, `GroupStats`, `OLS`, `ShowModel`,
  `RegressionTable`, `PanelSummary`, `ScatterPlot`, `BarChart`, ...), `LabSpec` / `LabStep` / `Check` / `NoteRef`.
  Etkileşimli adımlar `interactive_step(build, controls, uses)` ile yazılır: denetimler (`Choice`, `MultiChoice`,
  `NumberChoice`) adımın işlemlerini seçime göre kurar. `LabSpec.resolve(choices)` seçimi sonraki adımlara geçirir;
  notlardan farklı bir seçimin etkilediği adımların kontrolleri gösterilmez. Bağımlılık işlem düzeyinde izlenir
  (`tainted_writes`): değişen adımda notlardakiyle birebir aynı ve seçime bağlı bir adı okumayan işlemler (ör. veriyi
  yeniden yükleme; çerçeveyi baştan kuran `LoadWooldridge`, `InlineData`, `NewSample` gibi işlemler eski çerçeveyi
  okumaz) çıktılarını notlardaki gibi kurar; yalnız farklı işlemlerin ve onları okuyan işlemlerin yazdığı
  adlar sonraki adımları notlardan ayırır. Bir veri çerçevesine seçime bağlı `Derive` yazmak bütün çerçeveyi seçime
  bağlı yapar; bu yüzden seçime bağlı türetmeler adımın kendi çerçevesinde ya da skalerle yapılır. İşaretlenmeyen
  adımların notlardaki sayıları vermeye devam ettiği her denetimin her tek değişikliği için sınanır
  (`tests/test_all_labs.py`).
- `core/labs/runner.py`: tanımı çalıştırır (`LabState`), kontrolleri değerlendirir. `core/labs/regression.py` EKK'yı
  (statsmodels) ve model niceliklerini, makale tipi tabloyu hesaplar. `core/labs/expr.py` dilden bağımsız ifade dilidir.
- `core/codegen/`: aynı tanımdan Python (`python_gen.py`) ve temel R (`r_gen.py`; tek paket `wooldridge`) kodu üretir.
  Model, tablo ve skaler adları iki dilde aynıdır. Veri üzerindeki deterministik hesaplar iki dilde basılı basamakta
  aynı sayıyı verir; Sezgi deneylerinde Python uygulamanın çekilişlerini birebir yapar, R yalnız dağılımda aynıdır.
  Seçilen spesifikasyonun betiği son satırında yalnız notlarla karşılaştırılan adımları anar (`closing_message`);
  `ShowModel(stars=False)` R özetini anlamlılık yıldızları olmadan yazdırır (yıldızlar Konu 7'den önce gösterilmez);
  `ShowModel(columns=("coef",))` ekranda yalnız katsayıları gösterir, üretilen kod tam çıktıyı yazdırır ve yorumda
  okunan alanları ve standart hata, t, p, güven aralığının Konu 7'de yorumlanacağını belirtir.
  `RegressionTable(standard_errors=False, r2=False)` standart hata ya da R² satırı olmayan makale tablosudur (Konu 3–4);
  `adj_r2=True` düzeltilmiş R² satırını ekler, `term_decimals` tek bir terimin basamağını değiştirir (ör. arsa
  katsayısı 0,002068; üretilen kod bu satırları ayrıca aynı basamakla yazdırır). `ScatterPlot(means=...)` aynı x
  değerindeki gözlemlerin ortalamasını (koşullu ortalamanın örneklem karşılığı) çizer. Konu 5–6 ile eklenen işlemler:
  `Residuals` (modelin artıkları, yalnız modelin bütün gözlemleriyle tahmin edildiği çerçeveye), `SummaryTable`
  (Monte Carlo sonuç tablolarını satır satır özetler; `column_decimals`), `CopyFrame` (bağımsız kopya; Python
  `.copy()`, R atama), `ScalarTable(heading, value)` ve `JoinColumns(heading)` sütun başlıkları. Kod yorumundaki
  "bu adımda yalnız … okunur" listesi `Generator.fields_read` ile aynı adımda okunan model bilgisini (R², düzeltilmiş
  R²) de sayar; Konu 7'den önce standart hata, t, p ve güven aralığının Konu 7'de, F istatistiğinin Konu 8'de
  yorumlanacağını belirtir. Gövdesinde EKK olan Monte Carlo betiklerine süre yorumu eklenir.
- `core/labs/runner.py` Monte Carlo döngüsünü, gövde uygunsa (`batchable`) toplu yoldan hesaplar: çekilişler aynı
  üreteç sırasıyla bir kerede yapılır, EKK tekrarlar üzerinde vektörel çözülür; sonuç ve üreteç durumu döngüyle bit
  düzeyinde aynıdır (R² için toplam kareler statsmodels'daki gibi hesaplanır), eksik değerde döngüye dönülür. Döngü
  adları (`tekrar`, `sonuclar`, `rng`) Monte Carlo içinde ad olarak kullanılamaz (`MONTE_CARLO_RESERVED`).
- `core/labs/konu00.py` … `konu06.py`: Uygulama laboratuvarları (notlardaki Bölüm 0 araç kutusu, §1.6, Bölüm 2
  uygulamaları, Bölüm 3–4'ün çözümlü örnekleri ve Bölüm 5–6'nın WAGE1/HPRICE1 uygulamaları: çoklu model, artıkların
  artıklara regresyonu, makale tablosu, eksik değişken ayrıştırması, tam bağlantı ve VIF); notlarda basılı her sayı
  bir `Check`. Konu 5–6'da kontrol değişkenleri kukla olmayan sayısal değişkenlerdir (kuklalar Konu 10). `consistency_notes` notlarla eski sürüm arasındaki
  kalıcı açıklamaları tutar.
- `core/labs/sezgi.py`, `sezgi_konu00.py` … `sezgi_konu06.py`: Sezgi deneyleri (`SimExperiment`):
  DGP LaTeX satırları, kaydırıcılar, bilinen gerçek ile tahminin karşılaştırması, kaydırıcı değerine göre değişen
  metin; tohum 305. `tables` girdisi `(ad, başlık[, bağlı parametreler])` biçimindedir; notlardaki numaralı bir tablo
  seçilen ayarlarla değiştiyse başlığı "seçtiğiniz ayarlarla" etiketini alır (`table_title`).
- `core/quiz/`: `model.py` (soru türleri ve notlandırma: Türkçe harf, tire ve binlik ayırıcı farklarını yok sayar),
  `expression.py` (formül güvenli biçimde okunur; `eval` yok; LaTeX alışkanlıkları (`\frac`, `\sqrt`, `\cdot`,
  `\ln`) ve yazılan sol taraf (`r = …`) okunur; eşdeğerlik rastgele noktalarda sayısal olarak denetlenir;
  `bar_aliases`, `beta_aliases`, `beta_hat_aliases` X̄, β₀, β̂₁ gibi yazımları sembollere bağlar; tanımsız bir ad,
  harf ya da alt çizgi farkıyla yalnız tek bir sembole uyuyorsa o sembol okunur: `B_1` → `b1`),
  `konu00.py` … `konu06.py` (24'er soru), `registry.py`.
- `topics/lab_ui.py`, `sim_ui.py`, `quiz_ui.py`: üç sekmenin gösterimi; `topics/shared.py` konu başlığı ve widget
  durumunun konu geçişlerinde korunması (`keep_widget_state`). `core/charts.py` Plotly grafikleri Türkçe sayı biçimiyle
  çizer. `lab_ui.upstream_steps`, bir adımın uyarısında yalnız o adımı gerçekten değiştiren önceki seçimleri anar
  (her önceki seçim tek başına `LabSpec.resolve` ile denenir). Sezgi sekmesi `ShowFrame` tablolarını da gösterir.
  Adım düğmeleri satırın tamamını kullanır, önceki/sonraki düğmeleri altındadır ve uçlarda devre dışıdır. Kesirli
  kaydırıcılar `lab_ui.decimal_slider` ile çizilir (`st.select_slider`; değer ondalık virgülle yazılır, çünkü
  `st.slider` biçimi yalnız noktayı bilir). `assets/styles.css` metrik etiketini üç noktayla kesmez, dar sütunda alt
  satıra geçirir ve aynı satırdaki metrik kutularını eşit yükseklikte tutar.
- Testler: `tests/test_konu00_content.py`, `tests/test_konu01_02_content.py`, `tests/test_konu03_04_content.py`,
  `tests/test_konu05_06_content.py` (veri,
  EKK formülleri, seçimlerin geçişi, uygulama ile üretilen Python'un 1e-12 düzeyinde eşitliği, R'nin çalışması, Sezgi
  kuramsal değerleri ve metinlerin kaydırıcı uçlarında doğruluğu, soru yazım çeşitleri), `tests/test_all_quizzes.py`,
  `tests/test_topic_contracts.py` (`MIGRATED_TOPICS`), `tests/test_app_smoke.py`.

Konu 7–12 aşağıda anlatılan eski sayfalarıyla çalışır; ikişerli bloklar hâlinde bu yapıya taşınır. Taşınan konunun eski
modülleri ve testleri aynı blokta kaldırılır (Konu 1–2 için `core/research_question_utils.py`,
`core/scenario_registry.py`, `core/data_structure_utils.py`, `core/group_comparison_utils.py` ve testleri; Konu 3–4
için `core/konu04_questions.py`, `tests/test_konu03_ui.py`, `tests/test_konu04_ui.py`, `tests/test_konu04_utils.py`,
`core/question_engine.py`'nin Konu 3 soru üreticisi ve `core/model_utils.py`'nin yalnız Konu 4 sayfasının kullandığı
kareler toplamı, birim dönüşümü ve fonksiyonel biçim yardımcıları; Konu 5–6 için `core/konu05_questions.py`,
`core/konu06_questions.py`, `core/multiple_regression_utils.py`, `core/assumption_diagnostics_utils.py` ve testleri,
`core/data_registry.py`'nin `konu05_model_specs` ve `konu06_model_specs` tanımları ve `core/model_utils.py`'nin artık
kullanılmayan basit EKK, tahmin ve artık yardımcıları kaldırıldı).

## Eski yapı (Konu 7–12)

Konu 07 uygulamanın ilk aktif çıkarım modülüdür. `core/regression_inference_utils.py` Streamlit'ten bağımsız ortak katmanda yalnızca geleneksel `nonrobust` EKK standart hatalarını, Student-t kritik değerlerini ve p-değerlerini, tek katsayı testlerini, güven aralıklarını, test–GA eşdeğerliğini, p/yıldız biçimlendirmesini, ölçeklemeyi ve vektörize benzetimleri sağlar. Heteroskedastisiteye dayanıklı standart hatalar Konu 12'ye, birlikte katsayı sınaması ve F testi Konu 08'e bırakılmıştır.

Uygulama, Streamlit arayüzü ile hesaplama mantığını ayıran küçük modüllerden oluşur.

- `app.py`: Ortak sayfa yapılandırması, ortak görsel iskelet ve konu yönlendirmesi.
- `topics/`: Her ders konusunun Streamlit görünümü.
- `core/data_registry.py`: Eski sayfaların Wooldridge kataloğu, değişken açıklamaları, veri yapısı metadata'sı ve izin verilen pedagojik eşleşmeler.
- `core/model_utils.py`: Eski sayfaların (Konu 10–11) öğrenci sayı biçimleri. Streamlit bağımlılığı yoktur.
- `core/question_engine.py`: Eski sayfaların ortak soru veri yapısı ve kararlı soru sırası.
- `core/session_utils.py`: Soru sırası ve cevap görünürlüğünün `st.session_state` içindeki anahtarlarını yönetir.
- `assets/styles.css`: Kurumdan bağımsız marka değişkenleri ve duyarlı görsel düzen.

## Veri akışı (eski sayfalar)

1. Konu modülü, `data_registry` üzerinden Wooldridge paketindeki veri setini önbellekli olarak yükler.
2. Konunun sabit model tanımları (`konu07_model_specs` …) ortak hesaplama modüllerine verilir.
3. Model çıktısı grafik, tablo ve soru motoru tarafından kullanılır.
4. Veri seti veya model değiştiğinde model kimliği değişir; `session_utils` önceki soru durumunu sıfırlar.
# Konu 08–09 ortak altyapısı

Konu 08 ve Konu 09 aynı geliştirme dalında yer alsa da ayrı `topics/`, soru ve test modülleridir. `core/joint_inference_utils.py`, yalnız geleneksel `nonrobust` kovaryans altında genel `Rβ=r` ortak F testi, iç içe modeller için SSR/R² F karşılaştırması, genel anlamlılık, tek kısıtta `F=t²`, F dağılım grafiği verisi ve NumPy batch büyük-örneklem benzetimini taşır.

`core/functional_form_utils.py`, ölçekleme, standartlaştırma, log-yüzde, karesel marjinal etki/dönüm noktası, WAGE1 M1–M4 ve merkezleme iş mantığını taşır. Karesel terimlerin ortak testi kendi F formülünü yazmaz; Konu 08'in `nested_exclusion_f_test` aracını kullanır. Konu ekranlarındaki pahalı sabit model ve benzetim sonuçları `st.cache_data` ile önbelleklenir.

Konu 09 model seçimini tek bir metriğe indirmez; teori, işaret, veri aralığı, basitlik, SSR/R²/düzeltilmiş R² ve ortak F birlikte sunulur. Merkezleme fitted değerleri, artıklar, SSR ve R²'yi değiştirmez; içselliği çözmez. Dayanıklı ortak testler Konu 12'ye, kukla değişkenler Konu 10'a bırakılır. Soru durumları `session_utils` içinde konu bazlı anahtarlarla tutulur ve konu geçişinde sıfırlanır.

`core/ui_preferences.py`, konu durumundan bağımsız metin ölçeği seçeneklerini ve CSS/Plotly yazı boyutlarını üretir. Seçim `text_scale_label` ile session state'te korunur; `app.py` bunu tek CSS değişkenine uygular. Tüm konu soru blokları birincil “Cevabı göster” ve ikincil “Yeni soru” düğme sistemini kullanır.

# Konu 10–12 ortak altyapısı

Konu 10, 11 ve 12 ayrı sayfa, saf hesaplama, soru ve test modülleridir. `core/categorical_regression_utils.py` 0/1 grup özeti, referans kodlama değişmezliği, kukla tuzağı rank denetimi, log-kukla yüzde dönüşümü ve Konu 08'in geleneksel ortak F testini kullanır. Türetilmiş WAGE1 sütunları girdiyi değiştirmeyen kopyada doğrulanır.

`core/interaction_utils.py`, Konu 10'un referans/kategori mantığı üzerinde D×X modellerinin iki grup doğrusunu, koşullu lineer bileşim farkını ve tam kovaryansla standart hatasını kurar. Merkezleme fitted değer, artık, SSR ve R² değişmezliğini ayrıca denetler. Konu 11 yalnız ayrıntılı ders notunun desteklediği etkileşim ve grup farkı kapsamındadır; LPM eklenmemiştir.

`core/robust_inference_utils.py` mevcut `OLSInferenceResult` ve Konu 08'in nonrobust anlamını değiştirmez. Bunun yerine aynı EKK katsayıları için ayrı HC0–HC3 sonuç katmanı, BP/White tanıları, robust Wald/F ortak testi, saf grafik verisi ve sabit-seed vektörize kapsama benzetimi sunar. Kovaryans türü öğrenci ekranında açıkça etiketlenir; robust standart hata katsayıyı, yanlılığı veya nedensel tasarımı düzeltmez.

Video incelemesi sonrasındaki öğretim katmanı genişletmesinde `core/ui_components.py`, eski sayfaların (Konu 07–12) soru eylemlerini tek kompakt primary/secondary bileşeninde toplar. CSS yalnız `question_actions_<topic>` anahtarlı kapsayıcıyı hedefler ve dar ekranda sarmalanır. Konu 10'a sayısal–kukla kodlama, kategori karşıtlığı ve rank önizlemesi; Konu 11'e grup doğrusu ve koşullu fark ızgaraları; Konu 12'ye BP/White manuel yardımcı-regresyon ayrıntıları, tüm HC türlerinin karşılaştırması ve tanı/benzetim grafik verileri eklenmiştir.

Konu 08'deki custom laboratuvarı serbest metin ayrıştırmaz. Kullanıcı kontrollü R satırları ve r sağ taraflarını girer; `validate_restriction_system` rank(R), rank([R|r]), sıfır bilgi satırı, bağımlı satır ve tutarsız sistemi görünür kılar. Geçersiz sistemlerde test çalışmaz. `classify_restriction_system`, yalnız basit sıfır-dışlama kısıtlarında nested SSR/R² karşılaştırmasını açar; eşitlik, sıfır dışı ve doğrusal birleşim kısıtları genel `Rβ=r` matrix F ile sınanır.
