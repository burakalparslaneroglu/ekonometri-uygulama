# Mimari

## Yapı: tek tanım, iki dil (Konu 0–12)

Konu 0–12 üç sekmelidir: **Uygulama**, **Sezgi**, **Kendini sına**. Bir konunun bütün hesabı ve metni Streamlit'ten
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
  Konu 7–8 ile eklenen çıkarım işlemleri: `CoefficientTable` (katsayı, standart hata, t, p ve yüzde `100·level` güven
  aralığı; statsmodels `conf_int(alpha)`, R `confint(level)`), `JointTest` (dışlama kısıtlarının F testi; Python
  `f_test`, R kısıtlı modelle `anova`), `HypothesisPlot` (t ya da F dağılımı, reddetme bölgesi, p-değeri alanı; eksen
  kuralı `core/labs/inference.py`'de, üç dilde aynı: t'de h = max(4, min(|t| + 1, 6)), F'de
  üst = max(2,4·kritik, min(1,15·F, 6·kritik)); gözlenen değer eksenin dışındaysa açıklamada yazılır),
  `CoefficientPlot`, `IntervalPlot(reference=...)` (tekrarlı örneklemedeki aralıklar; isteğe bağlı noktalı başvuru
  çizgisi, ör. sıfır). `ModelValue` model niceliklerini (`coef`, `se`, `t`, `p`, `ci_low`, `ci_high`; `nobs`, `r2`,
  `adj_r2`, `f`, `f_p`, `ssr`, `df_resid`) skalere yazar; `Scalar(p_value=True)` p-değeri biçimiyle gösterilir.
  İfade dili t ve F fonksiyonlarını (`tcdf`, `tsf`, `tinv`, `fsf`, `finv`; Python scipy, R `pt`/`qt`/`pf`/`qf`) ve
  `power`, `maximum`, `compare` işlemlerini içerir; ≥ 10¹⁶ tam sayılar kodda bilimsel gösterimle yazılır (`1e+41`).
  `RegressionTable(extra=..., extra_decimals=...)` terimlerden sonra skalerlerden ek satırlar (ortak F, p-değeri,
  SSR) ekler; `terms=()` yalnız ek satırlı karşılaştırma tablosudur. `ScatterPlot` bir veri çerçevesini ya da Monte
  Carlo sonuç tablosunu çizer. p-değerleri gösterim basamağında 0'a ya da 1'e yuvarlanıyorsa "< 0,001" ya da
  "> 0,999" yazılır (`core/charts.p_text`).
  Konu 9–10 ile eklenenler: `GroupSummary(labels, heading)` grup değerlerini adlandırır (ör. 0 → "Erkek"; Python
  `.rename(index=...)`, R `rownames`); `JoinColumns(column_decimals, row_decimals, p_columns)` sütun ya da satır
  başına basamak (satır önceliklidir) ve p-değeri biçimli sütunlar; `RegressionTable(r2_decimals=...)` R² satırlarının
  basamağı, ek satırda boş ad ("") o sütunun hücresini "—" yapar (Python `np.nan`, R `NA`); `ScatterPlot(curves=...)`
  saçılımın üstüne başka çerçeveden eğriler (ör. tahmin edilen karesel eğri); `LineChart(vlines=...)` skalerlerden
  etiketli dikey çizgiler (ör. dönüm noktası; `series` ile birlikte kullanılamaz); `CoefficientPlot(percent=True)`
  log bağımlı değişkende katsayı ve güven sınırlarını tam yüzdeye çevirir: 100·(exp(değer) − 1).
  Konu 11–12 ile eklenenler: `OLS` etkileşim terimini `a:b` yazar (iki dilde aynı ad; ana etkiler ve etkileşim art
  arda geliyorsa formül notlardaki gibi `a * b`; etkileşimin değişkenleri formülde ilk görünme sırasıyla yazılır,
  çünkü R terimi bu sırayla adlandırır). `OLS(cov_type=...)` heteroskedastisiteye dayanıklı HC0–HC3 kovaryansıdır:
  katsayılar aynı kalır; standart hata, t, p, güven aralığı ve ortak test t ve F dağılımıyla (n − k − 1 serbestlik
  derecesi) hesaplanır (Python `fit(cov_type=..., use_t=True)`, R açık sandviç formülü; HC1 = HC0·n/(n − k − 1),
  HC2 ve HC3 kaldıraçla düzeltir). `JointTest` dayanıklı modelde Wald testidir ve F biçiminde raporlanır:
  F = b'V⁻¹b / q. `HeteroskedasticityTest` Breusch–Pagan (Koenker n·R²) ve White testleridir (Python
  `het_breuschpagan`; White'ta yardımcı regresyon `numpy` ile açıkça kurulur, üretilen kodda `white_testi`; statsmodels
  `het_white` kullanılmaz, çünkü rank `assert`'ı kukla varken sayısal gürültüyle AssertionError verebilir; R yardımcı
  regresyonu `lm()` ile açıkça kurar). `ModelValue(cov_type=...)` aynı EKK tahmininin niceliklerini başka bir
  kovaryansla okur. `ModelValue`, `Scalar`, `JointTest` ve
  `HeteroskedasticityTest` `shown=False` ile ölçü kutusunda gösterilmez (sayı aynı adımda bir tabloda ya da çıktıda
  görünüyorsa); `JointTest(p_decimals=...)` p-değerinin basamağı; `JoinColumns(term_rows=True)` satırları
  "Türkçe etiket · terim" diye yazar; `CoefficientTable(exact=True)` katsayıyı notlardaki çıktı gibi tam basamakla
  yazar. Monte Carlo toplu yolu dayanıklı standart hataları da tekrarlar üzerinde vektörel hesaplar.
- `core/labs/runner.py` Monte Carlo döngüsünü, gövde uygunsa (`batchable`) toplu yoldan hesaplar: çekilişler aynı
  üreteç sırasıyla bir kerede yapılır, EKK tekrarlar üzerinde vektörel çözülür; sonuç ve üreteç durumu döngüyle bit
  düzeyinde aynıdır (R² için toplam kareler statsmodels'daki gibi hesaplanır), eksik değerde döngüye dönülür. Döngü
  adları (`tekrar`, `sonuclar`, `rng`) Monte Carlo içinde ad olarak kullanılamaz (`MONTE_CARLO_RESERVED`).
- `core/labs/konu00.py` … `konu12.py`: Uygulama laboratuvarları (notlardaki Bölüm 0 araç kutusu, §1.6, Bölüm 2
  uygulamaları, Bölüm 3–4'ün çözümlü örnekleri, Bölüm 5–6'nın WAGE1/HPRICE1 uygulamaları: çoklu model, artıkların
  artıklara regresyonu, makale tablosu, eksik değişken ayrıştırması, tam bağlantı ve VIF; Bölüm 7–8'in çıkarım
  örnekleri: t istatistiği, kritik değerler, p-değeri, güven aralığı, tek taraflı test, çıktı ve makale tablosu,
  iktisadi önem, raporlama, ortak F testi, kısıtlı–kısıtsız model, genel F, F = t²; Bölüm 9'un ölçü birimi,
  standartlaştırma, tam yüzde, karesel model, dönüm noktası, merkezleme, model karşılaştırması ve makale tablosu;
  Bölüm 10'un kukla regresyonu, kontrollü fark, paralel doğrular, kukla katsayısında çıkarım, çok kategorili
  değişken ve referans değişikliği, kukla tuzağı, log modelinde tam yüzde, bölge ve endüstri kuklalarının ortak
  testleri ve makale tablosu; Bölüm 11'in additif ve etkileşimli modelleri, grup denklemleri, dört yapı, koşullu
  fark ve merkezleme, WAGE1 ve HPRICE1 uygulamaları, eğim ve iki doğru testleri ve makale tablosu; Bölüm 12'nin artık
  grafikleri, Breusch–Pagan ve White testleri, HC0–HC3, geleneksel ve dayanıklı çıkarım, log modeli, makale tablosu,
  dayanıklı ortak test ve WAGE1 karşılaştırması); notlarda basılı her sayı bir `Check`. Konu 5–6'da kontrol değişkenleri kukla olmayan sayısal değişkenlerdir (kuklalar Konu 10). `consistency_notes` notlarla eski sürüm arasındaki
  kalıcı açıklamaları tutar.
- `core/labs/sezgi.py`, `sezgi_konu00.py` … `sezgi_konu12.py`: Sezgi deneyleri (`SimExperiment`):
  DGP LaTeX satırları, kaydırıcılar, bilinen gerçek ile tahminin karşılaştırması, kaydırıcı değerine göre değişen
  metin; tohum 305. `tables` girdisi `(ad, başlık[, bağlı parametreler])` biçimindedir; notlardaki numaralı bir tablo
  seçilen ayarlarla değiştiyse başlığı "seçtiğiniz ayarlarla" etiketini alır (`table_title`).
- `core/quiz/`: `model.py` (soru türleri ve notlandırma: Türkçe harf, tire ve binlik ayırıcı farklarını yok sayar),
  `expression.py` (formül güvenli biçimde okunur; `eval` yok; LaTeX alışkanlıkları (`\frac`, `\sqrt`, `\cdot`,
  `\ln`) ve yazılan sol taraf (`r = …`) okunur; eşdeğerlik rastgele noktalarda sayısal olarak denetlenir;
  `bar_aliases`, `beta_aliases`, `beta_hat_aliases` X̄, β₀, β̂₁ gibi yazımları sembollere bağlar; tanımsız bir ad,
  harf ya da alt çizgi farkıyla yalnız tek bir sembole uyuyorsa o sembol okunur: `B_1` → `b1`; çok terimli üs
  parantezle korunur: `e^{a - c s}` → `exp(a - c·s)`; `s\sqrt{…}` ve `b_1c` çarpım okunur),
  `konu00.py` … `konu12.py` (24'er soru), `registry.py`.
- `topics/lab_ui.py`, `sim_ui.py`, `quiz_ui.py`: üç sekmenin gösterimi; `topics/shared.py` konu başlığı ve widget
  durumunun konu geçişlerinde korunması (`keep_widget_state`). `core/charts.py` Plotly grafikleri Türkçe sayı biçimiyle
  çizer. `lab_ui.upstream_steps`, bir adımın uyarısında yalnız o adımı gerçekten değiştiren önceki seçimleri anar
  (her önceki seçim tek başına `LabSpec.resolve` ile denenir). Sezgi sekmesi `ShowFrame` tablolarını da gösterir.
  Adım düğmeleri satırın tamamını kullanır, önceki/sonraki düğmeleri altındadır ve uçlarda devre dışıdır. Kesirli
  kaydırıcılar `lab_ui.decimal_slider` ile çizilir (`st.select_slider`; değer ondalık virgülle yazılır, çünkü
  `st.slider` biçimi yalnız noktayı bilir). `assets/styles.css` metrik etiketini üç noktayla kesmez, dar sütunda alt
  satıra geçirir ve aynı satırdaki metrik kutularını eşit yükseklikte tutar.
- Testler: `tests/test_konu00_content.py`, `tests/test_konu01_02_content.py`, `tests/test_konu03_04_content.py`,
  `tests/test_konu05_06_content.py`, `tests/test_konu07_08_content.py`, `tests/test_konu09_10_content.py`,
  `tests/test_konu11_12_content.py` (veri,
  EKK formülleri, seçimlerin geçişi, uygulama ile üretilen Python'un 1e-12 düzeyinde eşitliği, R'nin çalışması, Sezgi
  kuramsal değerleri ve metinlerin kaydırıcı uçlarında doğruluğu, soru yazım çeşitleri), `tests/test_all_quizzes.py`,
  `tests/test_topic_contracts.py` (`MIGRATED_TOPICS`), `tests/test_app_smoke.py`.

Bütün konular bu yapıdadır. Taşınan her konunun eski modülleri ve testleri aynı blokta kaldırıldı (Konu 1–2 için
`core/research_question_utils.py`,
`core/scenario_registry.py`, `core/data_structure_utils.py`, `core/group_comparison_utils.py` ve testleri; Konu 3–4
için `core/konu04_questions.py`, `tests/test_konu03_ui.py`, `tests/test_konu04_ui.py`, `tests/test_konu04_utils.py`,
`core/question_engine.py`'nin Konu 3 soru üreticisi ve `core/model_utils.py`'nin yalnız Konu 4 sayfasının kullandığı
kareler toplamı, birim dönüşümü ve fonksiyonel biçim yardımcıları; Konu 5–6 için `core/konu05_questions.py`,
`core/konu06_questions.py`, `core/multiple_regression_utils.py`, `core/assumption_diagnostics_utils.py` ve testleri,
`core/data_registry.py`'nin `konu05_model_specs` ve `konu06_model_specs` tanımları ve `core/model_utils.py`'nin artık
kullanılmayan basit EKK, tahmin ve artık yardımcıları; Konu 7–8 için `core/konu07_questions.py`,
`core/konu08_questions.py` ve testleri, `core/data_registry.py`'nin `konu07_model_specs` ve `konu08_model_specs`
tanımları, `core/regression_inference_utils.py`'nin tek katsayı testi, aralık, ölçekleme ve benzetim yardımcıları ile
`core/joint_inference_utils.py`'nin genel F, F = t², kısıt sınıflama, F grafiği verisi ve büyük örneklem benzetimi;
Konu 9–10 için `core/konu09_questions.py`, `core/konu10_questions.py`, `core/functional_form_utils.py`,
`core/categorical_regression_utils.py` ve testleri, `core/data_registry.py`'nin `konu09_model_specs` ve
`konu10_model_specs` tanımları, `core/joint_inference_utils.py`'nin iç içe model F karşılaştırması
(`nested_exclusion_f_test`), `core/regression_inference_utils.py`'nin anlamlılık yıldızı, `core/ui_preferences.py`'nin
Plotly yazı boyutu ve `core/session_utils.py`'nin cevabı gösterme yardımcıları; Konu 11–12 için eski
sayfalar, `core/konu11_questions.py`, `core/konu12_questions.py`, `core/interaction_utils.py`,
`core/robust_inference_utils.py`, `core/joint_inference_utils.py`, `core/regression_inference_utils.py`,
`core/model_utils.py`, `core/question_engine.py`, `core/data_registry.py`, `core/ui_components.py`,
`core/session_utils.py`, `core/formatting.py` ve testleri).

## Kabuk ve ortak modüller

- `app.py`: Ortak sayfa yapılandırması, ortak görsel iskelet ve konu yönlendirmesi.
- `topics/`: Her ders konusunun Streamlit görünümü (üç sekme).
- `assets/styles.css`: Kurumdan bağımsız marka değişkenleri ve duyarlı görsel düzen.

`core/ui_preferences.py`, konu durumundan bağımsız metin ölçeği seçeneklerini ve CSS yazı boyutlarını üretir. Seçim `text_scale_label` ile session state'te korunur; `app.py` bunu tek CSS değişkenine uygular.
