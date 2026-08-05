# Mimari

Konu 07 uygulamanın ilk aktif çıkarım modülüdür. `core/regression_inference_utils.py` Streamlit'ten bağımsız ortak katmanda yalnızca geleneksel `nonrobust` EKK standart hatalarını, Student-t kritik değerlerini ve p-değerlerini, tek katsayı testlerini, güven aralıklarını, test–GA eşdeğerliğini, p/yıldız biçimlendirmesini, ölçeklemeyi ve vektörize benzetimleri sağlar. Heteroskedastisiteye dayanıklı standart hatalar Konu 12'ye, birlikte katsayı sınaması ve F testi Konu 08'e bırakılmıştır.

Uygulama, Streamlit arayüzü ile hesaplama mantığını ayıran küçük modüllerden oluşur.

- `app.py`: Ortak sayfa yapılandırması, ortak görsel iskelet ve konu yönlendirmesi.
- `topics/`: Her ders konusunun Streamlit görünümü. `konu05_coklu_regresyon.py`, sabit WAGE1/HPRICE1 model spesifikasyonlarıyla çoklu doğrusal regresyon, ceteris paribus yorumunu, profil katkılarını, kısmi ilişkiyi ve makale tablosu okumayı öğretir. `konu06_ols_varsayimlari_yanlilik.py`, A1–A4 varsayımlarını, yansızlık laboratuvarını, OVB ayrıştırmasını ve çoklu doğrusal bağlantı tanılarını gösterir.
- `core/data_registry.py`: Wooldridge kataloğu, değişken açıklamaları, veri yapısı metadata'sı ve izin verilen pedagojik eşleşmeler. Konu 02 WAGE1, PHILLIPS, CPS78_85, WAGEPAN ve JTRAIN2 kullanır.
- `core/data_structure_utils.py`: Tekrar eden birimleri, dönem kapsamını, zaman sırasını ve panel dengesini Streamlit bağımlılığı olmadan doğrular.
- `core/group_comparison_utils.py`: İki grubun gözlem sayısını, ortalamasını ve ortalama farkını hesaplar; çıkarımsal çıktı üretmez.
- `core/model_utils.py`: Veri hazırlama, basit EKK, belirli bir X değeri için tahmin, TKT/MKT/HKT ayrıştırması, R-kareyi iki eşdeğer biçimde hesaplama, doğrusal ölçü birimi dönüşümü ve dört temel fonksiyonel biçimin doğrulanmış log dönüşümünü sağlar. Streamlit bağımlılığı yoktur.
- `core/multiple_regression_utils.py`: Ortak complete-case örneklem, çoklu EKK, HPRICE1'in açık ölçek dönüşümleri, profil tahmin/katkıları, gözlem artıkları ve Streamlit'ten bağımsız FWL kısmi regresyon hesabını sağlar.
- `core/assumption_diagnostics_utils.py`: Konu 06 için seed'li ve vektörize tekrarlı EKK benzetimi, OVB yön/formül hesabı, sentetik ve WAGE1 kısa–uzun–yardımcı regresyon ayrıştırması, rank/tam bağlantı denetimi, VIF, yüksek bağlantı benzetimi ve küçük veri değişikliği duyarlılığını sağlar.
- `core/konu05_questions.py`: Konu 05'in model sonucuna bağlı, kararlı sırada dönen ve çözümü gizli tutulan soru türlerini üretir.
- `core/konu06_questions.py`: Konu 06'nın varsayım, yansızlık, OVB, WAGE1 ve bağlantı sonuçlarına bağlı, kararlı sırada dönen soru türlerini üretir.
- `core/konu04_questions.py`: Konu 04'ün model sonucuna bağlı, kararlı sırada dönen ve çözümü gizli tutulan soru türlerini üretir.
- `core/question_engine.py` ve `core/scenario_registry.py`: Ortak soru veri yapısı, kararlı soru sırası ve Konu 01–02 senaryo fabrikalarıyla deterministik soru/çözüm üretir.
- `core/session_utils.py`: Soru sırası ve cevap görünürlüğünün `st.session_state` içindeki anahtarlarını yönetir.
- `assets/styles.css`: Kurumdan bağımsız marka değişkenleri ve duyarlı görsel düzen.

## Veri akışı

1. Konu modülü, `data_registry` üzerinden Wooldridge paketindeki veri setini önbellekli olarak yükler.
2. Öğrencinin seçtiği izinli Y–X çifti `model_utils.fit_simple_ols` fonksiyonuna verilir.
3. Model çıktısı grafik, tablo, belirli bir X değeri için tahmin paneli ve soru motoru tarafından kullanılır.
4. Veri seti veya değişkenler değiştiğinde model kimliği değişir; `session_utils` önceki soru durumunu sıfırlar.

Konu 04 yalnız sabit terimli EKK sonuçlarını kullanır. Bir gözlemde `Yᵢ−Ȳ=(Ŷᵢ−Ȳ)+ûᵢ` ayrıştırması gösterilir; örneklemde `TKT=MKT+HKT` toleransla doğrulanır. Ölçü birimi aracı `Y'=aY`, `X'=bX` dönüşümünde katsayıları cebirsel olarak yeniden yazar ve R-kareyi değişmeden gösterir. Fonksiyonel biçim aracı düzey–düzey, log–düzey, düzey–log ve log–log biçimlerini kullanır; sıfır ya da negatif değerler için log dönüşümünü açık hata ile durdurur, gözlemleri sessizce düşürmez.

Konu 02, veri yapısı metadata'sını gerçek paket sütunlarıyla doğrular. WAGEPAN'da panel dengesi her kişinin gözlendiği dönem kümesi üzerinden; JTRAIN2'de ise eğitim ve kontrol gruplarının betimsel ortalamaları üzerinden incelenir.

Model sonuçları betimseldir. Bu pilotta istatistiksel çıkarım sütunları ve yorumları bilinçli olarak sunulmaz.

Konu 05 yalnız önceden tanımlanmış WAGE1 ve HPRICE1 modellerini kullanır. HPRICE1'de `lotsize1000=lotsize/1000` ve `sqrft100=sqrft/100` dönüşümleri açıkça gösterilir. Veri, model, seçili gözlem veya profil değiştiğinde model kimliği değişir ve soru/cevap durumu sıfırlanır; öğrenci sonuç tablolarına standart hata, test veya güven aralığı alanları taşınmaz.

Konu 06, WAGE1 üzerinde `wage ~ educ`, `wage ~ educ + exper`, `wage ~ educ + exper + tenure` ve `exper ~ educ` modellerini aynı complete-case örneklemde kurar; eğitim katsayısı ayrıştırması bu ortak örnekleme dayanır. Tekrarlı örnekleme ve yüksek bağlantı benzetimleri sabit seed kullanır, UI katmanında `st.cache_data` ile senaryo/boyut/tekrar/seed anahtarlarına göre önbelleklenir. Tam bağlantıda rank eksikliği benzersiz katsayı tahmininden önce görünür bir hata olarak verilir; yüksek fakat tam olmayan bağlantıda korelasyon, yardımcı R-kare, VIF, tekrarlı tahmin değişkenliği ve duyarlılık birlikte gösterilir. Konu 06 soru bağlamı değiştiğinde cevap görünürlüğü ve soru sırası sıfırlanır.

Standart hata, t istatistiği, p-değeri, güven aralığı ve F testi Konu 06 öğrenci tablolarında bilinçli olarak yer almaz. Bu ortak çıkarım altyapısı ve aktif öğretim alanı sonraki `shared-regression-inference` / Konu 07 çalışmasına bırakılmıştır.

Konu 06'nın öğrenci tabloları yalnız sunum için biçimlendirilmiş değerler kullanır: modelde bulunmayan katsayılar `—`, salt-okunur doğrulamalar `Evet/Hayır`, kayan nokta yuvarlama farkları ise “sayısal tolerans içinde 0” olarak gösterilir. Saf hesaplama sonuçları bu biçimlendirmeden etkilenmez.
# Konu 08–09 ortak altyapısı

Konu 08 ve Konu 09 aynı geliştirme dalında yer alsa da ayrı `topics/`, soru ve test modülleridir. `core/joint_inference_utils.py`, yalnız geleneksel `nonrobust` kovaryans altında genel `Rβ=r` ortak F testi, iç içe modeller için SSR/R² F karşılaştırması, genel anlamlılık, tek kısıtta `F=t²`, F dağılım grafiği verisi ve NumPy batch büyük-örneklem benzetimini taşır.

`core/functional_form_utils.py`, ölçekleme, standartlaştırma, log-yüzde, karesel marjinal etki/dönüm noktası, WAGE1 M1–M4 ve merkezleme iş mantığını taşır. Karesel terimlerin ortak testi kendi F formülünü yazmaz; Konu 08'in `nested_exclusion_f_test` aracını kullanır. Konu ekranlarındaki pahalı sabit model ve benzetim sonuçları `st.cache_data` ile önbelleklenir.

Konu 09 model seçimini tek bir metriğe indirmez; teori, işaret, veri aralığı, basitlik, SSR/R²/düzeltilmiş R² ve ortak F birlikte sunulur. Merkezleme fitted değerleri, artıklar, SSR ve R²'yi değiştirmez; içselliği çözmez. Dayanıklı ortak testler Konu 12'ye, kukla değişkenler Konu 10'a bırakılır. Soru durumları `session_utils` içinde konu bazlı anahtarlarla tutulur ve konu geçişinde sıfırlanır.

`core/ui_preferences.py`, konu durumundan bağımsız metin ölçeği seçeneklerini ve CSS/Plotly yazı boyutlarını üretir. Seçim `text_scale_label` ile session state'te korunur; `app.py` bunu tek CSS değişkenine uygular. Tüm konu soru blokları birincil “Cevabı göster” ve ikincil “Yeni soru” düğme sistemini kullanır.

Konu 08'deki custom laboratuvarı serbest metin ayrıştırmaz. Kullanıcı kontrollü R satırları ve r sağ taraflarını girer; `validate_restriction_system` rank(R), rank([R|r]), sıfır bilgi satırı, bağımlı satır ve tutarsız sistemi görünür kılar. Geçersiz sistemlerde test çalışmaz. `classify_restriction_system`, yalnız basit sıfır-dışlama kısıtlarında nested SSR/R² karşılaştırmasını açar; eşitlik, sıfır dışı ve doğrusal birleşim kısıtları genel `Rβ=r` matrix F ile sınanır.
