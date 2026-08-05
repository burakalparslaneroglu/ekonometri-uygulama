# Mimari

Uygulama, Streamlit arayüzü ile hesaplama mantığını ayıran küçük modüllerden oluşur.

- `app.py`: Ortak sayfa yapılandırması, ortak görsel iskelet ve konu yönlendirmesi.
- `topics/`: Her ders konusunun Streamlit görünümü. `konu05_coklu_regresyon.py`, sabit WAGE1/HPRICE1 model spesifikasyonlarıyla çoklu doğrusal regresyon, ceteris paribus yorumunu, profil katkılarını, kısmi ilişkiyi ve makale tablosu okumayı öğretir.
- `core/data_registry.py`: Wooldridge kataloğu, değişken açıklamaları, veri yapısı metadata'sı ve izin verilen pedagojik eşleşmeler. Konu 02 WAGE1, PHILLIPS, CPS78_85, WAGEPAN ve JTRAIN2 kullanır.
- `core/data_structure_utils.py`: Tekrar eden birimleri, dönem kapsamını, zaman sırasını ve panel dengesini Streamlit bağımlılığı olmadan doğrular.
- `core/group_comparison_utils.py`: İki grubun gözlem sayısını, ortalamasını ve ortalama farkını hesaplar; çıkarımsal çıktı üretmez.
- `core/model_utils.py`: Veri hazırlama, basit EKK, belirli bir X değeri için tahmin, TKT/MKT/HKT ayrıştırması, R-kareyi iki eşdeğer biçimde hesaplama, doğrusal ölçü birimi dönüşümü ve dört temel fonksiyonel biçimin doğrulanmış log dönüşümünü sağlar. Streamlit bağımlılığı yoktur.
- `core/multiple_regression_utils.py`: Ortak complete-case örneklem, çoklu EKK, HPRICE1'in açık ölçek dönüşümleri, profil tahmin/katkıları, gözlem artıkları ve Streamlit'ten bağımsız FWL kısmi regresyon hesabını sağlar.
- `core/konu05_questions.py`: Konu 05'in model sonucuna bağlı, kararlı sırada dönen ve çözümü gizli tutulan soru türlerini üretir.
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
