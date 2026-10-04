# İKT 305 Ekonometri I — Depo Kuralları

Bu depo, İzmir Bakırçay Üniversitesi İKT 305 Ekonometri I dersi (2026–2027) için Python ve Streamlit tabanlı
etkileşimli öğretim uygulamasıdır. Ana kaynak: Wooldridge, J. M. (2020). *Introductory Econometrics: A Modern
Approach* (7. baskı). Cengage.

## Bağlayıcı kurallar

1. Güncel yerel ders notları konu sırası, terminoloji, notasyon, estimand, varsayım dili ve yorumlama sınırları için
   bağlayıcıdır.
2. Ders notu, sunum, uygulama veya literatür arasında uyuşmazlık ya da notlarda hata bulunursa sessiz düzeltme
   yapılmaz. Düzeltme aynı turda kaynağından başlar: önce ders notu (LaTeX), sonra ilgili sunum, sonra uygulama; notlar
   ve sunumlar sıfır hatayla yeniden derlenir. Her değişiklik raporlanır. Notasyon veya terim gibi yoruma açık seçimler
   gerekçesiyle bildirilir; içerik kararı gerektiren uyuşmazlıklar `NOTE_CONSISTENCY_ISSUE` olarak raporlanır ve
   kullanıcı kararı beklenir. Notlar ve sunumlar bu depoda değil, öğretim elemanının yerel klasöründe tutulur (kural 12).
3. Look-ahead öğretim yapılmaz. Sonraki konunun yöntemi önceki konuda aktif laboratuvar olarak açılmaz: standart hata,
   t, p-değeri ve güven aralığı Konu 7'de; F testi Konu 8'de; heteroskedastisiteye dayanıklı çıkarım Konu 12'de açılır.
4. Ekonometrik hesaplama, veri hazırlama, simülasyon ve soru üretimi Streamlit'ten bağımsız `core/` katmanında tutulur.
5. `app.py` yalnız ortak kabuk, gezinme ve seçili konunun `render()` çağrısını içerir.
6. Çalışma zamanında büyük dil modeli, dış yapay zekâ API'si veya gizli anahtar kullanılmaz.
7. Rassallık `np.random.default_rng(seed)` ile yönetilir; Sezgi deneylerinde tohum 305'tir. Seed ve ayarlar sonuç
   bilgisinde görünür.
8. Yeni yöntem sayısal benchmark testi olmadan eklenmez.
9. Araştırma tasarımı desteklemiyorsa nedensel dil kullanılmaz.
10. Estimand, estimator ve estimate ayrımı arayüz metninde ve sonuç sözleşmelerinde korunur.
11. Dayanıklı standart hata içselliği, eksik değişken yanlılığını veya nedensel tasarımı düzeltiyor gibi sunulmaz.
12. Private ders materyali ve lisansı doğrulanmamış veri public depoya commit edilmez; `references_private/` izlenmez.
    Veri depoya kopyalanmaz: Wooldridge verileri çalışma anında `wooldridge` Python paketinden, üretilen R kodunda
    `wooldridge` R paketinden okunur. Wooldridge Excel veya CSV dosyaları depoya girmez.
13. Her grafikte eksen adı, gerektiğinde birim, legend ve model/veri bağlamı bulunur.
14. Ham teknik değişken adları öğrenci arayüzünde açıklamasız gösterilmez; her ad Türkçe bir etiket alır.
15. Her dal sonunda `pytest`, `compileall` ve `git diff --check` çalıştırılır.
16. Her konu üç sekmeden oluşur ve bu sıra korunur: **Uygulama** (notlardaki laboratuvarın adımları birebir; notlarda
    basılı her sayı bir `Check`), **Sezgi** (DGP'si açıkça yazılmış kontrollü deneyler; soru → DGP ve parametreler →
    neye bakıyoruz → sonuç → ne gördük → kod), **Kendini sına** (çoktan seçmeli, doğru–yanlış, boşluk doldurma,
    denklem yazma; her soru tek kavram ve bir bölüm; bölüm sonu egzersizleri ve mini quizler tekrar edilmez).
17. Python ve R kodu aynı tanımdan üretilir; elle yazılmış dile özgü şablon eklenmez. Notlarda Stata kodu olmadığı
    için Stata üretilmez (kullanıcı kararı). Her adım veya deney, iki dilde aynı sayının hangi anlamda beklendiğini
    (birebir / ayar sabitlenince / yalnız dağılımda) gösterir.
18. Uygulama adımlarında öğrencinin seçimi (ör. açıklayıcı değişken) notlardaki spesifikasyonu varsayılan olarak
    korur. Notlardan farklı bir seçimde kontroller gösterilmez; seçim sonraki adımlara geçer ve notlardaki model ile
    seçilen model yan yana gösterilir.
19. Bütün konular (0–12) yeni yapıdadır ve `tests/test_topic_contracts.py` içindeki `MIGRATED_TOPICS` kümesindedir;
    kaldırılan eski modüller ve testler (aynı dosyadaki liste) geri eklenmez. Yeni bir konu aynı sözleşmeyle eklenir.
20. Bütün konuların (0–12) Uygulama sekmesinde veri kaynağı seçimi vardır: **Notlardaki örnek** (varsayılan),
    **Alternatif örnek**, **Kendi verini yükle**. Notlardaki uygulama ve onun üretilen kodu ek kaynaklar yüzünden
    değişmez. İki ek kaynak aynı adım numaralarını ve aynı işlemleri genel uygulamadan (`core/labs/ornek_konuNN.py`)
    kurar; kontrollerin beklenen değerleri uygulamanın hesabıdır, alternatif örneklerin sayıları testlerde bağımsız bir
    hesapla doğrulanır. Alternatif örnekler gerçek Wooldridge verisidir; kurgusal veri yalnız veri üretim süreci modül
    belgesinde yazılıyken ve metinde kurgusal olduğu söylenerek kullanılır (Konu 2'nin deney adımı, Konu 3'ün sıfır–bir
    değişken adımı).
    Konu 3–12'de kendi verinin seçilen sayısal sütunlarında (sonuç, temel açıklayıcı, ek değişkenler) ve zorunlu
    kategorik sütununda (Konu 11) boş hücresi olan satırlar çıkarılır: bütün modeller aynı gözlemlerle kurulur.
    İsteğe bağlı kategorik sütunda (Konu 3, 10) boş hücre kabul edilmez. Notlardaki model log sonuçluysa (Konu 9–11)
    kendi veri de log sonuçla kurulur ve sonucun bütün değerleri pozitif olmalıdır; Konu 12'nin log modeli yalnız sonuç
    ve temel açıklayıcı pozitifse kurulur, değilse adım neye ihtiyacı olduğunu yazar. Notlarda basılı, veriye özgü
    sayılar (kaldıraç, HC3 payı, dönüm noktası, etkili gözlem) kendi veride veriden hesaplanır. Uyum tamsa (R² ≈ 1)
    standart hataya bağlı kontroller çıkarılır ve metin nedenini söyler; yuvarlama gürültüsü düzeyindeki bir katsayının
    işareti yorumlanmaz.
21. Yüklenen dosya yalnız oturumun belleğinde işlenir: ortak önbelleğe (`st.cache_data`, `st.cache_resource`), diske
    ya da günlüğe yazılmaz. Öğrencinin sütun ve kategori adları Markdown metnine ve widget etiketlerine `md` ile
    kaçırılarak girer; matematik ifadesinin içine yazılmaz. Okuma ve temizleme kuralları İKT 217 uygulamasıyla aynıdır
    (`core/labs/kendi_veri.py`); iki dilin aynı okuyamayacağı dosya ve hücreler açık bir iletiyle reddedilir.
22. Üretilen R kodunun `wooldridge` dışındaki tek paketi `readxl`'dir ve yalnız yüklenen Excel dosyasını okumak için
    kullanılır; CSV temel R ile okunur.

## Teknik çerçeve

- Python 3.12; Streamlit Community Cloud ile uyumlu kod.
- Sabit yerel dosya yolu kullanılmaz; üretilen kod çalıştığı klasörde dosya bırakmaz.
- Ağır veya tekrarlı işlemlerde uygun Streamlit önbelleği kullanılır.
- Fonksiyonlarda type hint ve kısa docstring; hata sessizce yutulmaz, kullanıcıya anlaşılır Türkçe mesaj gösterilir.
- Sayısal hesaplamalar için pytest testleri yazılır.

## Görsel tasarım

Sunumlarla aynı renk paleti: brandInk `#07373D`, brandDeep `#0C5B65`, brandTeal `#107C89`, brandBright `#15A4B5`,
brandGreen `#2F9E6B`, brandRed `#B3392F`. Açık tema; yeterli renk karşıtlığı. Doğru cevaplarda brandGreen, uyarılarda
brandRed kullanılır; bilgi yalnız renge dayandırılmaz.

## Metin ve biçim

- Arayüz dili Türkçedir. Önemli İngilizce terimler ilk kullanımda parantez içinde verilebilir. Cross-sectional data
  için "yatay kesit verisi" terimi kullanılır.
- Ondalık virgül, önde yüzde işareti (%16,5) ve tipografik eksi (−) kullanılır. Değişken bir sayıya Türkçe ek
  getirilmez ("0,35'ye" yazılmaz). Metin "80. ..." gibi sayı ve noktayla başlamaz.
- Beklenen değer LaTeX'te `\mathbb{E}(Y \mid X)`, düz metinde `E(Y | X)` biçiminde yazılır. Kod adları backtick ile
  yazılır.
- Kendini sına sekmesinde doğru cevap ve açıklama yalnız öğrenci cevabını "Kontrol et" düğmesiyle denetlediğinde
  gösterilir. İstatistiksel anlamlılık ile iktisadi önem ayrılır.
