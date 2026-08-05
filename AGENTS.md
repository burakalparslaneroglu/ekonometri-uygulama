\# Ekonometri Uygulaması ve Proje Talimatları



\## Proje amacı



Bu depo, lisans düzeyindeki Ekonometriye Giriş dersi için Python ve

Streamlit tabanlı etkileşimli bir öğretim uygulamasıdır.



Ders notları konu kapsamı, terminoloji, notasyon, konu sırası ve

ekonometrik yorumlar bakımından bağlayıcı kaynaktır.



\## Teknik çerçeve



\- Python 3.12 kullan.

\- Streamlit Community Cloud ile uyumlu kod yaz.

\- Sabit yerel dosya yolları kullanma.

\- iş mantığını Streamlit arayüzünden ayır.

\- app.py yalnızca ortak arayüz ve konu yönlendirmesi taşısın.

\- Konu modülleri topics/ altında bulunsun.

\- Ortak veri, model, soru ve oturum işlevleri core/ altında bulunsun.

\- Ağır veya tekrarlı işlemlerde uygun Streamlit cache mekanizmasını kullan.

\- Çalışma zamanı LLM veya dış API bağımlılığı ekleme.

\- Secret, parola veya kişisel veri commit etme.



\## Görsel tasarım



Sunumlarla aynı renk paletini kullan:



\- brandInk: #07373D

\- brandDeep: #0C5B65

\- brandTeal: #107C89

\- brandBright: #15A4B5

\- brandGreen: #2F9E6B

\- brandRed: #B3392F



Açık tema kullan. Ana metin ve vurgularda yeterli renk karşıtlığını koru.

Doğru cevaplarda brandGreen, uyarılarda brandRed kullan; ancak yalnızca

renge dayalı bilgi verme.

\## Veri politikası



\- Wooldridge verilerini wooldridge Python paketi üzerinden yükle.

\- Wooldridge Excel dosyalarını depoya kopyalama.

\- Her veri setinde kaynak ve değişken açıklamasını göster.

\- Benzetimlerde sabit ve açık bir seed kullan.

\- Kullanıcının yüklediği verileri kalıcı olarak kaydetme.



\## Pedagojik kurallar



\- Arayüz dili Türkçe olsun.

\- Önemli ingilizce terimler ilk kullanımda parantez içinde verilebilir.

\- Cross-sectional data için yatay kesit verisi terimini kullan.

\- Kavramları ders sırasından önce kullanma.

\- Sorular ders notlarındaki egzersizlerin kopyası olmasın.

\- Soruları hesaplanan sonuçlara göre deterministik olarak üret.

\- Cevabı yalnızca öğrenci Cevabı göster düğmesine bastığında göster.

\- istatistiksel anlamlılık ile iktisadi önemi ayır.

\- Araştırma tasarımı desteklemiyorsa nedensel dil kullanma.



\## Kod kalitesi



\- Fonksiyonlarda type hint ve kısa docstring kullan.

\- Sessizce hata yutma.

\- Kullanıcıya anlaşılır hata mesajı göster.

\- Sayısal hesaplamalar için pytest testleri yaz.

\- Değişiklik sonrasında testleri ve Streamlit başlangıç kontrolünü çalıştır.

