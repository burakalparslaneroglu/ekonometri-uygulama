# İKT 305 sunum bağlantıları

Temel adres: https://ikt-305-ekonometri-i.streamlit.app/.

```text
/?konu=03&sekme=sezgi&deney=2
/?konu=05&sekme=uygulama&adim=4
/?konu=03&sekme=sezgi&deney=1&ayar_n=50
/?konu=03&sekme=uygulama&adim=4&kaynak=kendi
```

- `konu`: 00–12 (tek haneli numara da kabul edilir).
- `sekme`: `uygulama`, `sezgi`, `sinama`. Varsayılan Uygulama; sekme sırası değişmez.
- `deney`: seçilen konunun mevcut Sezgi deney numarası; yalnız Sezgi hedefinde.
- `adim`: seçilen konunun gerçek laboratuvar adım numarası; yalnız Uygulama hedefinde.
- `kaynak`: `notlar` (varsayılan), `alternatif`, `kendi`; yalnız Uygulama hedefinde.
- `ayar_<parametre>`: seçilen deneyin kaydırıcı anahtarı. Deney numarası zorunludur; tip, sonlu değer,
  aralık ve kaydırıcı adımı doğrulanır. Belirtilmeyen ayarlar mevcut deneyin varsayılan değerlerini alır.

Tanımlar `core/labs/sezgi_konuNN.py` ve `core/labs/konuNN.py` dosyalarından okunur; hesaplar ve deneyler değişmedi.
Geçersiz/tekrarlı/bilinmeyen ya da seçilen sekmeyle çelişen parametreler açıklayıcı hata verir.
Eksik dosya kendi veri hedefini başka kaynağa yönlendirmez; seçilen adım ve yükleme alanı görünür.

Başlangıç durumu widget'lar çizilmeden önce yalnız URL parametreleri değiştiğinde uygulanır. Aynı URL'deki yeniden
çalışmalarda öğretim elemanının konu, sekme, deney, adım, spesifikasyon ve kaydırıcı seçimleri korunur. Notlardaki
uygulama bağlantısı daha önceki alternatif veri seçimini ve değiştirilmiş not spesifikasyonunu başlangıçta sıfırlar.

Streamlit alt sınırı 1.61: bu sürümde kurulu `st.tabs(key=..., on_change="rerun")` API'si kullanıldı ve test edildi.
Üç sekmenin içeriğinin hesaplanma düzeni korundu. Referans dersteki 1.64 koşulu bu uygulamaya aktarılmadı.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_navigation.py -q
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
```

Sunumlar ve öğretim elemanının yerel ders akışı rehberi public depoya eklenmez. Commit/push sonrası canlıda
en az bir Sezgi ve bir Uygulama bağlantısı yeniden açılmalı; tarayıcıda görünür sekme ve hedef seçimi doğrulanmalıdır.
