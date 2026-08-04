\# Ekonometri UygulamasÄ± â€” Proje TalimatlarÄ±



\## Proje amacÄ±



Bu depo, lisans dÃ¼zeyindeki Ekonometriye GiriÅŸ dersi iÃ§in Python ve

Streamlit tabanlÄ± etkileÅŸimli bir Ã¶ÄŸretim uygulamasÄ±dÄ±r.



Ders notlarÄ± konu kapsamÄ±, terminoloji, notasyon, konu sÄ±rasÄ± ve

ekonometrik yorumlar bakÄ±mÄ±ndan baÄŸlayÄ±cÄ± kaynaktÄ±r.



\## Teknik Ã§erÃ§eve



\- Python 3.12 kullan.

\- Streamlit Community Cloud ile uyumlu kod yaz.

\- Sabit yerel dosya yollarÄ± kullanma.

\- Ä°ÅŸ mantÄ±ÄŸÄ±nÄ± Streamlit arayÃ¼zÃ¼nden ayÄ±r.

\- app.py yalnÄ±zca ortak arayÃ¼z ve konu yÃ¶nlendirmesi taÅŸÄ±sÄ±n.

\- Konu modÃ¼lleri topics/ altÄ±nda bulunsun.

\- Ortak veri, model, soru ve oturum iÅŸlevleri core/ altÄ±nda bulunsun.

\- AÄŸÄ±r veya tekrarlÄ± iÅŸlemlerde uygun Streamlit cache mekanizmasÄ±nÄ± kullan.

\- Ã‡alÄ±ÅŸma zamanÄ± LLM veya dÄ±ÅŸ API baÄŸÄ±mlÄ±lÄ±ÄŸÄ± ekleme.

\- Secret, parola veya kiÅŸisel veri commit etme.



\## GÃ¶rsel tasarÄ±m



Sunumlarla aynÄ± renk paletini kullan:



\- pauInk: #07373D

\- pauDeep: #0C5B65

\- pauTeal: #107C89

\- pauBright: #15A4B5

\- pauGreen: #2F9E6B

\- pauRed: #B3392F



AÃ§Ä±k tema kullan. Ana metin ve vurgularda yeterli renk karÅŸÄ±tlÄ±ÄŸÄ±nÄ± koru.

DoÄŸru cevaplarda pauGreen, uyarÄ±larda pauRed kullan; ancak yalnÄ±zca renge

dayalÄ± bilgi verme.



\## Veri politikasÄ±



\- Wooldridge verilerini wooldridge Python paketi Ã¼zerinden yÃ¼kle.

\- Wooldridge Excel dosyalarÄ±nÄ± depoya kopyalama.

\- Her veri setinde kaynak ve deÄŸiÅŸken aÃ§Ä±klamasÄ±nÄ± gÃ¶ster.

\- Benzetimlerde sabit ve aÃ§Ä±k bir seed kullan.

\- KullanÄ±cÄ±nÄ±n yÃ¼klediÄŸi verileri kalÄ±cÄ± olarak kaydetme.



\## Pedagojik kurallar



\- ArayÃ¼z dili TÃ¼rkÃ§e olsun.

\- Ã–nemli Ä°ngilizce terimler ilk kullanÄ±mda parantez iÃ§inde verilebilir.

\- Cross-sectional data iÃ§in yatay kesit verisi terimini kullan.

\- KavramlarÄ± ders sÄ±rasÄ±ndan Ã¶nce kullanma.

\- Sorular ders notlarÄ±ndaki egzersizlerin kopyasÄ± olmasÄ±n.

\- SorularÄ± hesaplanan sonuÃ§lara gÃ¶re deterministik olarak Ã¼ret.

\- CevabÄ± yalnÄ±zca Ã¶ÄŸrenci CevabÄ± gÃ¶ster dÃ¼ÄŸmesine bastÄ±ÄŸÄ±nda gÃ¶ster.

\- Ä°statistiksel anlamlÄ±lÄ±k ile iktisadi Ã¶nemi ayÄ±r.

\- AraÅŸtÄ±rma tasarÄ±mÄ± desteklemiyorsa nedensel dil kullanma.



\## Kod kalitesi



\- Fonksiyonlarda type hint ve kÄ±sa docstring kullan.

\- Sessizce hata yutma.

\- KullanÄ±cÄ±ya anlaÅŸÄ±lÄ±r hata mesajÄ± gÃ¶ster.

\- SayÄ±sal hesaplamalar iÃ§in pytest testleri yaz.

\- DeÄŸiÅŸiklik sonrasÄ±nda testleri ve Streamlit baÅŸlangÄ±Ã§ kontrolÃ¼nÃ¼ Ã§alÄ±ÅŸtÄ±r.
