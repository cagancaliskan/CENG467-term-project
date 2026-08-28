# IDAP 2026 — Kamera-Hazır Revizyon Planı (7 gün)
## "Synthetic Data Distillation from LLMs for Turkish Abstractive News Summarization"

| | |
|---|---|
| **Mekan** | IDAP 2026, 10. Uluslararası Yapay Zekâ ve Veri İşleme Sempozyumu — 5–6 Eylül 2026, Marmara Üniv. / IEEE Xplore |
| **Durum** | **Kabul edildi**, yorumlar kamera-hazırda ele alınacak — **yeniden hakem değerlendirmesi yok** |
| **Deadline** | **~27 Ağustos 2026** (bugün 20 Ağustos → **7 gün**) |
| **Format** | IEEEtran iki sütun, US Letter, **7 sayfa**, 28 kaynak (`NLP_AYTUG_HOCA_SON4.pdf`) |
| **Bütçe** | ~11 sa Colab T4 · **\$1.00 Anthropic kredisi** · ~3 sa senin anotasyonun |

> **Bu doküman plandır. Hiçbir koda ve makaleye dokunulmadı.**
> Daha önce hazırlanan 3 haftalık geniş planın yerine geçer; o plan IDAP'ın 5-6 Eylül tarihiyle uyumlu değildi.

---

## 1. Çıta neresi

"Kabul edildi, yorumları kamera-hazırda dikkate alın" demek: **hakemi ikna etmek zorunda değilsin, dürüst olmak zorundasın.** 7 günde yapılabilecek en iyi şey:

1. **Yanlış olanı düzelt** — bilerek hatalı sayı IEEE Xplore'a gitmemeli. Bu pazarlık konusu değil.
2. **Her hakem maddesini görünür şekilde karşıla** — ya gerçek bir düzeltmeyle, ya da doğru yere konmuş dürüst bir limitation'la.
3. **Fazlasını yapmaya kalkma.** Yarım kalmış bir NLI deneyi, net yazılmış bir limitation'dan daha kötüdür.

Gönderilen sürüm zaten iyi: halüsinasyon azalmasını "trade-off" diye çerçeveliyor, sentinel artefaktını açıkça anlatıyor, §VII-B'de "Claude Opus 4.7 ile Claude Haiku 4.5 aynı geliştiriciden, bu self-enhancement bias yaratabilir; bağımsız bir hakemle kontrol gelecek çalışmaya bırakılmıştır" diyor. **Hakemler tam da bu üç "gelecek çalışmaya bırakıldı" cümlesini okuyup "hayır, şimdi yap" dedi.** İyi haber: üçünün de 7 günlük versiyonu var.

---

## 2. 🔴 Zorunlu düzeltme: makalede yanlış bir cümle var

**Sayfa 4, sağ sütun:**
> *"Two consequences matter for interpreting Table IV. First, the lexical metrics in Tables I–II are **unaffected**, because ROUGE's tokenizer already discards angle-bracketed sentinels as punctuation."*

**Bu yanlış.** `src/eval/rouge_tr.py` içindeki `_PUNCT_RE = [^\w\s]` yalnızca `<` ve `>` karakterlerini siliyor. `\w` alt çizgi ve rakamı kapsadığı için geriye **`extra_id_0` diye tam bir kelime token'ı** kalıyor (stem modunda `extra`). Projenin kendi tokenizer'ıyla doğruladım:

```
ref   : Ankara Büyükşehir Belediyesi ulaşım için 15 milyon lira ödenek ayırdı.
dirty : <extra_id_0> Ankara Büyükşehir Belediyesi ulaşım için 15 milyon lira ayırdı <extra_id_1>

standard  ROUGE-1:  kirli P=0.818 F1=0.8571  →  temiz P=1.000 F1=0.9474   ΔF1 = +0.090
standard  ROUGE-2:  kirli P=0.700 F1=0.7368  →  temiz P=0.875 F1=0.8235   ΔF1 = +0.087
standard  ROUGE-L:  kirli P=0.818 F1=0.8571  →  temiz P=1.000 F1=0.9474   ΔF1 = +0.090
stem-5    (aynı büyüklükte etki — 'extra' de bir token)
```

Dokuz kelimelik bir özetteki iki sentinel **9 ROUGE-1 puanı** götürüyor. Hakem notlarına göre 180 satırın 74'ünde (%41) sentinel var; küçük-model sistemlerinde oran daha yüksek.

**Sonuçları:**
- **Tablo I, II ve III'teki B1 / B2 / S-gpt / S-claude satırları düşük ölçülmüş.** BERTScore, `extract` ve `lr` de aynı şekilde.
- Cümlenin kendisi silinip yerine ölçülmüş etki yazılmalı.
- Bu düzeltme büyük olasılıkla **lehimize**: küçük modeller yükselince "öğrenci öğretmenin %91'ine ulaşıyor" iddiası güçlenir. Ama ne çıkarsa o yazılacak — sentinel silmek geriye içerik boşluğu bırakabilir, ters yönde de çıkabilir. **İki yön de önceden ilan ediliyor.**
- §VII-B'deki "the lexical metrics are unaffected" mantığına dayanan tüm ifadeler gözden geçirilecek.

---

## 3. Ne yapacağız / ne yapmayacağız

| Hakem maddesi | 7 günde yapılacak | Kalan açık → limitation |
|---|---|---|
| **R5** Artefakt düzeltildi ama deneyler tekrarlanmadı | Tüm tahminler temizlenip **Tablo I, II, III yeniden hesaplanacak**; hakem **temiz çıktılarla yeniden koşulacak**; kirli-vs-temiz A/B tablosu | — (tamamen kapanıyor) |
| **R1** Tek eğitim koşusu, CI ve anlamlılık yok | **B2, S-gpt, S-claude için 3 tohum** (Tablo I'deki tüm eğitilmiş sistemler) + **eşleştirilmiş bootstrap %95 CI** + permütasyon testleri, tüm tablolarda | Ablasyonların (boyut, prompt, r4) çoğu tek tohum kalıyor |
| **R2** "Eşdeğer", "öğretmen etkisiz", "rank 8 en iyi" iddiaları desteksiz | **TOST eşdeğerlik testi** (B3a↔B3b, S-gpt↔S-claude) + **r16/r32 için 2 ek tohum** → rank 8 iddiası CI'larla test edilecek | Öğretmenlerin decode varyansı ölçülemiyor (API bütçesi yok) |
| **R3** "5×" tek metriğe dayanıyor; kişi/olay/ilişki sayılmıyor | **Varlık düzeyi sadakat** (Türkçe NER, precision/recall, PER ayrı) + **numeric_precision** + **100 token başına oran** (uzunluk konfaundunu kırar) + **insan anotasyonunda `entity_hallucination`** | NLI/entailment tabanlı önerme sadakati → gelecek çalışma |
| **R4** Kısa özetler → atlama ve kişi/eylem karışması | **ROUGE precision/recall ayrı** (atlamanın en doğrudan kanıtı, bedava) + **entity recall** + **insan anotasyonunda `misattribution` ve `salient_omission`** | Otomatik SRL/rol metriği yok (Türkçe pro-drop; insan anotasyonu yerine geçiyor) |
| **R6** 30 örnek; Claude Claude'u değerlendiriyor | Örneklem **30 → 60 makale**; **körlenmiş** (v1'de prompt'ta `system=B3a` yazıyordu); bağlam **600 → 3000 karakter**; **2 bağımsız hakem, ikisi de Anthropic dışı**; self-preference farkı ölçülüyor | Tek insan anotatör; hakem örneklemi hâlâ orta boy |

**Kasten yapılmayacaklar** (§VII-B'ye net yazılacak): NLI tabanlı sadakat · otomatik yanlış-atıf metriği · ablasyonlarda tam çok-tohum · öğretmen decode varyansı · ikinci anotatör (bulabilirsen değişir).

---

## 4. İş kalemleri

### A. Temizlik ve yeniden hesap — *zorunlu*
- `src/eval/clean_predictions.py`: `infer.py`'deki `<extra_id_\d+>` regex'iyle birebir aynı temizlik, arşivlenmiş 18 tahmin dosyasına
- **Özdeşlik ispatı:** `S_gpt_n10000_r8` ile 200 makalede düzeltilmiş `infer.py` yeniden koşulup post-processing çıktısıyla string bazında %100 eşleşme gösterilecek → "yeniden üretmedik, post-processing yaptık" itirazı kapanır
- Tüm ROUGE (CPU) + BERTScore (GPU) + hata bayrakları yeniden
- **Sentinel görülme oranı sistem başına** — bağımsız bir üretim-kalitesi bulgusu, §VI'ya girer
- Kirli → temiz **Δ tablosu**

### B. Çok-tohumlu eğitim
Tohumlar: `42` (mevcut) + `1337` + `2024`. Yalnız optimizasyon gürültüsü değişir, eğitim alt kümesi sabit.

| Grup | Koşu | Neden |
|---|---|---|
| B2-human, S-gpt, S-claude @ n10k r8 | 6 | **Tablo I'deki her eğitilmiş sistem** → R1 kapanır |
| S-gpt @ r16, r32 | 4 | **"rank 8 Pareto-optimal"** iddiası → R2(c) kapanır |

+ 10 çıkarım (MLSUM) + 6 çıkarım (TR-News) + BERTScore.

### C. İstatistik (tamamen CPU)
`src/eval/stats.py`: eşleştirilmiş bootstrap BCa %95 CI · yaklaşık randomizasyon testi (Dror et al. 2018) + Holm–Bonferroni · **TOST eşdeğerlik** · Wilson CI · McNemar.

- **δ marjı WP-B koşulmadan önce sabitlenip commit'lenecek:** **δ = 0.010 ROUGE-1**, gerekçe v1 Tablo I'deki B3a–B3b (0.002) ve S-gpt–S-claude (0.002) farklarının 5 katı ve özetleme yazınındaki 1-puan pratik anlam eşiği. **Birincil ölçüt tek: ROUGE-1 standard.**
- Testler **tohum-ortalamalı örnek skorları** üzerinde koşacak — tek tohumun tahminleri üzerinde değil, yoksa R1 ayakta kalır.
- Tohum belirsizliği **t-aralığı, df=2 (çarpan 4.30)** olarak; m=3'te çıplak std yanıltıcı dar görünür.

### D. Sadakat — 7 günde yapılabilen kısmı
- **`rouge_pr.py`** — ROUGE precision ve recall'u **ayrı** raporla. Tamamen CPU, sıfır maliyet. Öğrencinin yüksek-precision / düşük-recall profili varsa **atlama kanıtlanmış olur**. R4'e karşı en ucuz güçlü kanıt.
- **`numeric.py`** — Türkçe normalizasyon (binlik `1.500` / ondalık `3,5` / `%15` / `yüzde 15` / `15 milyon dolar` / tarih formatları / yazıyla sayılar). Yeni metrikler: **`numeric_precision`** (doğrulanan sayı ÷ toplam sayı) ve **`halluc_per_100tok`** — ikisi de **uzunluk-değişmez**, "5×" iddiasının uzunluk konfaundunu kırar. Eski ikili bayrak karşılaştırılabilirlik için korunur.
- **`entity.py`** — Türkçe NER (`savasy/bert-base-turkish-ner-cased`) + morfoloji katmanı (kesme+ek: `Ankara'da`→`Ankara`; **açık i/İ–ı/I eşlemesi**, Python `casefold()` Türkçe'de yanlış; eşleştirme: tam → normalize Levenshtein ≥0.85 → ortak 5-karakter önek). Metrikler: `entity_precision` (uydurma varlık), `entity_recall_ref` (**referanstaki varlıklardan kaçı yakalandı — birincil atlama kanıtı**), `PER_precision`/`PER_recall` ayrı.
  > `entity_recall_source` **birincil yapılmayacak**: öğrenci daha ekstraktif (senin `extract`≈0.99 bulgun), kaynak cümlelerini kopyalamak bu metriği mekanik olarak şişirir — yani atlama şüphesi olan sistemi ödüllendirir.
- **Uzunluk katmanlı karşılaştırma** — özetleri token sayısına göre kutucukla, kutucuk içinde sistemleri karşılaştır.

### E. LLM-hakem 2.0
| v1 (gönderilen) | v2 |
|---|---|
| 30 makale × 6 = 180 satır | **60 makale × 6 = 360 satır** |
| Prompt'ta `system=B3a` yazıyordu (körleme yok) | **Anonim A–F**, makale başına rastgele permütasyon |
| 600 karakter bağlam, ama STRICT faktüellik puanı | **3000 karakter** (öğretmene verilenin aynısı) |
| Kirli çıktılar | **Temiz** + kirli-vs-temiz A/B kolu |
| Claude Opus 4.7 — B3b'yi üreten modelin ailesinden | **2 bağımsız hakem, ikisi de Anthropic dışı** |
| Ham log yok (`opus_judgments.csv` 0 byte) | Her çağrının tam prompt'u + ham yanıtı diske |

**Hakemler (ikisi de ücretsiz):**
- **J-A: Gemini Flash** (Google AI Studio ücretsiz katman) — model kimliği **tarihli tam string** olarak sabitlenecek
- **J-B: Qwen2.5-7B-Instruct 4-bit**, Colab T4'te yerel — tam yeniden üretilebilir, API'ye bağımlı değil. Çıktı serbest metin gerekçe içermeyen ~40 token'lık kısa JSON'a sınırlı (T4'te 4-bit 7B ~10–15 tok/s; uzun çıktı 4 saat sürer).
- Yedek: Gemini kotası yetmezse **Gemma-2-9B-it** ikinci yerel hakem olur. **Llama-3.1-8B kullanılmayacak** — resmî desteklediği 8 dil arasında Türkçe yok, Türkçe akıcılık puanlamak savunulamaz.
- **Anthropic \$1:** ana tabloda kullanılmıyor. İsteğe bağlı olarak ~\$0.38 ile v1 protokolünün Haiku replikasyonu yapılabilir; **öneri: harcama, yedekte kalsın.**

**Analizler:** hakemler arası Cohen κ · **self-preference farkı** (her hakemin Claude-türevi sistemlere B3b/S-claude verdiği puanın diğerine göre farkı, makale bazında CI ile) · tüm geçiş oranlarına Wilson CI · sistem çiftleri için McNemar.

**v1 Tablo IV ne olacak:** Opus 4.7'yi temiz çıktılarla yeniden koşacak bütçe yok. Tablo IV **yeni hakemlerden** yeniden kurulacak; Opus sütunu *"v1, pre-strip, tek Anthropic hakemi"* etiketiyle şeffaflık için kalacak.

### F. İnsan anotasyonu — R4'ün tek gerçek kanıtı
- **40 makale × 5 sistem (B2, B3a, B3b, S-gpt, S-claude) = 200 satır.** B1 çıkarıldı — hiçbir iddia onunla ilgili değil.
- **4 eksen:** `factual_correct` · `entity_hallucination` · **`misattribution`** · **`salient_omission`**
- **Makale-gruplu tasarım:** bir makaleyi oku, 5 sistemi arka arkaya puanla; makale içinde sistem sırası rastgele, makale sırası rastgele. *(Satırları global karıştırırsan her satırda 3000 karakteri yeniden okursun → 10+ saat. Grupluyken ~3 saat.)*
- Sistem etiketleri gizli. **Körlemenin sınırı makalede yazılacak:** öğretmen özetleri referansın 2.1–3.1 katı, öğrenciler 0.74–1.43 katı uzunlukta; anonim etiket **uzunluk sinyalini kaldırmaz**.
- **Güç:** n=40/sistem, p≈0.10 için Wilson CI ≈ ±10 puan. **Minimum saptanabilir fark makalede açıkça yazılacak**; bu güçle saptanamayan farklar "fark yok" diye sunulmayacak.
- Otomatik `entity_precision` metriği bu 200 satıra karşı doğrulanacak (Cohen κ).
- **Bölümden bir arkadaş 40 satır yaparsa** (~30 dk) anotatörler arası κ da raporlanır — küçük iş, büyük kazanç.

---

## 5. Günlük plan

| Gün | Sen | Ben |
|---|---|---|
| **Per 20 (bu akşam)** | `main.tex`/IEEE kaynak dosyalarını gönder · Gemini API key al · **kabul mektubundaki sayfa limitini teyit et** | `revision-v2` dalı, `clean_predictions.py`, `stats.py`, Colab notebook #1, `PREREGISTRATION.md` (δ) |
| **Cum 21** | **Colab #1** temizlik + yeniden hesap + BERTScore + özdeşlik ispatı (~2 sa) → bitince **Colab #2** çekirdek 6 eğitim koşusu (~3 sa, gözetimsiz) | `rouge_pr.py`, `numeric.py`, temiz v1 üzerinde ilk CI tabloları |
| **Cmt 22** | **Colab #3** LoRA 4 koşu + 16 çıkarım + BERTScore (~4 sa, gözetimsiz) | `entity.py`, TOST, `judge/` paketi |
| **Paz 23** | **Colab #4** NER + 2 hakem + kirli/temiz A/B (~2.5 sa) | Çok-tohumlu istatistikler, tüm figürlerin yeniden üretimi |
| **Pzt 24** | **Anotasyon: 200 satır (~3 sa)** | Tablo I/II/III yeniden yazımı (CI'lı), §V ve §VI revizyonu |
| **Sal 25** | Taslağı oku, iddia kontrolü | §VI/§VII/abstract/conclusion + hakem yanıt notu + κ analizleri |
| **Çar 26** | Düzeltmeler, Overleaf derleme, **sayfa sığdırma** | Figür/tablo son hali, repo hijyeni, README, `RESPONSE_TO_REVIEWERS.md` |
| **Per 27** | Son okuma → **teslim** | Son kontrol, `git tag v2-camera-ready` |

**Kritik yol:** Cum 21 ve Cmt 22'deki Colab koşuları. Bunlar kayarsa her şey kayar — ikisi de büyük ölçüde gözetimsiz, başlatıp bırakabilirsin.

---

## 6. Bütçe

**GPU (Colab T4) — 10.8 sa**

| Kalem | Saat |
|---|---|
| Temizlik sonrası BERTScore (18 dosya) + özdeşlik ispatı | 1.4 |
| 10 eğitim koşusu (çekirdek 6 + LoRA 4) | 5.0 |
| 16 çıkarım koşusu | 1.7 |
| BERTScore (16 yeni tahmin dosyası) | 0.5 |
| Türkçe NER | 0.7 |
| Yerel hakem J-B (600 yargı, kısa JSON) | 1.5 |
| **Toplam** | **10.8** |

**API — \$0.00 planlanan** (\$1.00 kredi tamamen yedekte). Gemini ücretsiz katman + yerel model. Opsiyonel Haiku replikasyonu \$0.38.

**Senin zamanın — ~13 sa aktif**: Colab başlatma/izleme ~3 sa (duvar saati ~11) · anotasyon ~3 sa · makale okuma/derleme ~7 sa.

---

## 7. Makale değişiklik haritası (IEEE, 7 sayfa)

| Yer | Değişiklik |
|---|---|
| **Abstract** | "hallucinates numeric facts approximately five times less often" → uzunluk-normalize edilmiş, çok metrikli ifadeyle değiştir. Çok tohum + CI'dan bahset. Trade-off çerçevesi zaten var, korunacak. |
| **§III-D Evaluation** | Yeni metrikler (ROUGE P/R, numeric precision, entity P/R) + istatistiksel protokol (bootstrap, permütasyon, TOST, δ) |
| **Tablo I** | **Temiz sayılar** + %95 CI + tohum t-aralığı + **ROUGE P/R sütunları** |
| **Tablo II** | Aynı muamele |
| **Tablo III** | Temiz sayılarla güncelle (S-gpt satırı yükselecek) |
| **§IV-B/C** | "%91 of teacher quality", "five times less", "statistically tied" ifadeleri yeni sayılara ve CI'lara göre yeniden yazılacak — **"statistically tied" artık gerçekten test edilmiş olacak** |
| **§V Ablations** | LoRA rank iddiası CI'larla; **TOST sonucu** öğretmen-seçimi null result'ı için; boyut/prompt ablasyonlarının tek tohum olduğu belirtilecek |
| **§VI-A** | Sadakat paketi: numeric precision, entity P/R, uzunluk-normalize oranlar, uzunluk katmanlı karşılaştırma |
| **§VI-B** | **🔴 Yanlış cümle silinecek.** Yerine ölçülmüş artefakt etkisi + sistem başına sentinel oranı + kirli/temiz hakem A/B tablosu |
| **§VI-C (yeni, kısa)** | 2 bağımsız hakem + insan anotasyonu sonuçları (`misattribution`, `salient_omission`, κ değerleri) |
| **§VII-A** | Yanlış-atıf artık insan tarafından ölçülmüş — "6/30 judged" yerine gerçek oran + CI |
| **§VII-B** | Yeniden yazım: "left to future work" diyen 3 cümle → "yaptık, sonuç şu". Kalan açıklar (NLI, otomatik rol metriği, ablasyon tohumları, öğretmen varyansı, tek anotatör, n=40 MDE) dürüstçe listelenecek |
| **§VII-C** | `MANIFEST.md`, `PREREGISTRATION.md`, `v1-submitted` tag'i; ölü Gradio linki kaldırılacak |
| **§VIII Conclusion** | İki yönlü sonuç; OOD dayanıklılığı hâlâ en güçlü argüman, artık CI'lı |
| **Kaynakça** | +Dror et al. 2018 (anlamlılık) · +Lakens 2017 (TOST) · Kryściński [28] zaten var, sadakat tartışmasında kullanılacak |

**Sayfa yönetimi — 7 sayfa dolu, yer açmak şart:**
- **Fig. 1** (a+b bar grafik) Tablo I ile büyük ölçüde mükerrer → tek panele indir veya kaldır → **~½ sütun**
- **Fig. 2** Tablo II ile mükerrer → kaldır → **~⅓ sütun**
- **Tablo III + iki paragraf** → 5 satırlık metne sıkıştır → **~⅓ sütun**
- Related Work'ten ~10 satır kırp → **~¼ sütun**
- Kazanılan ~1.5 sütun ≈ **¾ sayfa** → yeni tablolar + ~20 satır yeni metin + genişletilmiş limitation'a yeter
- CI'lar mevcut hücrelere kompakt notasyonla girer, ek yer istemez

---

## 8. Senden gerekenler

| Ne zaman | İş | Süre |
|---|---|---|
| **🔴 Bu akşam** | **IEEE LaTeX kaynağını gönder** (`.tex` + `.bib` + figürler / Overleaf linki). PDF üzerinden revizyon yapılamaz. | 5 dk |
| **🔴 Bu akşam** | **Kabul mektubundaki sayfa limitini** teyit et (7 sayfa geçti ama limit 6 mı 8 mi?) ve **revizyon teslim kanalını** (e-posta / sistem) | 5 dk |
| **🔴 Bu akşam** | Google AI Studio'dan **ücretsiz Gemini API key** + günlük istek limitini not et | 10 dk |
| Bu akşam | Hakem yorumlarının **orijinal metnini** gönder (bana özet geçtin; yanıt notunda birebir alıntı gerekiyor) | 2 dk |
| Yarın (21) | Colab #1 + #2 | ~5 sa duvar, ~1 sa aktif |
| Cmt (22) | Colab #3 | ~4 sa duvar, ~30 dk aktif |
| Paz (23) | Colab #4 | ~2.5 sa duvar |
| **Pzt (24)** | **Anotasyon: 200 satır** | **~3 sa** |
| Pzt (24) | Bölümden birine 40 satır anotasyon yaptır (opsiyonel ama değerli) | 10 dk ayarlama |
| Sal–Per | Makale okuma, derleme, teslim | ~7 sa |

---

## 9. İş kayarsa kesme sırası

Önce kesilecek → sonra kesilecek:

1. **LoRA r16/r32 tohumları** (4 koşu, 2.7 sa) → "rank 8" iddiası CI'sız kalır, limitation'a yazılır
2. **İkinci hakem** (J-B yerel) → tek bağımsız hakem, yine de Anthropic dışı → R6 kısmen kapanır
3. **Hakem örneklemi 60 → 40 makale**
4. **Anotasyon 200 → 120 satır** (24 makale × 5 sistem)
5. **`entity.py`** → sadakat kanıtı ROUGE P/R + numeric precision ile sınırlı kalır

**Asla kesilmeyecek:** sentinel temizliği + tüm tabloların yeniden hesabı (§2) · bootstrap CI'lar · çekirdek 3 sistem × 3 tohum · temiz çıktılarla hakem koşumu. Bu dördü olmadan revizyon hakem maddelerini karşılamaz.

---

## 10. Sonraki adım

LaTeX kaynağını gönder, ben bu akşam `revision-v2` dalını, temizlik scriptini, istatistik modülünü ve yarın sabah çalıştıracağın Colab notebook'unu hazırlayayım. Kaynak gelene kadar makaleye dokunmam — tüm analiz işi format-bağımsız, ona paralel başlayabilirim.
