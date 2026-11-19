# Decision-model adversarial rewrite — main/server

## Koneksi main di server

Main mengakses layanan pada localhost server langsung. Pilih menu utama **7** atau jalankan `bash run_cli.sh --task tunnel` untuk menguji endpoint profil. Nama argumen `tunnel` dipertahankan untuk kompatibilitas CLI; profil main memakai koneksi HTTP langsung.

Pada pemeriksaan 19 November 2026, host SSH `37.60.243.124` menyediakan layanan pada 11438–11441. Jika kode dijalankan di host GPU lain, sesuaikan URL melalui menu 5 dengan port yang aktif di host tersebut.

Main menjalankan **Clef dan d3 sebagai defender**, **PPLX sebagai evaluator validitas**, serta dua jalur rewrite terpisah: **Gemma 3** dan **GAN SST**. Setiap jalur maksimal dua step. PPLX tidak menjadi classifier spam/ham pada profil main.

## Mulai di server

Dari root proyek, gunakan Python 3.10+:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
bash run_cli.sh
```

Endpoint default memakai API lokal tanpa key. Jika suatu layanan memerlukan autentikasi, isi variabel pada `key_env` dan simpan nilainya di `.env`.

| Peran | Model | Endpoint localhost |
|---|---|---|
| Clef | clef-flash-q4 | http://127.0.0.1:11439/v1/systemone |
| d3 | d3-flash-q4 | http://127.0.0.1:11440/v1/systemone |
| Gemma | google/gemma-3-27b-it | http://127.0.0.1:11441/v1/chat/completions |
| PPLX evaluator | perplexity-ai/pplx-decider-flash-v1.2 | http://127.0.0.1:11438/v1/systemone |

`revision` boleh kosong untuk pilot maupun benchmark. Run mencatat `UNVERIFIED_WEIGHTS` dan status yang sama dalam kontrak checkpoint. Isi revisi bobot sebenarnya bila diketahui. Jika bobot berganti, gunakan folder hasil baru atau isi revisinya agar hasil lama tidak dipakai ulang. Kesamaan ID model dan endpoint saja tidak menjamin kesamaan bobot.

Konfigurasi dasar: `decision_models/adaptive_main.json`. Perubahan menu disimpan pada `decision_models/adaptive_local.json`, yang mendapat prioritas. Untuk memilih konfigurasi dasar secara eksplisit, tambahkan `--config decision_models/adaptive_main.json`.

## Menu dan perintah

1. **Pilot:** 10 original, terdiri dari 5 SPAM + 5 NOT_SPAM.
2. **Benchmark:** seluruh dataset.
3. **Resume/retry:** lanjutkan pekerjaan yang sama; ulangi hanya tahap error.
4. **Hasil/evaluasi:** lihat CSV dan hitung metrik kemiripan tanpa memanggil attacker/defender.
5. **Pengaturan:** model, endpoint, evaluator, folder output, seed dan persiapan bobot GAN.
6. **Bantuan.**

```bash
# Pemeriksaan offline: tidak menghubungi model atau mengunduh bobot
bash run_cli.sh --mode pilot --task all --model all --dry-run

# Pilot server: baseline + Gemma + GAN
bash run_cli.sh --mode pilot --task all --model clef

# Benchmark penuh setelah konfigurasi dan pilot diverifikasi
bash run_cli.sh --mode full --task all --model all

# Lanjutkan dan ulangi hanya error
bash run_cli.sh --mode full --task all --model all --retry-errors
```

`--task` menerima `baseline`, `llm`, `gan`, atau `all`. Dua cabang memakai ulang baseline yang identitas dataset/modelnya cocok. Windows memakai `run_cli.bat` dengan argumen yang sama. Tidak ada launcher tambahan per model.

## Dataset dan alur

Input adalah `dataset/paper_baseline3000.csv`: 3.000 original, 1.500 SPAM + 1.500 NOT_SPAM. Baseline mengklasifikasikan semua input terpilih. Rewrite hanya berlaku untuk original berlabel SPAM yang juga diprediksi SPAM oleh defender terkait. Pool Clef dan d3 dapat berbeda.

```text
Baseline Clef/d3
  -> pilih true-positive SPAM
  -> Gemma ATAU GAN: rewrite 1
  -> validasi PPLX terhadap ORIGINAL
       INVALID / UNCERTAIN / NO_CHANGE -> catat -> selesai untuk data/jalur ini
       VALID -> klasifikasi defender
           NOT_SPAM -> flip valid -> selesai
           SPAM -> rewrite 2 dari kandidat pertama
               -> validasi PPLX terhadap ORIGINAL
                   INVALID / UNCERTAIN / NO_CHANGE -> catat -> selesai
                   VALID -> klasifikasi defender -> selesai
```

Error teknis tidak berarti INVALID: simpan error, jangan klasifikasikan, lalu gunakan resume/retry. Invalid Step 1 tidak mendapat Step 2. Tidak ada percobaan rewrite ketiga.

Rubrik berada di `scripts/rewrite_validity.py`: tujuan komunikasi, informasi penting, konsistensi makna, tidak menambah klaim material, dan koherensi. Masing-masing dinilai dengan kategori `PASS/FAIL/UNCERTAIN`. Semua PASS menghasilkan VALID; satu FAIL menghasilkan INVALID; tanpa FAIL tetapi ada UNCERTAIN menghasilkan UNCERTAIN. Lima panggilan tool per kandidat disimpan dalam cache per kriteria. Alasan penolakan adalah nama kriteria yang gagal, bukan penjelasan buatan model chat.

## Gemma dan GAN

Prompt Gemma tetap berasal dari `decision_models/paper_prompts.json`, sesuai prompt sumber yang sudah ada. Rubrik validasi merupakan tambahan protokol baru, tidak mengganti prompt attacker.

GAN memakai generator SST dari sumber OpenAttack yang disertakan. Kandidat diambil **sebelum** query defender internal; kemudian melewati PPLX dan baru boleh diklasifikasikan. Step 2 GAN adalah perluasan protokol, bukan klaim reproduksi identik dengan eksperimen sumber yang satu langkah. Seed dibedakan menurut sampel dan step. Kualitas bahasa GAN tidak dijamin; teks rusak dapat ditolak evaluator.

Saat menjalankan tugas yang mencakup GAN, CLI memeriksa bobot dan memuat runtime GAN **sebelum baseline**. Jika bobot belum ada, CLI mengunduhnya sekali dari sumber resmi OpenAttack. Jika runtime belum siap, tugas berhenti sebelum memanggil classifier. `--dry-run` tidak menyiapkan bobot; `--check-server` hanya memverifikasi bobot yang sudah ada. Persiapan manual tetap tersedia melalui menu 5 atau:

```bash
python scripts/prepare_gan.py
```

Pip tidak mengunduh bobot GAN; persiapan otomatis dilakukan oleh CLI saat menjalankan tugas GAN. Skrip pipeline langsung tetap memerlukan bobot yang sudah disiapkan.

## Output, checkpoint, dan evaluasi

**Mulai membaca dari `outputs/BUKA_HASIL.md`.** Laporan utama berada dalam satu folder `outputs/hasil_pilot10` (atau `outputs/hasil_benchmark`):

- `RINGKASAN.md`: status baseline, Gemma dan GAN serta tautan CSV.
- `baseline.csv`, `rekap_per_email.csv`: klasifikasi original dan status akhir kedua cabang.
- `gemma_step1.csv`, `gemma_step2.csv`, `gan_step1.csv`, `gan_step2.csv`: rewrite beserta validasi, label dan waktu.
- `validitas_per_step.csv`, `errors.csv`: rekap validitas dan error tahap yang belum pulih.

CLI memperbarui laporan setelah tiap tugas, termasuk jika tugas mengembalikan error. Menu **4 → Perbarui ringkasan** atau `bash run_cli.sh --results` membangunnya dari hasil tersimpan tanpa inference. CSV kosong pada tugas yang belum dijalankan tidak berarti hasil nol. Laporan ini adalah salinan untuk dibaca; checkpoint dan artefak asli tetap di folder teknis berikut.

Pilot disimpan dalam `outputs/direct_v1/pilot_baseline`, `pilot_llm`, dan `pilot_gan`. Benchmark memakai `outputs/direct_v1/full_llm` dan `full_gan`, dengan subfolder defender. Folder baru ini dipilih agar identitas endpoint dan revision terbaru tidak bertabrakan dengan checkpoint sebelumnya.

| File | Isi |
|---|---|
| `step1_rewrites.csv`, `step2_rewrites.csv` | Kandidat, status/kriteria validitas, klasifikasi jika dilakukan, dan waktu per tahap |
| `rewrites/step1/*.csv`, `rewrites/step2/*.csv` | Satu CSV tersendiri per sampel dan step |
| `validity_summary.csv` | Jumlah valid/invalid/uncertain/error/no-change dan alasan per step |
| `sample_results.csv` | Status akhir setiap original dan waktu proses |
| `metrics.json`, `metrics.csv` | Baseline, valid flip, acceptance rate dan ASR |
| `events.csv`, `responses/`, `checkpoint.json`, `run.json` | Log tahap, respons mentah, identitas run, dan resume |

CSV individual ditulis setiap tahap; ringkasan diperbarui berkala dan pada akhir/interupsi. Resume memakai event log yang disimpan setelah tiap tahap. Identitas mencakup dataset, konfigurasi model, prompt, rubrik dan kontrak generator. Gunakan folder baru jika kontrak berubah. Hasil lama tanpa validasi tetap terpisah.

ASR keseluruhan = jumlah original unik dengan flip VALID dalam maksimal dua step / pool true-positive SPAM. ASR kandidat valid dilaporkan terpisah. Denominator nol ditulis kosong/null; error yang belum selesai menghasilkan batas bawah/atas, bukan ASR final. Waktu pada CSV adalah waktu tahap yang diamati, termasuk overhead/cache; bukan waktu GPU murni atau total seluruh retry historis.

Menu 4 menyediakan evaluasi SBERT, BLEU, ROUGE-L, Jaccard, length ratio dan keyword retention dari teks tersimpan. Evaluasi leksikal dapat berjalan tanpa unduhan SBERT. Hasil validasi PPLX tetap dibedakan dari skor kemiripan dan validasi manusia. Kalibrasikan rubrik dengan audit manual sebelum benchmark penuh.
