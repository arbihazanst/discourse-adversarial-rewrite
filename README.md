# Decision Research — Main

Pipeline server untuk mengklasifikasikan **3.000 email**, membuat rewrite dengan **Gemma atau GAN**, lalu menguji perubahan label **SPAM → NOT_SPAM**.

## Mulai

Gunakan Python 3.10+ pada environment yang sama untuk CLI dan dependensi. Jika belum ada .env, salin [.env.example](.env.example), lalu isi **OWUI_API_KEY**. Jangan menimpa file yang sudah berisi key.

| Terminal                  | Perintah dari root proyek |
| ------------------------- | ------------------------- |
| Linux / server / Git Bash | bash run_cli.sh           |
| PowerShell / CMD          | .\run_cli.bat             |

Hanya dua launcher main tersebut yang diperlukan. Launcher lama per model/metode sudah dihapus; semua pilihan tersedia dalam CLI.

## Menu

```text
1. Pilot 10 data - mulai di sini
2. Benchmark seluruh 3.000 data
3. Resume / ulangi tahap yang error
4. Hasil CSV dan evaluasi
5. Pengaturan model, endpoint, output, dan GAN
6. Panduan singkat
0. Keluar
```

**Pilot lengkap:** pilih **1 → 4 (Gemma + GAN) → 1 (Jalankan)**. Pilot PPLX mengambil 5 spam dan 5 ham dari dataset utama. Pilih **Periksa konfigurasi offline** untuk pemeriksaan tanpa API.

**Baseline saja:** pilih **1 → 1 → 1**. Pilihan ini tidak menghasilkan rewrite.

**Benchmark penuh:** isi revision/checkpoint sebenarnya melalui Pengaturan, lalu pilih **2**, metode, dan Jalankan. Model lain dapat dipilih di Pengaturan; pilot koneksi khusus PPLX.

Sebelum eksekusi, CLI memperlihatkan ukuran data, defender, urutan tahap, dan folder hasil. Jika satu tahap error, tahap berikutnya berhenti.

## Model dan endpoint

| Peran               | Konfigurasi main                                         |
| ------------------- | -------------------------------------------------------- |
| Pemanggil PPLX Tool | google/gemma-3-27b-it melalui Open WebUI                 |
| Tool classifier     | server:pplxdeciderflash                                  |
| Open WebUI          | https://llms.si-lab.org/api/chat/completions             |
| Attacker Gemma      | SGLang http://127.0.0.1:30007/v1/chat/completions        |
| Attacker model ID   | google/gemma-3-27b-it; cocokkan dengan /v1/models server |
| GAN                 | Bobot SST lokal di CPU                                   |
| Clef / d3           | Open WebUI Pipe decision_spam.clef / decision_spam.d3    |

PPLX adalah defender aktif bawaan. Clef/d3 belum diaktifkan untuk pilihan semua model. Pipe Clef/d3 menargetkan **llama.cpp /v1/systemone**, port host **30005 / 30006**; pada jaringan Docker gunakan nama container dan port internal 8080. Layanan ini tidak memakai Ollama. CLI tidak mengelola Docker.

Gemma pemanggil tool dan Gemma attacker mempunyai peran berbeda. Label hanya diterima dari hasil PPLX Tool terverifikasi; request rewrite tidak meminta tool.

Profil bawaan: [adaptive_main.json](decision_models/adaptive_main.json). Perubahan menu disimpan ke **decision_models/adaptive_local.json**, lalu otomatis dipakai saat membuka CLI lagi. Model/endpoint/revision pada profil menjadi acuan; .env menyediakan kredensial. Ubah route melalui menu. API key tidak ditampilkan atau disimpan dalam JSON.

```bash
# Memilih profil eksplisit
bash run_cli.sh --config decision_models/adaptive_main.json
```

## Alur sesuai source eksperimen

Input: [paper_baseline3000.csv](dataset/paper_baseline3000.csv), **1.500 spam + 1.500 ham**. Teks, urutan, dan duplikat cohort yang dipulihkan dipertahankan. Prediksi GPT-OSS lama tidak menjadi prediksi defender baru. [Provenance](dataset/paper_baseline3000.provenance.json).

```text
Original -> klasifikasi baseline
  |
  +-- true SPAM + prediksi SPAM -> eligible
  |      |
  |      +-- Gemma: rewrite 1 -> klasifikasi
  |      |             jika masih SPAM -> rewrite 2 -> klasifikasi
  |      |
  |      +-- GAN SST: kandidat -> probe -> klasifikasi akhir
  |
  +-- HAM / spam yang lolos baseline -> tidak di-attack
```

Gemma menggunakan dua prompt dari **OpenAttack-Share/LLMattLLM.py**. Step 2 hanya berjalan bila Step 1 masih SPAM. GAN mengikuti default SST **GANAttacker()** pada GANSSILAB.py; kandidat yang ditolak tetap tercatat dan original digunakan jika tidak ada adversarial yang dikembalikan.

Kedua cabang independen dan memakai original yang sama. **GAN tidak menyerang hasil Gemma.**

## Persiapan GAN

Di environment Python yang digunakan CLI:

```bash
python -m pip install -r scripts/requirements-gan.txt
```

Buka **5 Pengaturan → 6 Siapkan bobot GAN**. Enter pada folder sumber mengunduh resource resmi; isi folder jika bobot sudah tersedia. Folder bobot dan seed dapat diubah melalui Pengaturan.

## Resume dan hasil

Gunakan **3 Resume**, pilih ukuran run dan metode yang sama, lalu Lanjutkan + ulangi error. Hasil OK tidak diulang. Ctrl+C menghentikan proses dengan checkpoint. Simpan folder output beserta events.csv dan responses/.

Perubahan dataset, model, endpoint, revision, atau prompt memerlukan output baru. Jangan menghapus checkpoint untuk mencampur eksperimen.

| Pekerjaan                  | Folder default                         |
| -------------------------- | -------------------------------------- |
| Pilot baseline             | outputs/pplx_gemma_fixed_pilot10/pplx/ |
| Pilot Gemma                | outputs/pplx_llm_pilot10/pplx/         |
| Pilot GAN                  | outputs/pplx_gan_pilot10/pplx/         |
| Benchmark Gemma / baseline | outputs/paper3000_adaptive/<model>/    |
| Benchmark GAN              | outputs/paper3000_gan/<model>/         |

File utama: **sample_results.csv**, **metrics.csv**, **step1_rewrites.csv**, **step2_rewrites.csv**, atau **gan_rewrites.csv**. CSV individual berada di rewrites/step1/, rewrites/step2/, dan rewrites/gan/. Waktu klasifikasi/generation tersimpan di CSV; latency API mencakup jaringan, orchestration, dan tool.

**4 Hasil CSV dan evaluasi** menampilkan run tersimpan. Pilih evaluasi leksikal tanpa model embedding, atau evaluasi lengkap untuk SBERT. Evaluasi tidak memanggil classifier/attacker lagi. [Definisi metrik](decision_models/EVALUATION.md).

## Tanpa menu interaktif

```bash
# Tanpa API atau inferensi
bash run_cli.sh --task all --dry-run

# Pilot baseline + Gemma + GAN
bash run_cli.sh --task all

# Lanjutkan pilot, ulangi hanya ERROR
bash run_cli.sh --task all --retry-errors

# Benchmark penuh setelah revision siap
bash run_cli.sh --mode full --task all

# Satu model yang sudah tersedia
bash run_cli.sh --mode full --task gan --model clef
```

Opsi sama tersedia pada run_cli.bat. **--task**: baseline, llm, gan, all. **--mode**: pilot (default), full. **--show** menampilkan profil tanpa API. **--check-server** bersama --task memeriksa layanan/dependensi; pada benchmark penuh pemeriksaannya hanya konfigurasi/revision/resume, bukan bukti inferensi berhasil.

## Jika ada masalah

| Gejala                   | Tindakan                                                                        |
| ------------------------ | ------------------------------------------------------------------------------- |
| Tidak ada rewrite        | Pilih Gemma/GAN, bukan baseline; periksa jumlah eligible                        |
| Key tidak terbaca        | Isi OWUI_API_KEY dalam .env root; nilai terminal nonkosong lebih diprioritaskan |
| Timeout                  | Lanjutkan melalui menu 3; hasil valid tetap tersimpan                           |
| Parameter tool berbeda   | Periksa raw response; teks dan kategori harus sama dengan kontrak SPAM/NOT_SPAM |
| Identitas resume berbeda | Kembalikan konfigurasi yang sama atau pilih output baru                         |
| GAN belum siap           | Instal dependensi, siapkan bobot melalui Pengaturan                             |
| CSV terkunci             | Tutup file di Excel, lalu resume                                                |

Pilot mengizinkan revision belum diketahui dengan penanda **UNVERIFIED_WEIGHTS**. Itu bukan benchmark dengan bobot terpin. Label flip belum membuktikan makna tetap terjaga. Layanan SGLang GPU server harus tersedia saat main dijalankan.

Detail: [konfigurasi](decision_models/ADAPTIVE_README.md) · [GAN](decision_models/GAN.md) · [evaluasi](decision_models/EVALUATION.md).
