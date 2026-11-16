# Pilot lokal — PPLX classifier dan evaluator

## Pilot empat layanan melalui SSH

Dari root proyek jalankan `bash ./experimenting/run_cli.sh --task services`, atau pilih **1 (Pilot) lalu 5 (Pilot layanan SSH)**. Di Windows gunakan launcher `.bat` dengan argumen yang sama. Pilot memilih lima SPAM dan lima NOT_SPAM pertama dari dataset 3.000 email.

| Layanan | ID model | Port lokal | API dan pengujian |
|---|---|---:|---|
| PPLX | `perplexity-ai/pplx-decider-flash-v1.2` | 11438 | `/v1/systemone`, lima kriteria validitas untuk setiap pasangan pilot yang tersedia di bank rewrite |
| Clef | `clef-flash-q4` | 11439 | `/v1/systemone`, klasifikasi 10 original |
| d3 | `d3-flash-q4` | 11440 | `/v1/systemone`, klasifikasi 10 original |
| Gemma | `google/gemma-3-27b-it` | 11441 | `/v1/chat/completions`, satu pertanyaan RAG untuk memeriksa respons chat |

Di PC lokal, buka tunnel berikut pada terminal tersendiri dan biarkan aktif selama pilot. Masukkan sandi melalui prompt SSH. Pada host yang sudah menyediakan keempat port di localhost, langsung jalankan pilot.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:11438:127.0.0.1:11438 -L 127.0.0.1:11439:127.0.0.1:11439 -L 127.0.0.1:11440:127.0.0.1:11440 -L 127.0.0.1:11441:127.0.0.1:11441 -p 28659 rinaldi@37.60.243.124
```

Untuk mengganti port lokal pada satu run, sesuaikan port kiri pada `-L` lalu tambahkan misalnya `--port pplx=12438 --port gemma=12441` ke perintah pilot. Port kanan adalah port layanan di server. Untuk menyimpan perubahan, pilih menu 5 lalu **Ubah port model untuk pilot layanan SSH**, atau edit `service_pilot` pada JSON profil aktif. `--dry-run` menampilkan endpoint yang akan dipakai.

Hasil dan checkpoint tersedia di `experimenting/outputs/pilot_layanan10/<id-run>/`. Buka `ringkasan.json` untuk jumlah klasifikasi benar dan verdict per pasangan, serta `hasil.csv` untuk label, waktu dan error tiap request. Perintah yang sama melanjutkan hasil tersimpan. Tambahkan `--retry-errors` untuk mengulang request yang gagal. Perubahan endpoint, model, input, atau pertanyaan menghasilkan ID run baru.

Jika server mengembalikan HTTP 500 dengan pesan `increase the physical batch size`, periksa jumlah token yang disebutkan. Pada pilot ini batas Clef/d3 adalah 512 token dan input terpanjang memerlukan 1.890 token pada Clef. Sesudah kapasitas server disesuaikan, jalankan kembali perintah pilot dengan `--retry-errors` untuk memproses request yang gagal.

Bagian berikut mendokumentasikan workflow adaptif lama. Pilihan **1 → 4** memakai konfigurasi workflow tersebut. Hasil pilot layanan mengukur klasifikasi dan validasi pasangan tetap.

Profil ini untuk menguji pipeline sebelum Clef/d3 tersedia di server. **PPLX memiliki dua peran dengan permintaan yang terpisah**:

| Peran | Model/layanan |
|---|---|
| Classifier SPAM/NOT_SPAM | Tool PPLX `server:pplxdeciderflash` |
| Evaluator validitas rewrite | Tool PPLX yang sama, dengan rubrik dan kategori PASS/FAIL/UNCERTAIN |
| Attacker LLM | `google/gemma-3-27b-it` melalui Open WebUI |
| Attacker GAN | Generator SST lokal di CPU |

Open WebUI menggunakan `https://llms.si-lab.org/api/chat/completions`. Model penghubung tool PPLX adalah `google/gemma-3-27b-it`. Tidak ada panggilan Clef/d3, Ollama, atau port GPU lokal dari profil ini. Hasil pilot tidak mewakili robustness Clef/d3; classifier dan evaluator berbagi backend PPLX sehingga bukan penilaian yang independen.

## Jalankan pilot 10 data

Dari **root proyek**, aktifkan lingkungan Python 3.10+, lalu:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Jika `.env` sudah ada, cukup edit file itu; jangan menimpanya. Isi `OWUI_API_KEY` dengan key sendiri. Script membaca `.env` root secara otomatis.

```powershell
.\experimenting\run_cli.bat
```

Pada Git Bash/Linux:

```bash
bash experimenting/run_cli.sh
```

Pilih **1 — Pilot 10 data**, lalu **Gemma + GAN**. Pilihan ini menjalankan baseline, cabang Gemma, dan cabang GAN. CLI memeriksa bobot serta runtime GAN sebelum baseline. Jika bobot belum tersedia, CLI menyiapkannya sekali secara otomatis; jika runtime gagal, belum ada panggilan classifier. Persiapan manual tetap tersedia melalui **5 — Pengaturan → Siapkan bobot GAN**.

Tanpa menu:

```powershell
# Offline; tidak memanggil API dan tidak mengunduh bobot
.\experimenting\run_cli.bat --task all --dry-run

# Baseline + Gemma + GAN + metrik validitas/klasifikasi/ASR
.\experimenting\run_cli.bat --task all

# Hanya baseline dan Gemma
.\experimenting\run_cli.bat --task llm

# Hanya baseline dan GAN
.\experimenting\run_cli.bat --task gan

# Resume dan ulangi tahap error saja
.\experimenting\run_cli.bat --task all --retry-errors
```

Dataset sama dengan main: `dataset/paper_baseline3000.csv`. Pilot memilih 5 SPAM + 5 NOT_SPAM original. Hanya SPAM yang terdeteksi SPAM oleh PPLX yang masuk rewrite; jumlah rewrite dapat kurang dari lima. Tidak ada penyeimbangan ulang untuk memaksa jumlah serangan tertentu.

## Alur masing-masing cabang

```text
Baseline PPLX -> true-positive SPAM
  -> rewrite 1 (Gemma ATAU GAN)
  -> PPLX evaluator membandingkan original dengan rewrite
      INVALID -> catat kriteria gagal -> berhenti -> data berikutnya
      UNCERTAIN / NO_CHANGE -> catat terpisah -> berhenti
      VALID -> PPLX classifier SPAM/NOT_SPAM
          NOT_SPAM -> flip valid -> berhenti
          SPAM -> rewrite 2 dari kandidat pertama
              -> PPLX evaluator membandingkan ORIGINAL dengan rewrite 2
                  INVALID / UNCERTAIN / NO_CHANGE -> catat -> berhenti
                  VALID -> PPLX classifier -> berhenti
```

Maksimal dua step berlaku untuk Gemma dan GAN. Error API atau respons tidak terverifikasi dicatat sebagai ERROR, bukan INVALID. Tidak ada klasifikasi jika validasi belum berhasil.

Rubrik tetap: tujuan komunikasi, informasi penting, makna, tidak menambah klaim material, dan koherensi. Semua lima kriteria harus PASS. Kode dan instruksi evaluator: `scripts/rewrite_validity.py`. Prompt attacker Gemma: `decision_models/paper_prompts.json`, tetap memakai prompt sumber. Teks email diperlakukan sebagai data, bukan instruksi evaluator.

Untuk audit per kriteria, satu kandidat memerlukan hingga lima panggilan PPLX evaluator. Jadi pilot 10 original bukan hanya 10 panggilan API. Respons per kriteria dicache; resume tidak mengulang kriteria yang sudah selesai. Output assistant biasa tidak diterima sebagai keputusan PPLX. Jika model penghubung mengubah argumen tool, run menyimpan ERROR agar tidak menggunakan keputusan untuk rubrik yang salah.

GAN menghasilkan kandidat sebelum query defender internal. Dua step dan validity gate merupakan perluasan eksperimen sumber. Generator SST dapat menghasilkan teks tidak koheren; teks tersebut tetap disimpan dan ditolak sesuai rubrik, bukan dihitung sebagai serangan berhasil.

## Hasil yang diperiksa

**Buka `experimenting/outputs/BUKA_HASIL.md` atau langsung `experimenting/outputs/hasil_pilot10/RINGKASAN.md`.** Semua CSV utama sudah dikumpulkan di folder `hasil_pilot10`:

| File | Dibaca untuk |
|---|---|
| `RINGKASAN.md` | Status tiap tugas, termasuk GAN yang belum dijalankan |
| `baseline.csv` | Klasifikasi original |
| `rekap_per_email.csv` | Status akhir Gemma dan GAN dalam satu tabel |
| `gemma_step1.csv`, `gemma_step2.csv` | Teks dan hasil rewrite Gemma |
| `gan_step1.csv`, `gan_step2.csv` | Teks dan hasil rewrite GAN |
| `validitas_per_step.csv` | Kriteria validitas dan jumlah penolakan |
| `errors.csv` | Error tahap yang belum terselesaikan |

Laporan diperbarui setelah tiap tugas. Untuk memperbarui hasil lama tanpa API, gunakan menu **4 → Perbarui ringkasan pilot** atau:

```powershell
.\experimenting\run_cli.bat --results
```

CSV kosong/header-only ditandai sebagai belum ada hasil, bukan hasil nol. `INVALID` merupakan keputusan validator, bukan error program. Jangan mengedit laporan untuk mengubah checkpoint.

Folder berikut tetap menjadi **arsip teknis**; Anda tidak perlu membukanya untuk membaca hasil sehari-hari. Struktur dan identitas lama dipertahankan agar baseline/Gemma yang sudah selesai bisa di-resume tanpa mengulang inference.

Folder default, masing-masing dengan subfolder `pplx`:

- `experimenting/outputs/validated_baseline_pilot10_v1`
- `experimenting/outputs/validated_llm_pilot10_v1`
- `experimenting/outputs/validated_gan_pilot10_v1`

Kedua cabang menghasilkan `step1_rewrites.csv`, `step2_rewrites.csv`, `validity_summary.csv`, `sample_results.csv`, `metrics.json`, dan CSV individual di `rewrites/step1` serta `rewrites/step2`. Kolom mencakup hasil setiap kriteria, alasan penolakan, status klasifikasi/skipped, serta waktu generation/validation/classification. Waktu adalah durasi tahap, bukan waktu GPU murni.

Metrik klasifikasi, acceptance rate dan ASR dihitung otomatis. **Evaluasi kemiripan paper yang lengkap** (SBERT, BLEU, ROUGE-L, Jaccard, length ratio dan keyword retention) dijalankan lewat **menu 4 — Hasil/evaluasi**, tanpa mengulangi inference. Pilihan leksikal tidak membutuhkan SBERT. Invalid/uncertain tetap dapat dianalisis sebagai teks, tetapi tidak dihitung sebagai flip valid.

## Resume dan konfigurasi

Jalankan perintah yang sama untuk resume. `--retry-errors` hanya mengulang tahap error; INVALID tidak diulang. Checkpoint serta raw response disimpan per tahap. Hasil lama tanpa validity gate tidak dimasukkan otomatis ke run baru.

Konfigurasi dasar berada pada `experimenting/adaptive_config.json`; perubahan menu disimpan pada `experimenting/adaptive_local.json`. Profil lokal lama tanpa evaluator ditolak dengan petunjuk memilih konfigurasi baru. Gunakan `--config experimenting/adaptive_config.json` untuk memilih konfigurasi dasar secara eksplisit.

Pilot mengizinkan revision backend belum terverifikasi dan menandainya `UNVERIFIED_WEIGHTS`. Metadata katalog tidak sama dengan revision bobot. Untuk hasil penelitian, audit manual sekitar 50 pasangan rewrite terlebih dahulu, lalu bekukan rubrik dan konfigurasi. Main memakai Clef/d3 untuk klasifikasi, sementara PPLX tetap evaluator.

Status validasi implementasi: tersedia pengujian offline dengan respons model simulasi untuk gate, dua step, retry, resume, parser tool dan pemisahan peran. Keberhasilan respons layanan nyata dan kualitas keputusan PPLX harus diverifikasi melalui pilot Anda.
