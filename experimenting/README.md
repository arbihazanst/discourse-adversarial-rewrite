# Experimenting melalui SSH — Clef/d3 classifier dan PPLX evaluator

## Membiarkan tunnel tetap aktif

Dari root proyek, buka terminal pertama dan jalankan:

```sh
python scripts/open_model_tunnel.py
```

Masukkan password jika diminta. Setelah muncul `Port lokal siap`, gunakan terminal kedua untuk memanggil model atau menjalankan uji menu 7. Tunnel tetap aktif setelah uji selesai selama terminal pertama terbuka. Tekan Ctrl+C di terminal pertama untuk menghentikan forward milik skrip. Jika semua port sudah digunakan, periksa layanan dengan menu 7.

## Uji tunnel SSH

Dari root proyek di PowerShell, jalankan `.\experimenting\run_cli.bat --task tunnel` atau pilih menu utama **7**. Di Bash gunakan launcher `.sh` dengan argumen yang sama. Tambahkan `--dry-run` untuk melihat alamat tanpa membuka koneksi.

Launcher memakai port lokal yang sudah tersedia dan membuka forward yang masih dibutuhkan. Jika diminta, ketik password di prompt OpenSSH. Tunnel yang dibuka launcher ditutup setelah pengujian. PPLX (11438), Clef (11439), dan d3 (11440) menerima pertanyaan refund melalui `/v1/systemone`. Gemma (11441) menerima pertanyaan RAG melalui `/v1/chat/completions`. Output menampilkan jawaban dan waktu tiap layanan. Menu ini membaca endpoint dan pemetaan SSH yang sama dengan pekerjaan `all`.

Profil experimenting memakai Clef/d3 sebagai classifier, PPLX sebagai evaluator dan Gemma sebagai generator. Pipeline serta dataset sama dengan main. Koneksi melewati SSH dari PC.

| Layanan | Port PC | Port localhost server |
|---|---:|---:|
| PPLX | 11438 | 11438 |
| Clef Flash | 11439 | 11439 |
| d3 Flash | 11440 | 11440 |
| Gemma | 11441 | 11441 |

Launcher menjaga tunnel selama seluruh pekerjaan `all`. Login dilakukan melalui prompt OpenSSH jika diperlukan. Tunnel yang sudah aktif dipakai kembali. Untuk mengganti host SSH, port login atau port server, edit `ssh` pada profil aktif; URL model menentukan port lokal. Skrip `open_model_tunnel.py` memakai profil experimenting yang sama.

## Jalankan pilot 10 data

Dari **root proyek**, aktifkan lingkungan Python 3.10+, lalu:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Profil default memakai API lokal melalui SSH. Kredensial login diminta OpenSSH di terminal. Jika endpoint memerlukan key, atur `key_env` melalui menu 5 dan isikan nilainya pada `.env`.

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

Dataset sama dengan main: `dataset/paper_baseline3000.csv`. Pilot memilih 5 SPAM + 5 NOT_SPAM original. Hanya SPAM yang terdeteksi SPAM oleh defender terkait yang masuk rewrite; jumlah rewrite dapat kurang dari lima. Tidak ada penyeimbangan ulang untuk memaksa jumlah serangan tertentu.

## Alur masing-masing cabang

```text
Baseline Clef/d3 -> true-positive SPAM
  -> rewrite 1 (Gemma ATAU GAN)
  -> PPLX evaluator membandingkan original dengan rewrite
      INVALID -> catat kriteria gagal -> berhenti -> data berikutnya
      UNCERTAIN / NO_CHANGE -> catat terpisah -> berhenti
      VALID -> Clef/d3 classifier SPAM/NOT_SPAM
          NOT_SPAM -> flip valid -> berhenti
          SPAM -> rewrite 2 dari kandidat pertama
              -> PPLX evaluator membandingkan ORIGINAL dengan rewrite 2
                  INVALID / UNCERTAIN / NO_CHANGE -> catat -> berhenti
                  VALID -> Clef/d3 classifier -> berhenti
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

Folder berikut tetap menjadi **arsip teknis**; Anda tidak perlu membukanya untuk membaca hasil sehari-hari. Folder baru dipakai untuk profil tunnel agar endpoint dan revision terbaru memiliki checkpoint sendiri.

Folder default, masing-masing dengan subfolder `clef` dan `d3`:

- `experimenting/outputs/tunnel_v1/pilot_baseline`
- `experimenting/outputs/tunnel_v1/pilot_llm`
- `experimenting/outputs/tunnel_v1/pilot_gan`

Kedua cabang menghasilkan `step1_rewrites.csv`, `step2_rewrites.csv`, `validity_summary.csv`, `sample_results.csv`, `metrics.json`, dan CSV individual di `rewrites/step1` serta `rewrites/step2`. Kolom mencakup hasil setiap kriteria, alasan penolakan, status klasifikasi/skipped, serta waktu generation/validation/classification. Waktu adalah durasi tahap, bukan waktu GPU murni.

Metrik klasifikasi, acceptance rate dan ASR dihitung otomatis. **Evaluasi kemiripan paper yang lengkap** (SBERT, BLEU, ROUGE-L, Jaccard, length ratio dan keyword retention) dijalankan lewat **menu 4 — Hasil/evaluasi**, tanpa mengulangi inference. Pilihan leksikal tidak membutuhkan SBERT. Invalid/uncertain tetap dapat dianalisis sebagai teks, tetapi tidak dihitung sebagai flip valid.

## Resume dan konfigurasi

Jalankan perintah yang sama untuk resume. `--retry-errors` hanya mengulang tahap error; INVALID tidak diulang. Checkpoint serta raw response disimpan per tahap. Hasil lama tanpa validity gate tidak dimasukkan otomatis ke run baru.

Konfigurasi dasar berada pada `experimenting/adaptive_config.json`; perubahan menu disimpan pada `experimenting/adaptive_local.json`. Profil lokal lama tanpa evaluator ditolak dengan petunjuk memilih konfigurasi baru. Gunakan `--config experimenting/adaptive_config.json` untuk memilih konfigurasi dasar secara eksplisit.

Pilot dan benchmark mengizinkan revision kosong dan mencatat `UNVERIFIED_WEIGHTS`. Gunakan folder hasil baru setelah mengganti bobot. Benchmark memakai `experimenting/outputs/tunnel_v1/full_llm` dan `full_gan`. Metadata katalog tidak sama dengan revision bobot. Untuk hasil penelitian, audit manual sekitar 50 pasangan rewrite terlebih dahulu, lalu bekukan rubrik dan konfigurasi. Main memakai Clef/d3 untuk klasifikasi, sementara PPLX tetap evaluator.

Status validasi implementasi: tersedia pengujian offline dengan respons model simulasi untuk gate, dua step, retry, resume, parser tool dan pemisahan peran. Keberhasilan respons layanan nyata dan kualitas keputusan PPLX harus diverifikasi melalui pilot Anda.
