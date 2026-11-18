# Ringkasan hasil benchmark

Profil: **main**. Diperbarui: 2026-11-17T11:22:26+00:00.

## Mulai membaca di sini

| File | Isi |
|---|---|
| [baseline.csv](baseline.csv) | Hasil klasifikasi original |
| [rekap_per_email.csv](rekap_per_email.csv) | Status akhir tiap email, Gemma dan GAN dalam satu tabel |
| [gemma_step1.csv](gemma_step1.csv), [gemma_step2.csv](gemma_step2.csv) | Teks rewrite Gemma dan hasil validasi/klasifikasi |
| [gan_step1.csv](gan_step1.csv), [gan_step2.csv](gan_step2.csv) | Teks rewrite GAN dan hasil validasi/klasifikasi |
| [validitas_per_step.csv](validitas_per_step.csv) | Jumlah diterima/ditolak serta alasan per step |
| [errors.csv](errors.csv) | Error tahap yang belum terselesaikan; error lama yang sudah pulih tidak ditampilkan |

## Status tugas

| Defender | Tugas | Status | Original | Eligible | Flip valid | Invalid | Tetap SPAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clef | baseline | INCOMPLETE | 3000 | 1184 |  |  |  |
| clef | llm | BASELINE SAJA; REWRITE BELUM DIJALANKAN | 3000 | 1184 | 0 | 0 | 0 |
| clef | gan | BELUM DIJALANKAN |  |  |  |  |  |
| d3 | baseline | INCOMPLETE | 3000 | 1146 |  |  |  |
| d3 | llm | BASELINE SAJA; REWRITE BELUM DIJALANKAN | 3000 | 1146 | 0 | 0 | 0 |
| d3 | gan | BELUM DIJALANKAN |  |  |  |  |  |

**BELUM DIJALANKAN bukan hasil nol atau gagal.** CSV kosong/header-only berarti belum ada baris tersimpan untuk tahap tersebut. Status RUNNING/INCOMPLETE berarti ringkasan bisa belum memuat semua progres. Kolom Flip/Invalid/Tetap SPAM pada baseline tidak digunakan untuk menilai serangan.

INVALID adalah hasil validasi, bukan error program. Hanya VALID boleh diklasifikasikan. Tidak ada Step 2 jika Step 1 ditolak atau sudah berhasil mengubah label. Bila classifier dan evaluator sama-sama PPLX, hasil ini belum mengukur ketahanan Clef/d3.

## Cara melanjutkan

Gunakan menu utama **1 → 4 → 1** untuk melanjutkan kedua cabang; tahap OK dilewati. Gunakan menu 3 / `--retry-errors` hanya jika ingin mencoba ulang tahap ERROR. CLI memeriksa bobot dan runtime GAN sebelum memulai pekerjaan yang mencakup GAN.

Laporan ini adalah salinan baca dari hasil tersimpan, bukan checkpoint. Jangan mengedit CSV laporan untuk mengubah status run. Menu **4 → Perbarui ringkasan** membangunnya kembali tanpa API. Evaluasi kemiripan lengkap tersedia melalui menu 4 → Evaluasi/arsip teknis.

## Arsip teknis (hanya bila diperlukan)

Event log, respons mentah, CSV individual dan cache tetap berada di folder run asal agar resume dan audit tetap cocok. Lokasi hasil asli:
- clef / baseline: [folder run](../paper3000_validated_llm_v1/clef).
- clef / llm: [folder run](../paper3000_validated_llm_v1/clef).
- d3 / baseline: [folder run](../paper3000_validated_llm_v1/d3).
- d3 / llm: [folder run](../paper3000_validated_llm_v1/d3).
