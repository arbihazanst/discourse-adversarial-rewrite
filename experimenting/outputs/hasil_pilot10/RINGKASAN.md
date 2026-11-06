# Ringkasan hasil pilot 10 data

Profil: **experimenting**. Diperbarui: 2026-11-06T08:06:51+00:00.

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
| pplx | baseline | COMPLETE | 10 | 4 |  |  |  |
| pplx | llm | COMPLETE | 10 | 4 | 0 | 1 | 3 |
| pplx | gan | COMPLETE | 10 | 4 | 0 | 4 | 0 |

**BELUM DIJALANKAN bukan hasil nol atau gagal.** CSV kosong/header-only berarti belum ada baris tersimpan untuk tahap tersebut. Status RUNNING/INCOMPLETE berarti ringkasan bisa belum memuat semua progres. Kolom Flip/Invalid/Tetap SPAM pada baseline tidak digunakan untuk menilai serangan.

INVALID adalah hasil validasi, bukan error program. Hanya VALID boleh diklasifikasikan. Tidak ada Step 2 jika Step 1 ditolak atau sudah berhasil mengubah label. Bila classifier dan evaluator sama-sama PPLX, hasil ini belum mengukur ketahanan Clef/d3.

## Cara melanjutkan

Gunakan menu utama **1 → 4 → 1** untuk melanjutkan kedua cabang; tahap OK dilewati. Gunakan menu 3 / `--retry-errors` hanya jika ingin mencoba ulang tahap ERROR. CLI memeriksa bobot dan runtime GAN sebelum memulai pekerjaan yang mencakup GAN.

Laporan ini adalah salinan baca dari hasil tersimpan, bukan checkpoint. Jangan mengedit CSV laporan untuk mengubah status run. Menu **4 → Perbarui ringkasan** membangunnya kembali tanpa API. Evaluasi kemiripan lengkap tersedia melalui menu 4 → Evaluasi/arsip teknis.

## Arsip teknis (hanya bila diperlukan)

Event log, respons mentah, CSV individual dan cache tetap berada di folder run asal agar resume dan audit tetap cocok. Lokasi hasil asli:
- pplx / baseline: [folder run](../validated_baseline_pilot10_v1/pplx).
- pplx / llm: [folder run](../validated_llm_pilot10_v1/pplx).
- pplx / gan: [folder run](../validated_gan_pilot10_v1/pplx).
