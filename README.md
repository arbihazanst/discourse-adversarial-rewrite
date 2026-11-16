# Eksperimen melalui tunnel SSH

Menu **1 → 4** menjalankan pilot 5 SPAM dan 5 NOT_SPAM. **2 → 4** menjalankan seluruh dataset. Kedua profil memakai PPLX 11438, Clef 11439, d3 11440 dan Gemma 11441. OpenSSH meminta password di terminal jika tunnel perlu dibuka.

```bash
bash ./run_cli.sh --mode pilot --task all --dry-run
bash ./run_cli.sh --mode pilot --task all --run-id uji-01
bash ./run_cli.sh --mode full --task all --run-id benchmark-01
```

Gunakan `baseline`, `llm`, atau `gan` untuk satu pekerjaan. Gunakan `--model clef` atau `--model d3` untuk satu defender. Menu 5 mengatur port semua pekerjaan. Gunakan ID yang sama untuk resume, tambahkan `--retry-errors` untuk mengulang request gagal, atau gunakan ID baru untuk mulai lagi. ID model dari layanan tidak membuktikan revisi bobot; gunakan ID run baru setelah mengganti bobot.

1. Clef dan d3 mengklasifikasikan original. Tiap defender melanjutkan email berlabel SPAM yang ia deteksi sebagai SPAM.
2. Gemma menghasilkan peringatan untuk penerima. GAN membaca kandidat tetap dari `dataset/gan_replay.csv`. Kedua cabang berbagi hasil baseline. File arsip mencatat ID, tahap, hash original, teks, file sumber dan nomor rekaman. Rekaman pilot yang ID dan teksnya cocok diprioritaskan; tambahan dari arsip GPT-OSS paper dicocokkan secara unik berdasarkan teks asli setelah normalisasi spasi.
3. PPLX menilai lima kriteria preservasi asli. Hanya hasil VALID diklasifikasikan. Step 2 dijangkau jika hasil valid masih SPAM. `NO_ARCHIVE` berarti tidak tersedia kandidat pada tahap itu. Rubrik preservasi dapat menolak peringatan yang baik karena peringatan mengubah tujuan email. Label flip pada cabang Gemma menggambarkan klasifikasi atas peringatan.
4. Folder run yang dicetak CLI memuat `baseline.csv`, CSV per cabang/tahap, CSV per kandidat, `checkpoint.sqlite`, dan `ringkasan.json`. Menu utama 4 menampilkan ringkasan yang sudah selesai.

## Opsi 5: pemeriksaan empat layanan melalui SSH

Dari root proyek jalankan `bash ./run_cli.sh --task services`, atau pilih **1 (Pilot) lalu 5 (Pilot layanan SSH)**. Di Windows gunakan launcher `.bat` dengan argumen yang sama. Pilot memilih lima SPAM dan lima NOT_SPAM pertama dari dataset 3.000 email.

| Layanan | ID model | Port lokal | API dan pengujian |
|---|---|---:|---|
| PPLX | `perplexity-ai/pplx-decider-flash-v1.2` | 11438 | `/v1/systemone`, lima kriteria validitas untuk setiap pasangan pilot yang tersedia di bank rewrite |
| Clef | `clef-flash-q4` | 11439 | `/v1/systemone`, klasifikasi 10 original |
| d3 | `d3-flash-q4` | 11440 | `/v1/systemone`, klasifikasi 10 original |
| Gemma | `google/gemma-3-27b-it` | 11441 | `/v1/chat/completions`, satu pertanyaan RAG untuk memeriksa respons chat |

Untuk seluruh dataset gunakan menu **2 → 5**, atau dari root proyek:

```bash
bash ./run_cli.sh --mode full --task services --run-id uji-01
```

Ganti ID untuk run baru. Gunakan ID yang sama untuk resume. Hasil mode full ada di `outputs/layanan_full/<id-run>/`. Clef dan d3 mengklasifikasikan semua email asli, PPLX mengevaluasi pasangan yang tersedia di bank rewrite, dan Gemma menerima satu pertanyaan uji. Opsi `--dry-run` memeriksa rencana secara offline.

Launcher membuka tunnel untuk port loopback yang belum aktif. OpenSSH meminta password di terminal jika diperlukan dan launcher menutup tunnel miliknya ketika selesai. Tunnel atau layanan lokal yang sudah berjalan bisa langsung digunakan. Untuk membuka tunnel secara manual:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:11438:127.0.0.1:11438 -L 127.0.0.1:11439:127.0.0.1:11439 -L 127.0.0.1:11440:127.0.0.1:11440 -L 127.0.0.1:11441:127.0.0.1:11441 -p 28659 rinaldi@37.60.243.124
```

Untuk mengganti port lokal pada satu run, sesuaikan port kiri pada `-L` lalu tambahkan misalnya `--port pplx=12438 --port gemma=12441` ke perintah pilot. Port kanan adalah port layanan di server. Untuk menyimpan perubahan, pilih menu 5 lalu **Ubah port tunnel untuk semua pekerjaan**, atau edit `service_pilot` pada JSON profil aktif. `--dry-run` menampilkan endpoint yang akan dipakai.

Hasil dan checkpoint tersedia di `outputs/pilot_layanan10/<id-run>/`. Buka `ringkasan.json` untuk jumlah klasifikasi benar dan verdict per pasangan, serta `hasil.csv` untuk label, waktu dan error tiap request. Perintah yang sama melanjutkan hasil tersimpan. Tambahkan `--retry-errors` untuk mengulang request yang gagal. Perubahan endpoint, model, input, atau pertanyaan menghasilkan ID run baru.

Jika server mengembalikan HTTP 500 dengan pesan `increase the physical batch size`, periksa jumlah token yang disebutkan. Pada pilot ini batas Clef/d3 adalah 512 token dan input terpanjang memerlukan 1.890 token pada Clef. Sesudah kapasitas server disesuaikan, jalankan kembali perintah pilot dengan `--retry-errors` untuk memproses request yang gagal.
