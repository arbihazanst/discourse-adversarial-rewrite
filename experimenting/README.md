# Experimenting — pilot PPLX, Gemma, dan GAN

Uji pipeline pada **10 original** (5 spam + 5 ham), dari dataset yang sama dengan main. Hasil selalu disimpan di folder experimenting.

## Buka CLI

Dari root proyek:

~~~powershell
.\experimenting\run_cli.bat
~~~

Git Bash / Linux:

~~~bash
bash experimenting/run_cli.sh
~~~

Dari dalam folder ini cukup run_cli.bat atau bash run_cli.sh. Gunakan Python 3.10+. Isi **OWUI_API_KEY** di **.env root proyek**; file dimuat otomatis.

Hanya kedua launcher CLI ini diperlukan. Launcher lama per model/metode sudah digantikan oleh menu.

## Cara paling mudah

1. Buka CLI, pilih **1 — Pilot 10 data**.
2. Pilih **4 — Gemma + GAN**.
3. Periksa folder hasil, pilih **1 — Jalankan / lanjutkan**.

Urutannya **baseline → Gemma → GAN**. Kedua cabang menggunakan ulang baseline. Jika hasil sudah selesai, pilihan yang sama tidak mengulang inferensi.

Untuk baseline saja: **1 → 1 → 1**. Untuk satu cabang, pilih Gemma atau GAN pada langkah 2.

## Menu

~~~text
1. Pilot 10 data
2. Benchmark seluruh 3.000 data
3. Resume / ulangi tahap yang error
4. Hasil CSV dan evaluasi
5. Pengaturan model, endpoint, output, dan GAN
6. Panduan singkat
0. Keluar
~~~

Pilih **Periksa konfigurasi offline** sebelum eksekusi untuk pemeriksaan tanpa API. **Cek layanan** pada pilot memeriksa katalog model/tool; inferensi hanya dibuktikan oleh run nyata. Menu 2 menjalankan seluruh dataset dan memerlukan revision sebenarnya.

## Model dan alur

| Peran | Pengaturan |
| --- | --- |
| API | https://llms.si-lab.org/api/chat/completions |
| Pemanggil PPLX Tool | google/gemma-3-27b-it |
| Tool classifier | server:pplxdeciderflash |
| Attacker rewrite | google/gemma-3-27b-it melalui Open WebUI |
| GAN | Model SST lokal di CPU |
| Defender aktif | PPLX; Clef/d3 belum dijalankan |

Gemma memanggil PPLX untuk klasifikasi. Request terpisah membuat rewrite tanpa tool. Label berasal dari hasil PPLX Tool; parameter SPAM/NOT_SPAM diverifikasi dari metadata eksekusi.

Attacker profil ini memakai Open WebUI, sedangkan main memakai SGLang port 30007. Profil ini tidak membutuhkan Ollama port 11435.

~~~text
10 original -> baseline PPLX
  |
  +-- true SPAM + prediksi SPAM -> dua cabang independen
        |
        +-- Gemma: rewrite 1 -> klasifikasi
        |             jika masih SPAM -> rewrite 2 -> klasifikasi
        |
        +-- GAN SST: kandidat -> probe -> klasifikasi akhir
~~~

Prompt Gemma sama dengan source **LLMattLLM.py**; GAN mengikuti default SST **GANAttacker()** dari GANSSILAB.py. HAM dan false negative baseline tidak ditulis ulang. GAN tidak menyerang hasil Gemma.

## Persiapan GAN satu kali

Dari root proyek, pada environment Python yang digunakan CLI:

~~~powershell
python -m pip install -r scripts/requirements-gan.txt
~~~

Buka **5 Pengaturan → 6 Siapkan bobot GAN**. Enter mengunduh resource resmi; isi folder jika bobot sudah tersedia. EXPERIMENT_PYTHON dapat menunjuk executable tertentu; default launcher memakai environment terminal.

## Folder hasil

Path berikut relatif terhadap experimenting/:

| Hasil | Folder |
| --- | --- |
| Baseline pilot | outputs/pplx_gemma_fixed_pilot10/pplx/ |
| Gemma pilot | outputs/pplx_llm_pilot10/pplx/ |
| GAN pilot | outputs/pplx_gan_pilot10/pplx/ |
| Benchmark Gemma | outputs/paper3000_adaptive/<model>/ |
| Benchmark GAN | outputs/paper3000_gan/<model>/ |

**Gemma:** step1_rewrites.csv, step2_rewrites.csv; CSV individual di rewrites/step1/ dan rewrites/step2/.

**GAN:** gan_rewrites.csv; CSV individual di rewrites/gan/. candidate_text menyimpan kandidat termasuk yang ditolak. returned_text kosong berarti tidak ditemukan adversarial yang lolos, bukan generator tidak berjalan.

**Semua run:** sample_results.csv, metrics.csv, events.csv, checkpoint.json, serta raw response di responses/. CSV menyimpan waktu generation/klasifikasi.

## Resume dan pengaturan

- **Menu 3:** pilih ukuran run/metode yang sama; lanjutkan pekerjaan belum selesai dan ulangi ERROR.
- **Menu 4:** pilih run tersimpan untuk melihat lokasi CSV dan evaluasi. Evaluasi leksikal tanpa model embedding; SBERT membutuhkan embedding. Tidak ada inferensi ulang.
- **Menu 5:** pilih model, edit endpoint, dan ganti output. Disimpan ke **experimenting/adaptive_local.json**, terpisah dari main.

CLI memakai route/model/revision dari profil; .env menyediakan kredensial. Jangan mengubah model/dataset/prompt lalu memakai output lama. Ctrl+C menyimpan progres; tutup CSV di Excel bila file terkunci.

## Tanpa menu

~~~powershell
.\experimenting\run_cli.bat --show
.\experimenting\run_cli.bat --task all --dry-run
.\experimenting\run_cli.bat --task all
.\experimenting\run_cli.bat --task all --retry-errors
.\experimenting\run_cli.bat --task baseline
.\experimenting\run_cli.bat --task llm
.\experimenting\run_cli.bat --task gan
~~~

Bash memakai opsi sama setelah bash experimenting/run_cli.sh. Gunakan --mode full hanya untuk seluruh dataset.

## Hasil sebelumnya

Pilot live: 10 baseline valid, 4 spam eligible, 8 rewrite Gemma (0 label flip), dan 4 kandidat GAN (4 label flip). [Catatan validasi](VALIDATION.md). Hasil lama tetap disimpan.

Pilot ditandai **UNVERIFIED_WEIGHTS**. Ini bukan benchmark publikasi; label flip belum membuktikan makna tetap terjaga.
