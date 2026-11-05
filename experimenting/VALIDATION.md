# Validasi pilot — 5 November 2026

> Catatan historis sebelum penyederhanaan launcher. Perintah saat ini: .\experimenting\run_cli.bat --task all. Launcher lama yang disebut di bawah telah diganti CLI; hasil inferensi tetap sama.

Pilot live si-lab selesai memakai `google/gemma-3-27b-it` sebagai pemanggil tool `server:pplxdeciderflash`. Model yang sama digunakan sebagai attacker Gemma dalam request terpisah tanpa `tool_ids`. GPT-OSS, Clef, dan d3 tidak dipanggil.

| Tahap | Hasil | Output |
| --- | --- | --- |
| Baseline | 10/10 klasifikasi valid; 5 spam + 5 ham; 4 true-positive spam eligible | `outputs/pplx_gemma_fixed_pilot10/pplx/` |
| Gemma Step 1 | 4 rewrite, 4 klasifikasi valid, 0 label flip | `outputs/pplx_llm_pilot10/pplx/step1_rewrites.csv` |
| Gemma Step 2 | 4 rewrite, 4 klasifikasi valid, 0 label flip tambahan | `outputs/pplx_llm_pilot10/pplx/step2_rewrites.csv` |
| GAN SST | 4 kandidat dikembalikan, 4 klasifikasi akhir valid, 4 label flip | `outputs/pplx_gan_pilot10/pplx/gan_rewrites.csv` |

Masing-masing cabang attack menggunakan ulang seluruh 10 baseline. Delapan CSV individual Gemma ada di `rewrites/step1/` dan `rewrites/step2/`; empat CSV individual GAN ada di `rewrites/gan/`, di dalam output masing-masing. CSV mencatat waktu generation/probe/classification. Latency klasifikasi mencakup jaringan, Open WebUI, orchestration, dan tool.

## Mengapa sebelumnya tidak ada rewrite

Run sebelumnya menyimpan `baseline_only=true`. Launcher `run_pilot.bat`, `run_pplx.bat`, dan `run_all.bat` adalah pilot baseline. Gunakan `run_pilot_all.bat` untuk baseline + Gemma + GAN, atau `run_pilot_llm.bat` dan `run_pilot_gan.bat` untuk cabang individual.

Error lain yang diperbaiki: `.env` belum dimuat otomatis, parser belum mengenali provenance external-tool-server, kategori tool dapat diubah orchestration, dan timestamp katalog `updated_at` berubah setiap request sehingga memblokir resume. Parameter tool kini diverifikasi terhadap state/instructions/criteria SPAM/NOT_SPAM yang tetap. Kompatibilitas checkpoint lama hanya diterima setelah hash lama berhasil direkonstruksi dengan perbedaan timestamp saja; bukti disimpan dalam `pilot_catalog_compatibility.json`.

Satu baseline timeout berhasil diulang dengan timeout 300 detik; sembilan hasil valid dipakai ulang. Catatan percobaan gagal tetap tersimpan dalam event log. Hasil GPT-OSS lama dan pilot Gemma sebelum kontrak tetap tidak dicampurkan ke hasil ini.

## Kesesuaian source dan pemeriksaan

- Kedua prompt Gemma identik dengan konstanta `ATTACKER_PROMPT_1` dan `ATTACKER_PROMPT_2_TEMPLATE` di `OpenAttack-Share/LLMattLLM.py` (SHA-256 `fafd3581c0cf7fb4b01243d91f6844bd453d58b5832ab36232871dd628d3ba01`). Step 2 hanya berjalan jika Step 1 masih SPAM.
- GAN mengikuti default SST `GANAttacker()` dan pola probe/fallback/post-classification source `GANSSILAB.py` (SHA-256 `214250e6ee02a57496aa6be14185332c2d077b8745ada5d619bda65eb2b3c317`). Bobot GAN lokal asli digunakan, bukan mock atau prompt Gemma.
- 94 automated tests lolos. HTTP pada regression tests menggunakan mock lokal; pilot di atas terpisah dan live.
- `run_pilot_all.bat --resume` selesai dengan semua pending=0. Jumlah event sebelum/sesudah tetap 11 baseline, 26 LLM, dan 22 GAN; tidak ada inferensi tambahan. Hash semua artifact pada kedua cabang diverifikasi.
- Evaluasi tersimpan di subfolder `evaluation/` masing-masing cabang. Evaluasi leksikal dijalankan tanpa mengulang inference; SBERT belum dihitung karena model embedding belum disiapkan.

Ini pilot 10 original, bukan hasil benchmark 3.000 data. Revision bobot layanan belum terpin (`UNVERIFIED_WEIGHTS`). Empat label flip GAN tidak membuktikan makna email tetap terjaga; SST source membatasi input ke 100 token. Main memakai attacker SGLang `http://127.0.0.1:30007/v1/chat/completions`; layanan GPU main belum diuji langsung dari PC ini.
