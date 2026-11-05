# Konfigurasi controller

Panduan: [main](../README.md), [experimenting](../experimenting/README.md).

| Profil | Default | Perubahan menu |
| --- | --- | --- |
| Main | decision_models/adaptive_main.json | decision_models/adaptive_local.json |
| Experimenting | experimenting/adaptive_config.json | experimenting/adaptive_local.json |

CLI memilih profil lokal bila ada. --config PATH memilih profil eksplisit. Path relatif dihitung dari root proyek. Dataset bersama: dataset/paper_baseline3000.csv.

## Field

- **defenders**: endpoint, transport, model/tool ID, revision, nama variabel API key.
- **default_model**: pilihan menu; **active_models**: model untuk pilihan semua.
- **attacker**: main menggunakan SGLang port 30007; experimenting memakai Open WebUI. Keduanya transport openai.
- **output_dir / gan_output_dir**: output benchmark penuh.
- **tool_pilot_output_dir / tool_llm_pilot_output_dir / tool_gan_pilot_output_dir**: output pilot terpisah. Dua field terakhir opsional.
- **gan.resource_dir / gan.seed**: resource SST dan seed replay.
- **reuse_historical_step1**: benchmark penuh saja; historis ditandai unverified. Pilot membuat rewrite baru.

CLI memakai route/model/revision yang ditampilkan; .env menyediakan kredensial. Controller Python langsung tetap mendukung override environment.

## Identitas dan resume

Baseline diidentifikasi oleh dataset, endpoint, model, revision, question/criteria, dan protokol tool. Attack juga mencakup prompt/attacker atau resource GAN. Perubahan ini memerlukan output baru.

Pilot khusus PPLX, 10 original, dengan revision kosong ditandai UNVERIFIED_WEIGHTS. Hash katalog stabil bukan hash bobot. Timestamp katalog sementara tidak menjadi identitas baru. Benchmark penuh memerlukan revision sebenarnya.

events.csv dan raw artifact adalah catatan utama. Checkpoint dan tabel dapat dibangun ulang. Baseline hanya dipakai ulang bila identitas cocok.

## Adapter lanjutan

Transport tool: Open WebUI PPLX Tool dengan parameter SPAM/NOT_SPAM tetap. openwebui: Pipe. direct: endpoint native; Clef/d3 memakai llama.cpp /v1/systemone, PPLX native memerlukan template API yang diverifikasi.

Backend Ollama historis dipertahankan untuk inspeksi lewat modul Python/profil arsip, tidak dipakai profil aktif. [Arsip](../experimenting/OLLAMA_README.md).
