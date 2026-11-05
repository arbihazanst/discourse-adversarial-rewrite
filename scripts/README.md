# Modul pipeline

Buka run_cli.bat atau bash run_cli.sh di root proyek. CLI terpisah tersedia di experimenting/run_cli.bat dan .sh. Folder scripts tidak mempunyai launcher per model.

| Modul | Fungsi |
| --- | --- |
| experiment_cli.py | Menu, profil, rencana eksekusi, hasil, evaluasi |
| pilot_openwebui.py | Pilot 10 original, katalog tool, baseline reuse |
| adaptive_pipeline.py | Baseline, dua langkah rewrite, checkpoint |
| gan_pipeline.py | SST GAN, probe, fallback, post-classification |
| env_config.py | Membaca .env secara literal |
| evaluate_paper.py | Evaluasi artifact tersimpan |
| run_decision_benchmark.py | Adapter klasifikasi statis/native lanjutan |
| reparse_saved_tools.py | Memulihkan parsing response lama tanpa HTTP |

Automation: python scripts/experiment_cli.py --profile main --task all --dry-run. Pilih profil experimenting untuk output terpisah. Modul Python tetap menyediakan --help untuk opsi khusus. CLI tidak mengelola Docker.

Panduan: [main](../README.md), [experimenting](../experimenting/README.md), [evaluasi](../decision_models/EVALUATION.md).
