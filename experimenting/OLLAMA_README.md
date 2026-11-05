# Arsip konfigurasi Ollama

Profil aktif tidak memakai Ollama. Gunakan [CLI experimenting](README.md).

Helper Python dan adaptive_ollama_config.json dipertahankan untuk inspeksi eksperimen lama. Launcher khusus Ollama telah dihapus. Pemeriksaan offline dari root:

~~~powershell
python experimenting/adaptive_pipeline.py --config experimenting/adaptive_ollama_config.json --model all --dry-run --pilot
~~~

Layanan/model/revision arsip harus diverifikasi sebelum inferensi nyata. Hasilnya tidak dicampur dengan pipeline Open WebUI.
