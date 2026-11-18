"""Keep model port forwards open in this terminal until SSH exits or Ctrl+C."""
import sys

from check_model_tunnel import MODELS, tunnel


def main():
    try:
        with tunnel() as ssh:
            ports = ", ".join(str(port) for port in MODELS)
            if ssh is None:
                print(f"Semua port lokal ({ports}) sudah menerima koneksi. Gunakan sesi yang sudah aktif.")
                return 0
            print(f"Port lokal siap: {ports}. Biarkan terminal ini terbuka. Tekan Ctrl+C untuk berhenti.", flush=True)
            return ssh.wait()
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Gagal membuka tunnel: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
