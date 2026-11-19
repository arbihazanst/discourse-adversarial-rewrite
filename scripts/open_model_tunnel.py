"""Keep model port forwards open in this terminal until SSH exits or Ctrl+C."""
import sys

from urllib.parse import urlsplit
from check_model_tunnel import default_profile, routes, tunnel


def main():
    try:
        config = default_profile()
        with tunnel(config) as ssh:
            ports = ", ".join(str(urlsplit(r['url']).port) for r in routes(config).values())
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
