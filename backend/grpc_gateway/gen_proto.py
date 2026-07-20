"""
Regenerasi stub gRPC dari proto/transaction.proto (Phase 5.0).

KENAPA ADA SCRIPT INI (bukan sekadar panggil protoc manual): grpc_tools
menghasilkan `import transaction_pb2 as transaction__pb2` — import ABSOLUT
yang HANYA berhasil kalau direktori stub kebetulan ada di sys.path. Begitu
modul ini di-import sebagai paket (mis. `from grpc_gateway import server`),
import itu GAGAL. Script ini menambalnya jadi import RELATIF supaya paketnya
berdiri sendiri, dan menyimpan langkah tambal itu agar TIDAK hilang saat
proto diregenerasi lain waktu (kalau ditambal manual, orang berikutnya
regenerate lalu bingung kenapa rusak).

Cara pakai (butuh grpcio-tools — tersedia di image Docker backend):
    docker exec muleradar_consumer python /app/grpc_gateway/gen_proto.py
atau lokal kalau grpcio-tools terpasang:
    python gen_proto.py
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO_DIR = os.path.join(HERE, "proto")
PROTO_FILE = os.path.join(PROTO_DIR, "transaction.proto")
GRPC_STUB = os.path.join(HERE, "transaction_pb2_grpc.py")


def main():
    print(f"[1/2] protoc -> {HERE}")
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={HERE}",
        f"--grpc_python_out={HERE}",
        PROTO_FILE,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr)
        sys.exit(f"protoc gagal (rc={res.returncode})")

    print("[2/2] Tambal import absolut -> relatif di stub grpc")
    with open(GRPC_STUB, encoding="utf-8") as f:
        src = f.read()
    patched, n = re.subn(
        r"^import (\w+_pb2) as (\w+)$",
        r"from . import \1 as \2",
        src,
        flags=re.MULTILINE,
    )
    if n == 0 and "from . import" not in src:
        print("  PERINGATAN: pola import tak ketemu — cek manual "
              "(format keluaran grpc_tools mungkin berubah).")
    else:
        with open(GRPC_STUB, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"  {n} import ditambal jadi relatif.")
    print("Selesai.")


if __name__ == "__main__":
    main()
