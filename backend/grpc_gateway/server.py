"""
MuleRadar — gRPC Ingestion Gateway (Phase 5.0).

PERAN: pintu masuk transaksi dari CORE BANKING bank ke pipeline deteksi.
    Bank -> gRPC (protobuf biner, streaming) -> Kafka 'transactions.raw'
         -> consumer -> RealtimeScorer (TGN) -> alert

Gateway ini BERDAMPINGAN dgn streaming/producer.py (replay/simulate/demo),
BUKAN menggantikannya: producer = jalur data internal utk demo & replay
historis; gRPC = jalur "bank sungguhan". Keduanya menulis ke TOPIC YANG SAMA
sehingga hilir (consumer, scorer, alert) tak perlu tahu bedanya.

KEPUTUSAN PENTING:
- Nama paket `grpc_gateway`, BUKAN `grpc` (spec awal menulis backend/grpc/).
  Alasan: direktori bernama `grpc` akan MENIMPA (shadow) paket `grpc` milik
  grpcio saat import -> server gagal jalan dgn error membingungkan.
- Kafka key = from_account, SAMA dgn producer (lihat producer._send_tx):
  menjaga account-affinity partisi supaya memory TGN per-akun tidak
  di-update dua consumer sekaligus (lost update). Kalau gateway ini
  mengirim tanpa key, jaminan itu bocor lewat pintu belakang.
- VALIDASI di gerbang: transaksi cacat DITOLAK di sini, tidak dibiarkan
  mengotori Kafka/graph/fitur. Lebih murah menolak di tepi daripada
  membersihkan di hilir.

Cara pakai:
    python -m grpc_gateway.server            (dari direktori backend/)
    GRPC_PORT=50051 python -m grpc_gateway.server
"""

import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from datetime import datetime

import grpc
from kafka import KafkaProducer
from kafka.errors import KafkaError

from . import transaction_pb2 as pb
from . import transaction_pb2_grpc as pb_grpc

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpc_gateway")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TX = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions.raw")
# Port default 50061, BUKAN 50051 (port konvensi gRPC) — fix 20-Jul:
# di Windows, port 50051 SUDAH DIPAKAI Docker Desktop sendiri
# (com.docker.backend.exe listen di 0.0.0.0:50051 & [::]:50051), sehingga
# server gagal bind dan client malah menabrak layanan gRPC milik Docker
# (error membingungkan: "unknown service muleradar.ingestion.v1...").
GRPC_PORT = int(os.getenv("GRPC_PORT", "50061"))
# Bind 0.0.0.0 (IPv4-any) lebih portabel drpd "[::]" (IPv6-any) di Windows.
GRPC_BIND = os.getenv("GRPC_BIND", "0.0.0.0")
MAX_WORKERS = int(os.getenv("GRPC_MAX_WORKERS", "10"))
# Batas ukuran pesan: cegah client mengirim payload raksasa yg bisa
# menghabiskan memori server (default gRPC 4MB; dinaikkan utk batch besar
# TAPI tetap DIBATASI — jangan unlimited).
MAX_MSG_MB = int(os.getenv("GRPC_MAX_MSG_MB", "16"))
# Sampel error yg dikembalikan di IngestSummary — jangan kirim semua error
# kalau jutaan baris ditolak (respons bisa meledak).
MAX_ERROR_SAMPLES = 20
# Jeda minimum antar percobaan sambung ulang ke Kafka saat broker mati —
# lihat penjelasan di IngestionServicer._get_producer.
RECONNECT_COOLDOWN_S = float(os.getenv("GRPC_KAFKA_RECONNECT_COOLDOWN_S", "5"))

_VALID_CHANNELS = {"mobile", "atm", "internet", "teller", "qris"}
_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f")


def _validate(tx: "pb.TransactionEvent") -> str | None:
    """Return None kalau valid, atau STRING alasan penolakan.

    Divalidasi di gerbang supaya data cacat tak pernah masuk pipeline:
    tx_id kosong -> idempotensi consumer (dedup by tx_id) tak bisa bekerja;
    akun kosong -> node hantu di graph; amount negatif/NaN -> merusak fitur
    agregat; timestamp tak terparse -> fitur temporal (jam, dormancy) salah.
    """
    if not tx.tx_id.strip():
        return "tx_id kosong (wajib — dipakai dedup idempotensi)"
    if not tx.from_account.strip():
        return "from_account kosong"
    if not tx.to_account.strip():
        return "to_account kosong"
    if tx.amount < 0:
        return f"amount negatif ({tx.amount})"
    if tx.amount != tx.amount or tx.amount in (float("inf"), float("-inf")):
        return "amount bukan angka hingga (NaN/Inf)"
    if not tx.tx_timestamp.strip():
        return "tx_timestamp kosong"
    ts_ok = any(_try_parse(tx.tx_timestamp, f) for f in _TS_FORMATS)
    if not ts_ok:
        return (f"tx_timestamp '{tx.tx_timestamp}' tak dikenal "
                f"(harap ISO-8601, mis. '2026-07-20 13:45:00')")
    if tx.channel and tx.channel.lower() not in _VALID_CHANNELS:
        # Tidak ditolak — channel asing tetap diterima (bank bisa punya
        # channel baru), tapi dicatat supaya ketahuan kalau ada yg tak
        # terpetakan ke fitur channel_entropy.
        logger.warning("channel tak dikenal: %r (tx_id=%s)", tx.channel, tx.tx_id)
    return None


def _try_parse(value: str, fmt: str) -> bool:
    try:
        datetime.strptime(value.strip()[:26], fmt)
        return True
    except ValueError:
        return False


def _to_event(tx: "pb.TransactionEvent") -> dict:
    """Protobuf -> dict, format SAMA dgn producer._row_to_event supaya hilir
    (consumer/scorer) tak perlu membedakan asal data."""
    return {
        "tx_id": tx.tx_id,
        "from_account": tx.from_account,
        "to_account": tx.to_account,
        "amount": float(tx.amount),
        "currency": tx.currency or "IDR",
        "channel": tx.channel or "internet",
        "payment_format": tx.payment_format or "Transfer",
        "tx_timestamp": tx.tx_timestamp,
        "device_id": tx.device_id or "",
        "institution_id": tx.institution_id or "",
        "is_laundering": 0,   # gateway TIDAK pernah mengarang label
        "typology": None,     # tipologi ditentukan deteksi, bukan input bank
    }


class IngestionServicer(pb_grpc.IngestionServiceServicer):
    """Servicer dgn koneksi Kafka TAHAN-GAGAL.

    Producer dibuat MALAS (lazy) & disambung ulang otomatis. Alasannya:
    kalau producer dibuat paksa saat startup dan Kafka belum siap,
    KafkaProducer() melempar NoBrokersAvailable -> proses mati -> dgn
    `restart: unless-stopped` jadi CRASH-LOOP berisik yg menyamarkan
    masalah aslinya. Lebih baik gateway TETAP HIDUP, HealthCheck jujur
    melaporkan kafka_ok=false, tulis ditolak dgn UNAVAILABLE (client tahu
    harus retry), lalu PULIH SENDIRI begitu Kafka kembali.
    """

    def __init__(self, producer: KafkaProducer | None):
        self.producer = producer
        self._last_connect_try = 0.0
        self._lock = threading.Lock()

    def _get_producer(self) -> KafkaProducer | None:
        """Producer siap-pakai, atau None kalau Kafka sedang tak terjangkau.

        COOLDOWN (fix 20-Jul): tanpa jeda, SETIAP request saat Kafka mati akan
        mencoba membuat KafkaProducer baru — dan pembuatannya BLOCKING beberapa
        detik. Di trafik tinggi itu menghabiskan thread pool gRPC sampai
        gateway ikut tak responsif (kegagalan Kafka menular jadi kegagalan
        gateway). Dgn cooldown, percobaan sambung ulang dibatasi 1x per
        RECONNECT_COOLDOWN_S; request di antaranya langsung ditolak CEPAT
        dgn UNAVAILABLE — client tahu harus retry, thread tidak tertahan.
        Lock: cegah beberapa thread membuat producer bersamaan.
        """
        if self.producer is not None:
            return self.producer
        now = time.time()
        if now - self._last_connect_try < RECONNECT_COOLDOWN_S:
            return None          # masih dalam jeda — tolak cepat, jangan blokir
        with self._lock:
            if self.producer is not None:      # thread lain sudah berhasil
                return self.producer
            if time.time() - self._last_connect_try < RECONNECT_COOLDOWN_S:
                return None
            self._last_connect_try = time.time()
            try:
                self.producer = _make_producer()
                logger.info("Koneksi Kafka pulih — producer tersambung kembali.")
            except Exception as e:
                logger.warning("Kafka masih tak terjangkau: %s "
                               "(coba lagi paling cepat %ds lagi)", e, RECONNECT_COOLDOWN_S)
                return None
        return self.producer

    # ── util internal ────────────────────────────────────────────
    def _publish(self, tx: "pb.TransactionEvent", confirm: bool = False) -> str | None:
        """Validasi + publish. Return None kalau sukses, atau alasan gagal.

        confirm=True (fix QC 20-Jul): TUNGGU konfirmasi broker sebelum
        menyatakan berhasil. Penting krn producer.send() itu ASINKRON — tanpa
        menunggu, gateway bisa menjawab accepted=true padahal pesan GAGAL
        terkirim sesudahnya, dan bank mengira transaksinya sudah masuk padahal
        HILANG. Dipakai di unary (transaksi bernilai tinggi, konfirmasi = inti
        kontraknya). Di streaming TIDAK dipakai per-pesan (akan meruntuhkan
        throughput); di sana jaminannya lewat flush() di akhir stream + acks.
        """
        err = _validate(tx)
        if err:
            return err
        producer = self._get_producer()
        if producer is None:
            return "kafka error: broker tak terjangkau"
        try:
            # key=from_account -> account-affinity partisi (lihat catatan modul)
            future = producer.send(TOPIC_TX, key=tx.from_account, value=_to_event(tx))
            if confirm:
                future.get(timeout=10)   # melempar KafkaError kalau gagal
            return None
        except KafkaError as e:
            logger.error("Kafka menolak tx_id=%s: %s", tx.tx_id, e)
            # Buang producer yg rusak supaya panggilan berikutnya menyambung ulang
            self.producer = None
            return f"kafka error: {e}"

    # ── RPC: unary ───────────────────────────────────────────────
    def SendTransaction(self, request, context):
        err = self._publish(request)
        if err:
            # Data cacat = kesalahan CLIENT -> INVALID_ARGUMENT. Kafka mati =
            # kesalahan SERVER -> UNAVAILABLE. Dibedakan supaya client tahu
            # apakah harus memperbaiki data atau cukup mencoba lagi nanti.
            if err.startswith("kafka error"):
                context.set_code(grpc.StatusCode.UNAVAILABLE)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(err)
            return pb.IngestAck(tx_id=request.tx_id, accepted=False, error=err)
        if self.producer is not None:
            self.producer.flush(timeout=5)
        return pb.IngestAck(tx_id=request.tx_id, accepted=True)

    # ── RPC: client-streaming ────────────────────────────────────
    def StreamTransactions(self, request_iterator, context):
        received = accepted = rejected = 0
        errors: list[str] = []
        t0 = time.time()
        for tx in request_iterator:
            received += 1
            err = self._publish(tx)
            if err:
                rejected += 1
                if len(errors) < MAX_ERROR_SAMPLES:
                    errors.append(f"{tx.tx_id or '(tanpa tx_id)'}: {err}")
            else:
                accepted += 1
        # flush SEKALI di akhir stream (jauh lebih efisien drpd per-pesan)
        try:
            if self.producer is not None:
                self.producer.flush(timeout=30)
        except Exception as e:
            logger.error("flush Kafka gagal di akhir stream: %s", e)
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(f"flush gagal: {e}")
        dur = time.time() - t0
        logger.info("StreamTransactions: %d diterima, %d masuk, %d ditolak (%.1fs, %.0f tx/s)",
                    received, accepted, rejected, dur, received / dur if dur > 0 else 0)
        return pb.IngestSummary(received=received, accepted=accepted,
                                rejected=rejected, errors=errors)

    # ── RPC: bidirectional ───────────────────────────────────────
    def StreamTransactionsWithAck(self, request_iterator, context):
        for tx in request_iterator:
            err = self._publish(tx)
            yield pb.IngestAck(tx_id=tx.tx_id, accepted=err is None, error=err or "")
        try:
            if self.producer is not None:
                self.producer.flush(timeout=30)
        except Exception as e:
            logger.error("flush Kafka gagal: %s", e)

    # ── RPC: health ──────────────────────────────────────────────
    def HealthCheck(self, request, context):
        # producer None = belum/putus tersambung -> jujur laporkan tidak sehat
        try:
            kafka_ok = bool(self.producer and self.producer.bootstrap_connected())
        except Exception:
            kafka_ok = False
        return pb.HealthResponse(serving=True, kafka_ok=kafka_ok,
                                 version="0.1.0", kafka_topic=TOPIC_TX)


def _make_producer() -> KafkaProducer:
    import json
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        acks="all",     # jangan anggap terkirim sebelum broker mengonfirmasi
        retries=3,
        linger_ms=10,   # kelompokkan kiriman -> throughput naik utk streaming
        # Timeout PENDEK & eksplisit (fix 20-Jul): default kafka-python bisa
        # menggantung lama saat broker mati -> RPC ikut menggantung, client
        # tak dapat jawaban. Lebih baik GAGAL CEPAT dgn UNAVAILABLE.
        api_version_auto_timeout_ms=3000,
        request_timeout_ms=10000,
        max_block_ms=5000,   # send() tak menunggu buffer/metadata lebih dari ini
    )


def serve():
    # Jangan MATI kalau Kafka belum siap saat startup — servicer akan
    # menyambung sendiri saat transaksi pertama datang (lihat _get_producer).
    try:
        producer = _make_producer()
    except Exception as e:
        logger.warning("Kafka belum terjangkau saat startup (%s) — gateway tetap "
                       "jalan, HealthCheck akan lapor kafka_ok=false & tulis "
                       "ditolak UNAVAILABLE sampai Kafka pulih.", e)
        producer = None
    opts = [
        ("grpc.max_receive_message_length", MAX_MSG_MB * 1024 * 1024),
        ("grpc.max_send_message_length", MAX_MSG_MB * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
                         options=opts)
    servicer = IngestionServicer(producer)
    pb_grpc.add_IngestionServiceServicer_to_server(servicer, server)
    bind = f"{GRPC_BIND}:{GRPC_PORT}"
    # add_insecure_port() balik 0 kalau bind GAGAL (port dipakai proses lain).
    # Tanpa cek ini server "seolah jalan" padahal tak mendengarkan apa pun —
    # gejalanya membingungkan di client (connection refused / unknown service).
    if server.add_insecure_port(bind) == 0:
        raise RuntimeError(
            f"Gagal bind {bind} — port kemungkinan sudah dipakai proses lain. "
            f"Di Windows, port 50051 dipakai Docker Desktop. Set GRPC_PORT "
            f"ke port lain, atau hentikan proses yg memakainya.")
    server.start()
    logger.info("gRPC Ingestion Gateway LISTEN di %s -> Kafka %s topic '%s'",
                bind, KAFKA_BOOTSTRAP, TOPIC_TX)
    logger.warning("Koneksi TANPA TLS (insecure). Untuk produksi WAJIB mTLS — "
                   "lihat PIPELINE.txt Phase 11.2a.")

    # Shutdown rapi: hentikan terima RPC baru, biarkan yg berjalan selesai,
    # lalu FLUSH Kafka supaya transaksi yg sudah diterima tidak hilang.
    def _shutdown(signum, _frame):
        logger.info("Sinyal %s diterima — shutdown rapi...", signum)
        server.stop(grace=10).wait()
        try:
            if servicer.producer is not None:
                servicer.producer.flush(timeout=15)
                servicer.producer.close(timeout=10)
            logger.info("Kafka producer di-flush & ditutup.")
        except Exception as e:
            logger.error("Gagal flush/close Kafka saat shutdown: %s", e)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
