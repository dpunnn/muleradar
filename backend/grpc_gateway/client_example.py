"""
Contoh client bank utk gRPC Ingestion Gateway (Phase 5.0).

Menunjukkan 3 pola pemakaian yg didukung gateway:
  1. unary   — kirim 1 transaksi, tunggu konfirmasi (transaksi bernilai tinggi)
  2. stream  — banjirkan banyak transaksi, terima 1 ringkasan (throughput tinggi)
  3. ack     — bidirectional, konfirmasi per-transaksi sambil stream jalan

Juga dipakai sbg alat UJI end-to-end: gRPC -> Kafka -> consumer -> alert.

Cara pakai (dari direktori backend/):
    python -m grpc_gateway.client_example --mode health
    python -m grpc_gateway.client_example --mode unary
    python -m grpc_gateway.client_example --mode stream --count 100
    python -m grpc_gateway.client_example --mode ack --count 10
    python -m grpc_gateway.client_example --mode invalid   # uji penolakan
"""

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta

import grpc

from . import transaction_pb2 as pb
from . import transaction_pb2_grpc as pb_grpc

# 50061 (bukan 50051) — lihat catatan konflik Docker Desktop di server.py
GRPC_TARGET = os.getenv("GRPC_TARGET", "localhost:50061")
CHANNELS = ["mobile", "atm", "internet", "teller", "qris"]


def _make_tx(i: int = 0, account: str | None = None) -> pb.TransactionEvent:
    ts = datetime.now() - timedelta(minutes=random.randint(0, 120))
    return pb.TransactionEvent(
        tx_id=f"GRPC-{uuid.uuid4().hex[:12].upper()}",
        from_account=account or f"GRPC-SRC-{i % 5:03d}",
        to_account=f"GRPC-DST-{i % 3:03d}",
        amount=round(random.uniform(50_000, 25_000_000), 2),
        currency="IDR",
        channel=random.choice(CHANNELS),
        payment_format="Transfer",
        tx_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
        device_id=f"DEV-GRPC-{i % 7:03d}",
        institution_id="BANK_A",
    )


def run_health(stub):
    resp = stub.HealthCheck(pb.HealthRequest())
    print(f"serving={resp.serving} kafka_ok={resp.kafka_ok} "
          f"version={resp.version} topic={resp.kafka_topic}")


def _explain(e: grpc.RpcError) -> str:
    """Terjemahkan RpcError jadi arahan yg bisa ditindaklanjuti.

    Ini POLA YANG SEHARUSNYA DISALIN integrator bank: bedakan kesalahan
    DATA (perbaiki payload, retry percuma) dari kesalahan LAYANAN (data
    sudah benar, cukup coba lagi nanti).
    """
    code = e.code()
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        saran = "-> DATA cacat: perbaiki payload, retry TIDAK akan menolong."
    elif code == grpc.StatusCode.UNAVAILABLE:
        saran = "-> LAYANAN sedang tak siap (mis. Kafka down): RETRY dgn backoff."
    elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
        saran = "-> timeout: naikkan deadline atau kecilkan batch."
    else:
        saran = "-> cek log server."
    return f"[{code.name}] {e.details()}\n   {saran}"


def run_unary(stub):
    tx = _make_tx()
    try:
        ack = stub.SendTransaction(tx)
        print(f"unary -> tx_id={ack.tx_id} accepted={ack.accepted} error={ack.error!r}")
    except grpc.RpcError as e:
        # Contoh penanganan yg BENAR — jangan biarkan RpcError jadi traceback
        # mentah di sistem bank.
        print(f"unary GAGAL {_explain(e)}")


def run_stream(stub, count: int):
    def gen():
        for i in range(count):
            yield _make_tx(i)
    try:
        summary = stub.StreamTransactions(gen())
    except grpc.RpcError as e:
        print(f"stream GAGAL {_explain(e)}")
        return
    print(f"stream -> received={summary.received} accepted={summary.accepted} "
          f"rejected={summary.rejected}")
    for e in summary.errors:
        print(f"   error sampel: {e}")


def run_ack(stub, count: int):
    def gen():
        for i in range(count):
            yield _make_tx(i)
    ok = bad = 0
    try:
        for ack in stub.StreamTransactionsWithAck(gen()):
            ok += ack.accepted
            bad += (not ack.accepted)
            if not ack.accepted:
                print(f"   ditolak {ack.tx_id}: {ack.error}")
    except grpc.RpcError as e:
        print(f"ack GAGAL {_explain(e)}")
        return
    print(f"ack -> diterima={ok} ditolak={bad}")


def run_invalid(stub):
    """Uji bahwa gerbang MENOLAK data cacat (bukan meneruskannya ke pipeline)."""
    kasus = [
        ("tx_id kosong", pb.TransactionEvent(
            tx_id="", from_account="A", to_account="B", amount=1000,
            tx_timestamp="2026-07-20 10:00:00")),
        ("amount negatif", pb.TransactionEvent(
            tx_id="BAD-1", from_account="A", to_account="B", amount=-5,
            tx_timestamp="2026-07-20 10:00:00")),
        ("timestamp ngawur", pb.TransactionEvent(
            tx_id="BAD-2", from_account="A", to_account="B", amount=1000,
            tx_timestamp="kemarin sore")),
        ("from_account kosong", pb.TransactionEvent(
            tx_id="BAD-3", from_account="", to_account="B", amount=1000,
            tx_timestamp="2026-07-20 10:00:00")),
    ]
    for nama, tx in kasus:
        try:
            ack = stub.SendTransaction(tx)
            print(f"  {nama:22s} -> accepted={ack.accepted} error={ack.error!r}")
        except grpc.RpcError as e:
            print(f"  {nama:22s} -> DITOLAK [{e.code().name}] {e.details()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="health",
                    choices=["health", "unary", "stream", "ack", "invalid"])
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--target", default=GRPC_TARGET)
    args = ap.parse_args()

    with grpc.insecure_channel(args.target) as ch:
        stub = pb_grpc.IngestionServiceStub(ch)
        print(f"[client] -> {args.target} mode={args.mode}")
        if args.mode == "health":
            run_health(stub)
        elif args.mode == "unary":
            run_unary(stub)
        elif args.mode == "stream":
            run_stream(stub, args.count)
        elif args.mode == "ack":
            run_ack(stub, args.count)
        elif args.mode == "invalid":
            run_invalid(stub)


if __name__ == "__main__":
    main()
