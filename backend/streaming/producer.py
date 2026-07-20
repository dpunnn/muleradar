"""
Kafka Producer — MuleRadar Transaction Stream

Empat mode:
  replay          : baca transaksi historis dari PostgreSQL → publish ke Kafka
                    (mensimulasikan transaksi yang sudah ada seolah baru masuk)
  simulate        : generate transaksi sintetis baru tanpa henti → publish ke Kafka
                    (presentasi real-time stream ke juri)
  demo-replay     : replay transaksi ASLI, tapi HANYA akun berlabel dyg_split==
                    "test" (akun yang TIDAK PERNAH dilihat model saat training) —
                    dgn interval ACAK, khusus utk video demo (PIPELINE.txt 5.0b).
                    Membuktikan model ENSEMBLE/TGN mendeteksi data genuinely
                    belum pernah dilihat, bukan menghafal training set.
                    CATATAN: transaksi acak/tunggal di mode ini TIDAK trigger
                    alert (real-time fast-path butuh POLA berulang, bukan
                    cuma label) — pakai demo-collector kalau mau alert pasti
                    muncul saat streaming.
  demo-collector  : replay SELURUH riwayat 1 akun kolektor ASLI (fan-in
                    tinggi dari transaksi is_laundering=1 asli) — urutan
                    WAJIB transaksi MASUK dulu (bangun sinyal fan_in/device
                    sharing), baru transaksi KELUAR (di titik itu kolektor
                    DIEVALUASI, alert genuinely trigger dari akumulasi
                    sinyal perilaku, BUKAN dari baca label is_laundering).
                    Fitur real-time (fan_in via Redis SET) TIDAK terikat
                    dyg_split (itu partisi train/test khusus model TGN/
                    DyGFormer/ensemble) — jadi valid dipakai independen
                    utk buktikan real-time fast-path scorer BEKERJA live.
                    Diverifikasi 17-Jul: risk_score 0,55-0,85+, decision
                    ALERT/ESCALATE genuinely muncul dari akumulasi sinyal
                    fan_in + device_sharing + model, BUKAN dipaksa/fabrikasi.

Cara pakai:
  python producer.py --mode replay --batch-size 100 --delay 0.5
  python producer.py --mode simulate --delay 0.1
  python producer.py --mode demo-replay --limit 40 --min-delay 1 --max-delay 4
  python producer.py --mode demo-collector --min-delay 0.3 --max-delay 0.8
"""

import os
import sys
import json
import time
import uuid
import random
import argparse
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TX        = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions.raw")
DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")

CHANNELS     = ["mobile", "atm", "internet", "teller", "qris"]
INSTITUTIONS = ["BANK_A", "BANK_B"]


def _make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        # key_serializer (fix QC 20-Jul, P4.8): lihat _send_tx — partisi
        # di-hash dari KEY, jadi key wajib bytes.
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        acks="all",
        retries=3,
    )


def _send_tx(producer: KafkaProducer, event: dict):
    """Publish transaksi dgn KEY = from_account (fix QC 20-Jul, P4.8).

    KENAPA PENTING (bukan kosmetik): sejak fast-path pakai TGN-streaming,
    tiap akun punya MEMORY STATE di Redis yg di-update read-modify-write.
    Kalau publish TANPA key, Kafka membagi round-robin -> transaksi akun yg
    SAMA bisa mendarat di partisi berbeda -> di-proses consumer berbeda
    SECARA BERSAMAAN -> dua consumer baca memory yg sama, hitung, lalu
    saling menimpa (LOST UPDATE) -> sinyal akun itu hilang sebagian.
    XGBoost dulu stateless per-transaksi jadi tak kena masalah ini; TGN kena.
    Dgn key=from_account, Kafka meng-hash akun ke partisi TETAP -> semua
    transaksi KELUAR akun itu diproses consumer yg sama, berurutan.
    BATAS JUJUR: ini menyelesaikan sisi PENGIRIM (akun sbg src, yaitu yg
    di-SKOR). Update memory sisi PENERIMA (dst) masih bisa lintas-partisi —
    lihat catatan multi-consumer di PIPELINE.txt Phase 4.8/15.5.
    """
    producer.send(TOPIC_TX, key=str(event.get("from_account") or ""), value=event)


def _row_to_event(row: dict) -> dict:
    """Konversi row DB/dict ke event JSON yang akan di-publish."""
    return {
        "tx_id":          row.get("tx_id", f"TX-{uuid.uuid4().hex[:12].upper()}"),
        "from_account":   str(row["from_account"]),
        "to_account":     str(row["to_account"]),
        "amount":         float(row["amount"]),
        "currency":       row.get("currency", "IDR"),
        "channel":        row.get("channel", "mobile"),
        "payment_format": row.get("payment_format", "Transfer"),
        "tx_timestamp":   str(row.get("tx_timestamp", datetime.utcnow())),
        "device_id":      str(row.get("device_id", "")),
        "institution_id": row.get("institution_id", "BANK_A"),
        "is_laundering":  int(row.get("is_laundering", 0)),
        "typology":       row.get("typology", None),
    }


# ── MODE 1: Replay dari PostgreSQL ──────────────────────────────────────────

def run_replay(producer: KafkaProducer, batch_size: int, delay: float):
    """
    Baca transaksi dari PostgreSQL berurutan (order by tx_timestamp),
    publish ke Kafka satu per satu dengan delay antar batch.
    """
    engine = create_engine(DATABASE_URL)
    print(f"[replay] Koneksi ke PostgreSQL...")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
    print(f"[replay] Total transaksi: {total:,} | batch={batch_size} | delay={delay}s")

    offset = 0
    published = 0

    while offset < total:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM transactions ORDER BY tx_timestamp LIMIT :lim OFFSET :off"
            ), {"lim": batch_size, "off": offset}).mappings().all()

        for row in rows:
            event = _row_to_event(dict(row))
            _send_tx(producer, event)
            published += 1

        producer.flush()
        offset += batch_size
        print(f"  published {published:,}/{total:,} transaksi", end="\r")

        if delay > 0:
            time.sleep(delay)

    print(f"\n[replay] Selesai - {published:,} transaksi dipublish ke topic '{TOPIC_TX}'")


# ── MODE 1b: Demo-replay (test-split accounts, interval acak) ──────────────

DYG_SPLIT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "dyg_scores_cache.pkl"
)


def _load_test_split_accounts(sample_size: int) -> list[str]:
    """
    Ambil sample account_id berlabel dyg_split=="test" dari cache DyGFormer —
    akun yang TIDAK PERNAH dipakai training model manapun (TGN/DyGFormer/
    XGBoost). Di-sample (bukan pakai semua ~297rb akun) supaya filter query
    Postgres di bawah tetap ringan dan hasilnya tetap acak/representatif.
    """
    try:
        split_df = pd.read_pickle(DYG_SPLIT_CACHE)
    except FileNotFoundError:
        print(f"[demo-replay] File cache tidak ditemukan: {DYG_SPLIT_CACHE}")
        return []
    test_accounts = split_df.loc[split_df["dyg_split"] == "test", "account_id"].tolist()
    if len(test_accounts) > sample_size:
        test_accounts = random.sample(test_accounts, sample_size)
    return test_accounts


def run_demo_replay(producer: KafkaProducer, limit: int, min_delay: float, max_delay: float,
                     account_sample: int, illicit_count: int = 0):
    """
    Replay transaksi ASLI yang melibatkan akun test-split DyGFormer saja,
    urut kronologis (tx_timestamp), dgn interval ACAK antar transaksi —
    supaya terasa spt perilaku nasabah nyata, bukan metronom tetap. tx_id
    yg dipakai tetap TX-... asli dari tabel transactions (bukan generate
    baru), sehingga dashboard (dana_berisiko, dll) otomatis akurat tanpa
    perubahan lain.

    illicit_count (fix 17-Jul, ditemukan saat tes live): rate illicit asli
    di dataset ini ~0,125% — sample MURNI acak sebesar --limit (biasanya
    puluhan) hampir pasti 0 transaksi ilegal secara statistik (bukan bug,
    cuma soal ukuran sample). Utk demo video meyakinkan (bukan hoki-hokian
    nunggu alert muncul), sengaja SELIPKAN N transaksi is_laundering=1 ASLI
    dari test-split (BUKAN fabrikasi) di antara transaksi normal acak —
    tetap "genuinely belum pernah dilihat model" krn tetap dari dyg_split
    =="test", cuma sample-nya dikurasi (bukan murni acak), sama spt rencana
    awal "curated sequence" yg sudah disepakati.
    """
    engine = create_engine(DATABASE_URL)
    print(f"[demo-replay] Ambil sample {account_sample} akun test-split (dyg_split=='test')...")
    test_accounts = _load_test_split_accounts(account_sample)
    if not test_accounts:
        print("[demo-replay] Tidak ada akun test-split ditemukan - cek DYG_SPLIT_CACHE.")
        return
    print(f"[demo-replay] {len(test_accounts):,} akun dipakai sbg filter. Query transaksi...")

    # Fix (16-Jul, ditemukan saat tes): banyak transaksi paling awal di
    # dataset AMLWorld ini self-loop (from_account == to_account, ~5,8%
    # dari total, cenderung menumpuk di awal timeline — kemungkinan
    # representasi setoran awal/cash deposit). Utk demo video, transaksi
    # "kirim ke diri sendiri" tidak menggambarkan aliran dana lintas
    # jaringan yg jadi inti nilai jual MuleRadar — dikecualikan.
    # Fix (QC, direvisi): awalnya cuma "amount > 0", tapi ternyata banyak
    # transaksi bernilai desimal receh (0,01-0,07 USD) yg tetap kebulat
    # jadi "Rp0" di layar krn format print :,.0f — ambang dinaikkan ke
    # >=1000 supaya nominal yg tampil di video selalu terlihat berarti.
    base_filter = """
        (from_account = ANY(:accs) OR to_account = ANY(:accs))
        AND from_account != to_account
        AND amount >= 1000
    """

    with engine.connect() as conn:
        illicit_rows = []
        if illicit_count > 0:
            illicit_rows = conn.execute(
                text(f"""
                    SELECT tx_id, from_account, to_account, amount, currency, channel,
                           payment_format, tx_timestamp, device_id, institution_id,
                           is_laundering, typology
                    FROM transactions
                    WHERE {base_filter} AND is_laundering = 1
                    ORDER BY tx_timestamp
                    LIMIT :lim
                """),
                {"accs": test_accounts, "lim": illicit_count},
            ).mappings().all()
            print(f"[demo-replay] {len(illicit_rows)} transaksi ILEGAL asli (test-split) disiapkan utk diselipkan.")

        normal_limit = max(limit - len(illicit_rows), 0)
        normal_rows = conn.execute(
            text(f"""
                SELECT tx_id, from_account, to_account, amount, currency, channel,
                       payment_format, tx_timestamp, device_id, institution_id,
                       is_laundering, typology
                FROM transactions
                WHERE {base_filter} AND is_laundering = 0
                ORDER BY tx_timestamp
                LIMIT :lim
            """),
            {"accs": test_accounts, "lim": normal_limit},
        ).mappings().all() if normal_limit > 0 else []

    rows = sorted(list(illicit_rows) + list(normal_rows), key=lambda r: r["tx_timestamp"])

    if not rows:
        print("[demo-replay] Tidak ada transaksi ditemukan utk akun sample ini - coba lagi (sample beda tiap run).")
        return

    print(f"[demo-replay] {len(rows)} transaksi siap direplay | interval acak {min_delay}-{max_delay}s | Ctrl+C utk stop")
    published = 0
    try:
        for i, row in enumerate(rows):
            event = _row_to_event(dict(row))
            _send_tx(producer, event)
            producer.flush()
            published += 1
            tag = " [ILLICIT-ASLI]" if event["is_laundering"] else ""
            print(f"  [{published}/{len(rows)}] {event['from_account']} -> {event['to_account']} "
                  f"Rp{event['amount']:,.0f} ({event['tx_id']}){tag}")
            # Fix (QC): skip sleep setelah transaksi TERAKHIR — sebelumnya ada
            # jeda tak perlu (sampai --max-delay detik) antara baris terakhir
            # tampil dan pesan "Selesai", terasa dead air di rekaman video.
            if i < len(rows) - 1:
                time.sleep(random.uniform(min_delay, max_delay))
    except KeyboardInterrupt:
        pass
    print(f"\n[demo-replay] Selesai - {published}/{len(rows)} transaksi dipublish ke topic '{TOPIC_TX}'")


# ── MODE 1c: Demo-collector (jamin alert genuinely trigger saat demo) ──────

def _find_top_collector(engine, pool_size: int = 25) -> str:
    """
    Cari akun kolektor asli utk demo. Fix (17-Jul, permintaan user "jangan
    akun kolektor yang sama terus, mau variasi kaya perilaku nasabah asli
    tapi boleh juga sama krn banyak transaksi"): SEBELUMNYA selalu ambil
    fan-in TERTINGGI #1 (deterministik, akun sama tiap run). Sekarang: ambil
    TOP `pool_size` kandidat (semua tetap fan-in tinggi & genuine, filter
    typology != 'NaN' supaya cuma dari kolektor yg tervalidasi berlabel
    tipologi asli hasil injeksi 17-Jul), lalu PILIH ACAK di antaranya —
    variasi akun antar-run, tapi tetap terjamin fan-in kuat (bukan asal
    akun random tanpa pola).
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT to_account, COUNT(DISTINCT from_account) AS n_senders
            FROM transactions
            WHERE is_laundering = 1 AND typology != 'NaN'
            GROUP BY to_account
            ORDER BY n_senders DESC
            LIMIT :pool
        """), {"pool": pool_size}).mappings().all()
    if not rows:
        return None
    return random.choice(rows)["to_account"]


def run_demo_collector(producer: KafkaProducer, min_delay: float, max_delay: float,
                        account_id: str = None, build_delay: float = 0.1,
                        max_outgoing: int = 15, max_incoming: int = 300):
    """
    Replay SELURUH riwayat transaksi 1 akun kolektor ASLI (lihat docstring
    modul utk penjelasan lengkap kenapa ini genuinely trigger alert, bukan
    fabrikasi). Urutan WAJIB: semua transaksi MASUK dulu (bangun fan_in di
    Redis feature store), baru transaksi KELUAR (di situ kolektor DIEVALUASI
    oleh realtime_scorer.py — evaluasi selalu di sisi from_account transaksi
    yg sedang diproses, lihat realtime_scorer.py::score()).

    Fix (17-Jul, ditemukan saat tes live): dua masalah pacing dari versi awal.
    (1) Transaksi self-loop kolektor (from==to==kolektor) muncul di KEDUA
    query (incoming DAN outgoing) krn to_account DAN from_account-nya sama
    -> terkirim dobel, TRIGGER EVALUASI LEBIH AWAL secara tidak sengaja
    (during fase "incoming" yg harusnya diam) krn score() selalu baca
    tx["from_account"]. Fix: dedup by tx_id, self-loop DIPRIORITASKAN masuk
    fase "outgoing" (memang di situ dia genuinely termasuk secara semantik).
    (2) --min-delay/--max-delay SATU nilai dipakai utk SEMUA 64 transaksi ->
    49 transaksi "diam" (build_delay CEPAT, tak perlu ditonton detail) dan
    15 transaksi "evaluasi" (yg justru harus terasa "beberapa detik sekali
    muncul alert" sesuai rencana demo) malah sama cepatnya -> alert
    menumpuk ~15 dalam 6 detik di akhir, bukan tersebar. Sekarang fase
    outgoing pakai min_delay/max_delay (default lebih lambat/dramatis),
    fase incoming pakai build_delay terpisah (default cepat, background).
    (3) max_outgoing (fix 17-Jul, ditemukan pas coba akun 83F4E0000): akun
    kolektor auto-pilih BISA punya ratusan transaksi keluar (hub/institusi
    dgn byk aktivitas legit di luar pola kolektor) — kalau semuanya direplay
    dgn delay lambat (2-4d), demo bisa makan puluhan menit. Outgoing di-CAP
    ke max_outgoing (ambil yg paling awal via ORDER BY tx_timestamp), supaya
    durasi demo selalu terprediksi. Tidak berlaku utk incoming (build_delay
    tetap cepat, aman walau ratusan baris).
    """
    engine = create_engine(DATABASE_URL)
    if not account_id:
        print("[demo-collector] Cari akun kolektor fan-in tertinggi (is_laundering=1 asli)...")
        account_id = _find_top_collector(engine)
    if not account_id:
        print("[demo-collector] Tidak ketemu akun kolektor - cek data transactions.")
        return
    print(f"[demo-collector] Akun kolektor: {account_id}")

    with engine.connect() as conn:
        incoming = conn.execute(text("""
            SELECT tx_id, from_account, to_account, amount, currency, channel,
                   payment_format, tx_timestamp, device_id, institution_id,
                   is_laundering, typology
            FROM transactions WHERE to_account = :acc
            ORDER BY tx_timestamp LIMIT :lim
        """), {"acc": account_id, "lim": max_incoming}).mappings().all()
        outgoing = conn.execute(text("""
            SELECT tx_id, from_account, to_account, amount, currency, channel,
                   payment_format, tx_timestamp, device_id, institution_id,
                   is_laundering, typology
            FROM transactions WHERE from_account = :acc
            ORDER BY tx_timestamp LIMIT :lim
        """), {"acc": account_id, "lim": max_outgoing}).mappings().all()

    if not incoming or not outgoing:
        print(f"[demo-collector] Akun {account_id} tidak punya cukup riwayat "
              f"(masuk={len(incoming)}, keluar={len(outgoing)}) - coba akun lain via --collector-account.")
        return

    # Dedup: self-loop (tx_id sama muncul di incoming DAN outgoing) cuma
    # dikirim SEKALI, di fase outgoing (di situ dia memang genuinely
    # "kolektor kirim keluar", walau tujuannya diri sendiri).
    outgoing_ids = {r["tx_id"] for r in outgoing}
    incoming = [r for r in incoming if r["tx_id"] not in outgoing_ids]

    print(f"[demo-collector] {len(incoming)} transaksi MASUK (bangun sinyal, cepat/background) "
          f"+ {len(outgoing)} transaksi KELUAR (kolektor dievaluasi, alert muncul di sini) | "
          f"build_delay={build_delay}s, eval_delay={min_delay}-{max_delay}s | Ctrl+C utk stop")

    published = 0
    total = len(incoming) + len(outgoing)
    try:
        for row in incoming:
            event = _row_to_event(dict(row))
            _send_tx(producer, event)
            producer.flush()
            published += 1
            print(f"  [{published}/{total}] {event['from_account']} -> {event['to_account']} "
                  f"Rp{event['amount']:,.0f} ({event['tx_id']}) <- membangun sinyal")
            time.sleep(build_delay)

        print(f"[demo-collector] Sinyal terbangun. Mulai fase evaluasi - alert akan muncul beberapa detik sekali...")
        for i, row in enumerate(outgoing):
            event = _row_to_event(dict(row))
            _send_tx(producer, event)
            producer.flush()
            published += 1
            print(f"  [{published}/{total}] {event['from_account']} -> {event['to_account']} "
                  f"Rp{event['amount']:,.0f} ({event['tx_id']}) <- KOLEKTOR DIEVALUASI, cek Alert List")
            if i < len(outgoing) - 1:
                time.sleep(random.uniform(min_delay, max_delay))
    except KeyboardInterrupt:
        pass
    print(f"\n[demo-collector] Selesai - {published}/{total} transaksi dipublish ke topic '{TOPIC_TX}'")


# ── MODE 2: Simulate transaksi baru ─────────────────────────────────────────

def _generate_tx() -> dict:
    """Generate satu transaksi sintetis random."""
    account_pool = [f"ACC-{i:06d}" for i in range(1000)]
    is_fraud = random.random() < 0.03  # 3% fraud rate

    if is_fraud:
        # Pola judol: banyak kecil ke satu kolektor
        from_acc = f"JUDOL-PLAYER-{random.randint(0, 499):04d}"
        to_acc   = f"JUDOL-COLL-{random.randint(0, 2):02d}"
        amount   = round(random.uniform(50_000, 500_000) / 1000) * 1000
        channel  = "mobile"
        typology = "judol"
    else:
        from_acc = random.choice(account_pool)
        to_acc   = random.choice(account_pool)
        while to_acc == from_acc:
            to_acc = random.choice(account_pool)
        amount   = round(random.uniform(10_000, 50_000_000))
        channel  = random.choice(CHANNELS)
        typology = None

    return {
        "tx_id":          f"SIM-{uuid.uuid4().hex[:12].upper()}",
        "from_account":   from_acc,
        "to_account":     to_acc,
        "amount":         amount,
        "currency":       "IDR",
        "channel":        channel,
        "payment_format": "Transfer",
        "tx_timestamp":   datetime.utcnow().isoformat(),
        "device_id":      f"DEV-{random.randint(0, 9999):06d}",
        "institution_id": random.choice(INSTITUTIONS),
        "is_laundering":  1 if is_fraud else 0,
        "typology":       typology,
    }


def run_simulate(producer: KafkaProducer, delay: float):
    """Generate transaksi sintetis tanpa henti dan publish ke Kafka."""
    print(f"[simulate] Streaming ke topic '{TOPIC_TX}' | delay={delay}s | Ctrl+C untuk stop")
    published = 0
    try:
        while True:
            tx = _generate_tx()
            _send_tx(producer, tx)
            published += 1
            if published % 100 == 0:
                producer.flush()
                print(f"  published {published:,} transaksi simulasi", end="\r")
            time.sleep(delay)
    except KeyboardInterrupt:
        producer.flush()
        print(f"\n[simulate] Dihentikan - {published:,} transaksi dipublish")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["replay", "simulate", "demo-replay", "demo-collector"], default="replay")
    parser.add_argument("--batch-size", type=int,   default=100)
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Detik antara batch (replay) atau antar transaksi (simulate)")
    parser.add_argument("--limit",          type=int,   default=40,
                        help="[demo-replay] jumlah transaksi yg direplay (target durasi demo video)")
    parser.add_argument("--min-delay",      type=float, default=1.0,
                        help="[demo-replay] batas bawah interval acak (detik)")
    parser.add_argument("--max-delay",      type=float, default=4.0,
                        help="[demo-replay] batas atas interval acak (detik)")
    parser.add_argument("--account-sample", type=int,   default=500,
                        help="[demo-replay] jumlah akun test-split di-sample sbg filter query")
    parser.add_argument("--illicit-count",  type=int,   default=0,
                        help="[demo-replay] jumlah transaksi ILEGAL ASLI (is_laundering=1, test-split) "
                             "yg diselipkan di antara transaksi normal, supaya alert dijamin muncul saat demo")
    parser.add_argument("--collector-account", type=str, default=None,
                        help="[demo-collector] account_id kolektor spesifik (default: auto-cari fan-in tertinggi)")
    parser.add_argument("--build-delay", type=float, default=0.1,
                        help="[demo-collector] interval CEPAT antar transaksi fase 'membangun sinyal' "
                             "(background, tidak menghasilkan alert) - beda dari --min/max-delay yg "
                             "dipakai di fase evaluasi (yg menghasilkan alert, dibuat lebih lambat/dramatis)")
    parser.add_argument("--max-outgoing", type=int, default=15,
                        help="[demo-collector] batas jumlah transaksi keluar yg direplay (fase evaluasi, "
                             "delay lambat) - cegah demo kepanjangan kalau akun kolektor punya ratusan tx keluar")
    parser.add_argument("--max-incoming", type=int, default=300,
                        help="[demo-collector] batas jumlah transaksi masuk yg direplay (fase build, cepat)")
    args = parser.parse_args()

    print(f"[producer] Connecting ke Kafka {KAFKA_BOOTSTRAP}...")
    producer = _make_producer()
    print(f"[producer] Connected. Mode: {args.mode}")

    if args.mode == "replay":
        run_replay(producer, args.batch_size, args.delay)
    elif args.mode == "demo-replay":
        run_demo_replay(producer, args.limit, args.min_delay, args.max_delay,
                         args.account_sample, args.illicit_count)
    elif args.mode == "demo-collector":
        run_demo_collector(producer, args.min_delay, args.max_delay,
                            args.collector_account, args.build_delay,
                            args.max_outgoing, args.max_incoming)
    else:
        run_simulate(producer, args.delay)

    producer.close()


if __name__ == "__main__":
    main()
