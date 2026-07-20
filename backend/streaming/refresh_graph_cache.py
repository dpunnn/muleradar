"""
Job periodik (17-Jul, keputusan arsitektur "Prioritas 1"): isi cache 4 fitur
network/graph — device_sharing_count, n_institutions, pagerank, kcore_number —
ke Redis, dibaca real-time oleh streaming/feature_store.py.

KENAPA JOB TERPISAH, bukan hitung live per-transaksi (lihat catatan di
feature_store.py FEATURE_COLS): ke-4 fitur ini LINTAS-AKUN (pagerank/kcore
butuh seluruh graph, device_sharing/n_institutions butuh riwayat akun LAIN),
tidak feasible dihitung <5ms per-transaksi. Job ini jalan BATCH, tulis
langsung ke hash Redis acct:{id} yang sama dibaca get_model_features() — tidak
ada perubahan ke hot path scoring sama sekali.

SCOPE AKUN: hanya akun di Redis SET "known_accounts" (diisi feature_store.py
tiap transaksi live), BUKAN seluruh akun historis (2,34 juta per 17-Jul).
pageRank/kcore tetap dihitung dari GRAPH PENUH Neo4j (metrik global, itu yang
benar), tapi WRITE-BACK ke Redis di-bound ke akun yang benar-benar aktif di
jalur real-time saat ini — bukan buang resource nge-cache akun yang tak
pernah discoring.

CADENCE TERUKUR (17-Jul, benchmark nyata di 176,5 juta edge / 2,34 juta akun,
lihat PIPELINE.txt bagian arsitektur fitur real-time):
    graph.project (dual orientation, sekali utk pageRank+kcore) : ~3m35s
    pageRank.write (20 iterasi, belum full converge, cukup utk sinyal)  : ~1m
    kcore.write (butuh relationship UNDIRECTED, projeksi terpisah)     : ~1m30s
    TOTAL                                                              : ~6 menit
=> jalankan job ini via scheduler EKSTERNAL (cron/Task Scheduler/systemd timer)
   tiap 15-30 menit, BUKAN di dalam proses consumer.py — job ini CPU/memory
   berat, jangan berbagi proses dengan hot path scoring real-time.

device_sharing_count butuh index Postgres idx_tx_device (reverse-lookup
"akun lain pakai device yang sama") yang MASIH DIBANGUN saat kode ini
ditulis (144 juta baris, CREATE INDEX CONCURRENTLY berjalan di background).
Kalau index belum valid, bagian ini di-SKIP dengan log jujur (bukan gagal
diam-diam) — nilai lama di Redis (kalau ada) dibiarkan, cold-start default 0
di feature_store.py. n_institutions TIDAK butuh index baru (cukup
idx_tx_from/idx_tx_to yang sudah ada, sifatnya "1-hop" dari riwayat akun itu
sendiri) — selalu dihitung tiap siklus.

Cara pakai: python refresh_graph_cache.py
"""

import os
import sys
import time
import logging

import redis
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.features import _compute_device_sharing  # reuse definisi kanonik

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "muleradar_neo4j")
DATABASE_URL   = os.getenv("DATABASE_URL",
                            "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")

GRAPH_NAME = "acctGraphCache"


def _refresh_graph_properties(driver):
    """Proyeksi graph Neo4j (dual orientation), tulis pagerank+kcore_number
    sbg node property, lalu drop proyeksi in-memory (bebasin RAM)."""
    with driver.session() as s:
        s.run("CALL gds.graph.drop($name, false)", name=GRAPH_NAME)

        t0 = time.time()
        res = s.run(
            "CALL gds.graph.project($name, 'Account', "
            "{TRANSFER: {orientation: 'NATURAL'}, "
            " TRANSFER_UNDIR: {type: 'TRANSFER', orientation: 'UNDIRECTED'}}) "
            "YIELD nodeCount, relationshipCount, projectMillis "
            "RETURN nodeCount, relationshipCount, projectMillis",
            name=GRAPH_NAME,
        ).single()
        logger.info("Graph projected: %s node, %s rel (%.1fs)",
                    res["nodeCount"], res["relationshipCount"], time.time() - t0)

        t0 = time.time()
        pr = s.run(
            "CALL gds.pageRank.write($name, {writeProperty: 'pagerank', "
            "maxIterations: 20, relationshipTypes: ['TRANSFER']}) "
            "YIELD ranIterations, didConverge, nodePropertiesWritten "
            "RETURN ranIterations, didConverge, nodePropertiesWritten",
            name=GRAPH_NAME,
        ).single()
        logger.info("pageRank: %s iterasi, converge=%s, %s node ditulis (%.1fs)",
                    pr["ranIterations"], pr["didConverge"], pr["nodePropertiesWritten"],
                    time.time() - t0)

        t0 = time.time()
        kc = s.run(
            "CALL gds.kcore.write($name, {writeProperty: 'kcore_number', "
            "relationshipTypes: ['TRANSFER_UNDIR']}) "
            "YIELD degeneracy, nodePropertiesWritten "
            "RETURN degeneracy, nodePropertiesWritten",
            name=GRAPH_NAME,
        ).single()
        logger.info("kcore: degeneracy=%s, %s node ditulis (%.1fs)",
                    kc["degeneracy"], kc["nodePropertiesWritten"], time.time() - t0)

        s.run("CALL gds.graph.drop($name, false)", name=GRAPH_NAME)


def _fetch_graph_features(driver, account_ids: list) -> dict:
    """Baca pagerank/kcore_number utk akun ter-track saja (bounded query)."""
    with driver.session() as s:
        rows = s.run(
            "MATCH (a:Account) WHERE a.account_id IN $ids "
            "RETURN a.account_id AS id, a.pagerank AS pagerank, "
            "a.kcore_number AS kcore_number",
            ids=account_ids,
        ).data()
    return {r["id"]: (r["pagerank"] or 0.0, r["kcore_number"] or 0.0) for r in rows}


def _device_index_ready(engine) -> bool:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT indisvalid FROM pg_index "
            "WHERE indexrelid::regclass::text = 'idx_tx_device'"
        )).fetchone()
    return bool(row and row[0])


def _fetch_n_institutions(engine, account_ids: list) -> dict:
    """n_institutions: institution unik yg disentuh akun itu sendiri (in+out).
    'sendiri' -> cukup idx_tx_from/idx_tx_to yg sudah ada, TAK butuh index baru."""
    sql = text("""
        SELECT account_id, COUNT(DISTINCT institution_id) AS n_institutions
        FROM (
            SELECT from_account AS account_id, institution_id FROM transactions
            WHERE from_account = ANY(:ids) AND institution_id IS NOT NULL
            UNION ALL
            SELECT to_account AS account_id, institution_id FROM transactions
            WHERE to_account = ANY(:ids) AND institution_id IS NOT NULL
        ) u
        GROUP BY account_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"ids": account_ids}).mappings().all()
    return {r["account_id"]: r["n_institutions"] for r in rows}


def _fetch_device_sharing(engine, account_ids: list) -> dict:
    """device_sharing_count: BUTUH idx_tx_device (reverse-lookup akun lain
    yg pakai device sama) -> caller HARUS cek _device_index_ready() dulu."""
    sql_my_devices = text("""
        SELECT DISTINCT device_id FROM transactions
        WHERE from_account = ANY(:ids) AND device_id IS NOT NULL
    """)
    with engine.connect() as conn:
        devices = [r[0] for r in conn.execute(sql_my_devices, {"ids": account_ids})]
        if not devices:
            return {}
        sql_cluster = text("""
            SELECT DISTINCT from_account AS account_id, device_id
            FROM transactions
            WHERE device_id = ANY(:devices)
        """)
        rows = conn.execute(sql_cluster, {"devices": devices}).mappings().all()

    import pandas as pd
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["account_id", "device_id"])
    feat = _compute_device_sharing(df)  # definisi kanonik (threshold <=20, fix 6-Jul)
    return dict(zip(feat["account_id"], feat["device_sharing_count"]))


def main():
    r = redis.from_url(REDIS_URL, decode_responses=True)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    engine = create_engine(DATABASE_URL)

    account_ids = sorted(r.smembers("known_accounts"))
    if not account_ids:
        logger.info("known_accounts kosong -> belum ada transaksi live, skip siklus ini.")
        return

    # PRUNE entri mati (fix QC 20-Jul): known_accounts adalah SET yg HANYA
    # bertambah — tak pernah dibersihkan. Di produksi jangka panjang ia tumbuh
    # tanpa batas (tiap akun yg pernah transaksi masuk selamanya) sehingga job
    # ini makin lama makin berat, PADAHAL sebagian anggotanya sudah "mati"
    # (hash acct:{id} kedaluwarsa/ter-evict LRU). Buang anggota yg hash-nya
    # sudah tak ada — kalau akun itu transaksi lagi, feature_store menambahkan
    # kembali secara otomatis. Aman & menjaga job tetap proporsional ke akun
    # yang benar-benar aktif.
    pipe = r.pipeline(transaction=False)
    for acc in account_ids:
        pipe.exists(f"acct:{acc}")
    exists_flags = pipe.execute()
    dead = [a for a, ok in zip(account_ids, exists_flags) if not ok]
    if dead:
        for i in range(0, len(dead), 1000):
            r.srem("known_accounts", *dead[i:i + 1000])
        account_ids = [a for a, ok in zip(account_ids, exists_flags) if ok]
        logger.info("Prune known_accounts: %d entri mati dibuang (state sudah "
                    "kedaluwarsa/ter-evict).", len(dead))
    if not account_ids:
        logger.info("Semua entri known_accounts mati -> tak ada yg di-refresh.")
        return

    logger.info("Refresh cache utk %d akun ter-track.", len(account_ids))

    _refresh_graph_properties(driver)
    graph_feats = _fetch_graph_features(driver, account_ids)

    n_inst = _fetch_n_institutions(engine, account_ids)

    dev_ready = _device_index_ready(engine)
    if dev_ready:
        dev_share = _fetch_device_sharing(engine, account_ids)
    else:
        dev_share = {}
        logger.warning("idx_tx_device belum ONLINE -> device_sharing_count "
                        "DI-SKIP siklus ini (self-heal siklus berikutnya "
                        "setelah index selesai dibangun).")

    p = r.pipeline(transaction=False)
    for acc in account_ids:
        pagerank, kcore = graph_feats.get(acc, (0.0, 0.0))
        fields = {"pagerank": pagerank, "kcore_number": kcore,
                  "n_institutions": n_inst.get(acc, 0)}
        # device_sharing_count (fix QC 20-Jul, BUG STALE): dulu HANYA ditulis
        # kalau `acc in dev_share`. Akibatnya kalau akun DULU masuk cluster
        # device ketat lalu TIDAK lagi, nilai lamanya MENGENDAP selamanya di
        # Redis (stale) — akun tampak masih berbagi device padahal tidak.
        # Sekarang: saat index device SIAP, SELALU tulis (default 0 = memang
        # tak ada sharing ketat). Saat index BELUM siap, tetap di-SKIP total
        # (jangan timpa nilai bagus dgn 0 palsu — perilaku lama yg benar).
        if dev_ready:
            fields["device_sharing_count"] = dev_share.get(acc, 0)
        p.hset(f"acct:{acc}", mapping=fields)
    p.execute()

    logger.info("Selesai: %d akun di-update (pagerank+kcore+n_institutions%s).",
                len(account_ids), "+device_sharing_count" if dev_ready else "")

    driver.close()


if __name__ == "__main__":
    main()
