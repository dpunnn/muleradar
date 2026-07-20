"""
Phase 4.5.5 — Deteksi jaringan bandar via shared rekening.

Prinsip: satu jaringan bandar judol memakai rekening yang SAMA di banyak situs
berbeda. Jadi rekening yang muncul di >= 2 situs adalah sinyal kuat sebuah
jaringan terorganisir, bukan situs tunggal.

Modul ini membangun graph bipartit (situs — rekening), lalu mengelompokkan
rekening + situs yang saling terhubung menjadi satu "network" (connected
component via Union-Find). Risk level ditentukan dari jumlah situs unik yang
menghubungkan komponen tersebut.

Input  : baca osint_accounts.sumber_url (diisi seeder 4.5.6).
Output : list network {rekening_list, site_list, risk_level}, di-persist ke
         osint_networks.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)


@dataclass
class Network:
    rekening_list: list[str] = field(default_factory=list)
    site_list: list[str] = field(default_factory=list)
    risk_level: str = "LOW"


class _UnionFind:
    """Union-Find sederhana dengan path compression untuk clustering bipartit."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def risk_from_site_count(n_sites: int) -> str:
    """
    HIGH: rekening/jaringan tersebar di 3+ situs; MED: 2; LOW: 1.

    Public (dipakai ulang oleh seeder.py & exporter.py) supaya threshold
    risiko HIGH/MED/LOW punya SATU sumber kebenaran — tidak di-hardcode
    ulang di tempat lain yang bisa jadi tidak konsisten.
    """
    if n_sites >= 3:
        return "HIGH"
    if n_sites == 2:
        return "MED"
    return "LOW"


def build_networks(account_sites: dict[str, list[str]]) -> list[Network]:
    """
    account_sites: {rekening: [url, url, ...]}.

    Union rekening dengan tiap situsnya (namespace dibedakan: prefix "acc:" /
    "site:" agar tidak bentrok), lalu kelompokkan per connected component.
    """
    uf = _UnionFind()
    for rekening, sites in account_sites.items():
        acc_key = f"acc:{rekening}"
        for url in sites:
            uf.union(acc_key, f"site:{url}")

    # Kelompokkan anggota per root.
    comp_accounts: dict[str, set[str]] = defaultdict(set)
    comp_sites: dict[str, set[str]] = defaultdict(set)
    for rekening, sites in account_sites.items():
        root = uf.find(f"acc:{rekening}")
        comp_accounts[root].add(rekening)
        for url in sites:
            comp_sites[root].add(url)

    networks: list[Network] = []
    for root, accts in comp_accounts.items():
        sites = comp_sites[root]
        networks.append(Network(
            rekening_list=sorted(accts),
            site_list=sorted(sites),
            risk_level=risk_from_site_count(len(sites)),
        ))
    # Jaringan paling berisiko/terluas di atas.
    networks.sort(key=lambda n: (n.risk_level != "HIGH", -len(n.site_list)))
    return networks


def load_account_sites(engine) -> dict[str, list[str]]:
    """Baca peta rekening→situs dari osint_accounts.sumber_url."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT rekening, sumber_url FROM osint_accounts
            WHERE sumber_url IS NOT NULL
        """)).fetchall()
    return {r[0]: list(r[1]) for r in rows if r[1]}


def persist_networks(networks: list[Network], engine) -> int:
    """
    Tulis ulang tabel osint_networks (refresh penuh — snapshot terkini).
    Hanya simpan jaringan MED/HIGH (2+ situs) — LOW = situs tunggal, bukan jaringan.
    """
    saved = 0
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE osint_networks RESTART IDENTITY"))
        for net in networks:
            if net.risk_level == "LOW":
                continue
            conn.execute(
                text("""
                    INSERT INTO osint_networks (rekening_list, site_list, risk_level)
                    VALUES (:rek, :sites, :risk)
                """),
                {
                    "rek": net.rekening_list,
                    "sites": net.site_list,
                    "risk": net.risk_level,
                },
            )
            saved += 1
    return saved


def detect(engine=None) -> dict:
    """
    Entry point: load rekening→situs, bangun jaringan, persist yang MED/HIGH.
    Kembalikan ringkasan untuk endpoint /osint/networks.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)
    account_sites = load_account_sites(engine)
    networks = build_networks(account_sites)
    saved = persist_networks(networks, engine)
    high = sum(1 for n in networks if n.risk_level == "HIGH")
    med = sum(1 for n in networks if n.risk_level == "MED")
    return {
        "total_accounts": len(account_sites),
        "networks_saved": saved,
        "high_risk": high,
        "med_risk": med,
    }


if __name__ == "__main__":
    print("[network] hasil:", detect())
