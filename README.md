# MuleRadar

**Graph-Based Investigation Workbench untuk Deteksi Jaringan Rekening Penampung & Money Laundering di Ekosistem Keuangan Digital Indonesia**

Digdaya × Hackathon 2026 — Bank Indonesia

---

## Masalah

Money laundering modern beroperasi sebagai **jaringan rekening penampung (mule network)** yang tersebar — bukan satu rekening tunggal. Pelaku memecah dana jadi banyak transaksi kecil (structuring), melapis melalui rantai rekening (layering), lalu cash-out. Sistem AML konvensional memeriksa rekening **satu per satu** dengan threshold, sehingga pola jaringan ini lolos.

OJK mencatat **Rp7,5 triliun** kerugian dari 530.794 rekening sejak IASC berjalan, dengan gap ~430.000 rekening yang dilaporkan tapi belum tertangani.

## Solusi

MuleRadar beroperasi dalam dua mode: **proaktif** (hunting rekening judol sebelum masuk sistem bank) dan **reaktif** (deteksi pola mencurigakan dari transaksi yang masuk):

```
[PROAKTIF] Situs Judol ──Playwright──► OSINT Engine ──► cekrekening.id
                                              │
                                     osint_accounts (PostgreSQL)
                                              │
                                    Seed nodes Neo4j ◄──────────────┐
                                                                     │
[REAKTIF]  Bank ──ingestion──► Kafka ──► REAL-TIME SCORING ──► Neo4j graph + Alert
                                              │
                         ┌────────────────────┴────────────────────┐
                         │  Feature Store (Redis) → Signal-first   │
                         │  XGBoost + TGN ensemble → decision      │
                         │  (ALERT / ESCALATE / FREEZE)            │
                         └─────────────────────────────────────────┘
                                              │
                                    Case Management + LTKM Otomatis
```

Deteksi **berlapis**:
1. **OSINT Intelligence** — crawl situs judol, ekstrak rekening bandar, deteksi jaringan via shared rekening, cross-check cekrekening.id (Komdigi)
2. **AML core rules** — structuring, fan-out, layering, cycle
3. **Statistical anomaly** — z-score/percentile adaptif (bukan threshold statis)
4. **Graph motif** — deteksi topologi cycle (A→B→C→A)
5. **ML ensemble** — XGBoost (tabular) + TGN (temporal graph)
6. **7 typology pack Indonesia** — judol ring, QRIS fraud, dormant activation, PEP network, vendor cangkang, dll.

**Diferensiasi vs GambitHunter:** GambitHunter menemukan rekening judol dan berhenti. MuleRadar meneruskan ke graph tracing jaringan money mule dan menghasilkan LTKM resmi untuk PPATK — pipeline investigasi penuh dari situs judol hingga laporan hukum.

## Hasil (Ablation Study)

Dilatih pada **AMLWorld** (money laundering benchmark, Altman et al. NeurIPS 2023) + injeksi 7 typology Indonesia. Evaluasi node classification pada split fair (data & split identik):

| Model | PR-AUC | F1 | Precision |
|---|---|---|---|
| XGBoost | 0.976 | 0.921 | 0.982 |
| **TGN** | **0.977** | 0.924 | 0.989 |
| **Ensemble (XGBoost+TGN)** | **0.979** | 0.923 | 0.989 |

TGN mengalahkan XGBoost solo; ensemble terbaik. **Precision 0.99** — false positive minimal, krusial agar analis tidak kebanjiran alert palsu.

Skala data: **181 juta transaksi** diproses, graph engine Neo4j live dengan **2,3 juta rekening & 176 juta edge**.

## Arsitektur (Lambda)

- **Fast path** (real-time, <5ms/transaksi): rolling feature store Redis + XGBoost + signal-first scoring (fan-in/out, velocity, rapid cash-out) — tahan cold-start, explainable, tanpa label.
- **Slow path** (batch): TGN ensemble untuk re-scoring mendalam berbasis pola jaringan.

## Tech Stack

| Layer | Teknologi |
|---|---|
| Data | PostgreSQL |
| Graph engine | Neo4j Community + GDS |
| Streaming | Kafka + Zookeeper |
| Feature store | Redis |
| Detection | Rules + XGBoost + TGN (PyTorch Geometric) |
| OSINT crawler | Playwright (async, stealth) + Tesseract OCR |
| OSINT validation | cekrekening.id — Komdigi public database |
| Ingestion | gRPC (produksi) / Kafka (MVP) |
| API | FastAPI |
| Frontend | React + Vite + Cytoscape.js |
| Orkestrasi | Docker Compose (dev) / Kubernetes (produksi) |

## Struktur Repo

```
muleradar/
├── backend/
│   ├── graph/          # Neo4j builder, analytics, visualisasi
│   ├── detection/      # rules, features, model (XGBoost), alerts
│   ├── ml/             # TGN dataset/model, training, ensemble, ablation
│   ├── streaming/      # Kafka producer/consumer, feature store, real-time scorer
│   ├── osint/          # crawler, extractor, network detector, cekrekening, seeder
│   ├── api/            # FastAPI routes: dashboard, alerts, graph, cases, osint
│   ├── llm/            # case summary + LTKM generation
│   └── db/             # schema PostgreSQL
├── frontend/           # React + Vite dashboard (6 halaman)
├── data/scripts/       # ETL: postprocess, inject typologi, load
├── docker-compose.yml  # PostgreSQL + Neo4j + Kafka + Redis + backend
└── PIPELINE.txt        # rencana build end-to-end
```

> Catatan: dataset (puluhan GB) dan model artifacts tidak di-commit (lihat `.gitignore`). AMLWorld dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

## Menjalankan

```bash
# 1. Infrastruktur
docker compose up -d

# 2. Proses data + inject typologi
cd data/scripts
python postprocess.py --input <AMLWorld.csv> --output ../processed/transactions.csv
python inject_typologies.py --input ../processed/transactions.csv --output ../processed/transactions_injected.csv
python load_to_db.py --input ../processed/transactions_injected.csv

# 3. Training + ablation
cd ../../backend
python -m ml.train_tgn
python -m ml.eval_ablation

# 4. Demo streaming real-time (2 terminal)
python -m streaming.consumer       # detektor
python -m streaming.producer --mode simulate --delay 0.1   # sumber transaksi

# 5. Graph Investigation Workbench (visualisasi LIVE dari Neo4j)
uvicorn graph.viz_server:app --port 8050
# buka http://localhost:8050  -> graph di-query langsung dari Neo4j tiap dibuka
#   - bar kontrol: ubah jumlah sampel (Illicit/Clean) atau isi Seed (account_id)
#   - http://localhost:8050/health  -> cek koneksi & jumlah account di Neo4j
```

> Catatan: `viz_server` menyajikan subgraph sampel yang **di-query langsung dari Neo4j**
> (graph penuh ~176 juta edge), bukan HTML statis. Jalankan dari folder `backend/`
> dengan Neo4j aktif (`docker compose up -d neo4j`).

## Tim

- **Dhafin Ahamad Athalla** (BINUS) — Project Lead & Full-Stack Developer
- **Farhan Kamalhadi Elevana** (UNPAD) — Data Scientist & AI Engineer
- **Ega Jismi Muwaaffaq** (UII) — Business Analyst & Product Strategist
- **Cheysa Afrayansyah Wahyu Putra** (UII) — Market Research & Compliance Lead

---

*Status: detection engine selesai & tervalidasi (ensemble 0.979 PR-AUC), graph engine 176 juta edge live, pipeline real-time AI berjalan. OSINT Intelligence module + API layer + frontend dalam pengembangan aktif.*
