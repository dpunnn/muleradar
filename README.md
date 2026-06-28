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
                         │  XGBoost + TGN + DyGFormer ensemble     │
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
5. **ML ensemble** — XGBoost (tabular) + ManualTGN (temporal memory) + DyGFormerNode (temporal graph transformer)
6. **7 typology pack Indonesia** — judol ring, QRIS fraud, dormant activation, PEP network, vendor cangkang, smurf layering, rapid in-out (sesuai Tipologi PPATK + kalibrasi BI)

**Diferensiasi vs GambitHunter:** GambitHunter menemukan rekening judol dan berhenti. MuleRadar meneruskan ke graph tracing jaringan money mule dan menghasilkan LTKM resmi untuk PPATK — pipeline investigasi penuh dari situs judol hingga laporan hukum.

## Hasil (Ablation Study)

Dilatih pada **AMLWorld** (money laundering benchmark, Altman et al. NeurIPS 2023) + injeksi 7 typology Indonesia. Evaluasi node classification pada **stratified split** (distribusi illicit seragam 48.6% di semua split, data & metodologi identik antar model):

| Model                                                     | PR-AUC            | F1@0.5 | Precision |
| --------------------------------------------------------- | ----------------- | ------ | --------- |
| XGBoost (20 fitur)                                        | 0.9768            | 0.9214 | 0.9818    |
| ManualTGN (20 fitur)                                      | 0.9771            | 0.9238 | 0.9894    |
| Ensemble XGBoost+TGN (0.5/0.5)                            | 0.9793            | —     | —        |
| **DyGFormerNode** (Yu et al. 2023)                  | **0.9843**  | 0.8360 | 0.7259    |
| **Ensemble 3-model** (xgb=0.30, tgn=0.30, dyg=0.40) | **≥0.984** | —     | —        |

**20 node features** per rekening: 13 baseline (degree, amount, night ratio, dll.) + 7 behavioral (burst ratio, dormancy days, structuring score, counterparty HHI, channel entropy, inter-tx std, round amount ratio).

**Precision 0.99** pada ManualTGN — false positive minimal, krusial agar analis tidak kebanjiran alert palsu. DyGFormerNode unggul di PR-AUC (threshold-agnostic), TGN unggul di F1 pada threshold=0.5.

Skala data: **56 juta transaksi** diproses, **3,37 juta rekening**, graph engine Neo4j live dengan **2,3 juta rekening & 176 juta edge**.

## Arsitektur (Lambda)

- **Fast path** (real-time, <5ms/transaksi): rolling feature store Redis + XGBoost + signal-first scoring (fan-in/out, velocity, rapid cash-out) — tahan cold-start, explainable, tanpa label.
- **Slow path** (batch): 3-model ensemble (XGBoost + ManualTGN + DyGFormerNode) untuk re-scoring mendalam berbasis pola jaringan temporal. DyGFormerNode menggunakan K=10 temporal neighbor terbaru per rekening, fp16 + gradient checkpointing untuk efisiensi memori.

## Tech Stack

| Layer            | Teknologi                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Data             | PostgreSQL                                                                                      |
| Graph engine     | Neo4j Community + GDS                                                                           |
| Streaming        | Kafka + Zookeeper                                                                               |
| Feature store    | Redis                                                                                           |
| Detection        | Rules + XGBoost + ManualTGN + DyGFormerNode (implementasi sendiri, terinspirasi Yu et al. 2023) |
| OSINT crawler    | Playwright (async, stealth) + Tesseract OCR                                                     |
| OSINT validation | cekrekening.id — Komdigi public database                                                       |
| Ingestion        | gRPC (produksi) / Kafka (MVP)                                                                   |
| API              | FastAPI                                                                                         |
| Frontend         | React + Vite + Cytoscape.js                                                                     |
| Orkestrasi       | Docker Compose (dev) / Kubernetes (produksi)                                                    |

## Struktur Repo

```
muleradar/
├── backend/
│   ├── graph/          # Neo4j builder, analytics, visualisasi
│   ├── detection/      # rules, features, model (XGBoost), alerts
│   ├── ml/             # DyGFormerNode/ManualTGN dataset/model, training, ensemble, ablation
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

# 3. Training semua model
cd ../../backend
python -m ml.train_tgn                                                  # ManualTGN baseline
python -m ml.train_dyg --fp16 --k-neighbors 10                         # DyGFormerNode (primary)

# 4. Ablation & ensemble scoring
python -m ml.eval_ablation                                              # XGBoost + TGN (stratified)
python -m ml.ensemble                                                   # 3-model ensemble batch

# 5. Demo streaming real-time (2 terminal)
python -m streaming.consumer       # detektor
python -m streaming.producer --mode simulate --delay 0.1   # sumber transaksi

# 6. Graph Investigation Workbench (visualisasi LIVE dari Neo4j)
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

*Status: 3-model ensemble selesai & tervalidasi (DyGFormerNode 0.984 PR-AUC, stratified split). 20 node features aktif. 7 typologi PPATK + kalibrasi BI ter-inject. Graph engine 176 juta edge live. Pipeline real-time AI berjalan. OSINT Intelligence module, Retrospective Sweep, Dual-Source Active Learning, API layer + frontend dalam pengembangan aktif.*
