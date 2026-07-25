// =============================================================
// AlertList.jsx — Halaman 2: Daftar Alert
// Diadaptasi dari pagemuleradar/alerts.jsx — layout dipertahankan,
// data + filter disambungkan ke API asli (alerts.py).
// =============================================================
import { useEffect, useState, useCallback } from "react";
import { DS, Icons, Card, MonoText, TypologyBadge, RiskBar, StatusBadge, getInputStyle } from "../design/system";
import { listAlerts } from "../lib/api";

// Deteksi = AI ensemble (TGN+DyG). Alert dikategorikan 1 kolom "Kategori" dgn
// dua SUMBER (cascade): AML (pola struktural dari graph) atau Tipologi (konteks
// ID dari model). Akun tanpa tipologi jelas TIDAK lagi "UNKNOWN" — ditampilkan
// sbg pola AML-nya (fan-in/out/dst). Jadi UNKNOWN hilang dari UI.
const STATUS_OPTIONS = ["Semua", "NEW", "IN_REVIEW", "CONFIRM", "FP", "CLOSED"];
const SUMBER_OPTIONS = [
  { v: "Semua",    l: "Semua" },
  { v: "AML",      l: "AML (struktural)" },
  { v: "Tipologi", l: "Tipologi Indonesia" },
];
// Nilai kategori per sumber (dropdown kedua cascade — TANPA UNKNOWN).
const AML_VALUES = [
  { v: "FAN_IN",     l: "Fan-in (collector)" },
  { v: "FAN_OUT",    l: "Fan-out (distributor)" },
  { v: "RELAY",      l: "Relay (layering)" },
  { v: "PERIPHERAL", l: "Peripheral (ujung)" },
];
const TYPO_VALUES = [
  { v: "judol", l: "judol" }, { v: "qris", l: "qris" }, { v: "dormant", l: "dormant" },
  { v: "scam", l: "scam" }, { v: "bendahara", l: "bendahara" }, { v: "pep", l: "pep" }, { v: "vendor", l: "vendor" },
];
const AML_KEYS = new Set(AML_VALUES.map((o) => o.v));
const TYPO_KEYS = new Set(TYPO_VALUES.map((o) => o.v));
// Opsi dropdown kategori tergantung sumber terpilih.
const kategoriOptions = (sumber) => {
  const base = [{ v: "Semua", l: "Semua" }];
  if (sumber === "AML") return [...base, ...AML_VALUES];
  if (sumber === "Tipologi") return [...base, ...TYPO_VALUES];
  return [...base, ...AML_VALUES, ...TYPO_VALUES];
};
const AML_SHORT = { FAN_IN: "Fan-in", FAN_OUT: "Fan-out", RELAY: "Relay", PERIPHERAL: "Peripheral", CYCLE: "Cycle" };
const AML_COLOR = { FAN_IN: "#FF6B6B", FAN_OUT: "#FFA94D", RELAY: "#4F8EF7", PERIPHERAL: "#8C8CA1", CYCLE: "#B084F7" };
const PAGE_SIZE = 16;

const AmlBadge = ({ pattern }) => {
  if (!pattern) return <span style={{ color: DS.color.textSec, fontSize: 11 }}>—</span>;
  const c = AML_COLOR[pattern] || DS.color.textSec;
  return (
    <span style={{ fontSize: 10, fontWeight: 600, color: c, background: `${c}1a`, border: `1px solid ${c}44`, borderRadius: 5, padding: "2px 7px", whiteSpace: "nowrap" }}>
      {AML_SHORT[pattern] || pattern}
    </span>
  );
};

// Kolom "Kategori" tunggal: tipologi ID kalau jelas, kalau UNKNOWN tampil pola AML.
const KategoriBadge = ({ typology, aml_pattern }) => {
  const known = typology && typology !== "UNKNOWN";
  return known ? <TypologyBadge type={typology} small /> : <AmlBadge pattern={aml_pattern} />;
};

const FilterBar = ({ sumber, setSumber, kategori, setKategori, status, setStatus, onApply, onReset }) => (
  <Card style={{ marginBottom: 12 }} pad={12}>
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <label style={{ fontSize: 10, color: DS.color.textSec, fontWeight: 500 }}>SUMBER DETEKSI</label>
        <select value={sumber} onChange={(e) => { setSumber(e.target.value); setKategori("Semua"); }} style={{ ...getInputStyle({ borderRadius: 7 }), color: DS.color.textPri, fontSize: 12, padding: "5px 8px" }}>
          {SUMBER_OPTIONS.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
        </select>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <label style={{ fontSize: 10, color: DS.color.textSec, fontWeight: 500 }}>KATEGORI</label>
        <select value={kategori} onChange={(e) => setKategori(e.target.value)} style={{ ...getInputStyle({ borderRadius: 7 }), color: DS.color.textPri, fontSize: 12, padding: "5px 8px" }}>
          {kategoriOptions(sumber).map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
        </select>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <label style={{ fontSize: 10, color: DS.color.textSec, fontWeight: 500 }}>STATUS</label>
        <select value={status} onChange={(e) => setStatus(e.target.value)} style={{ ...getInputStyle({ borderRadius: 7 }), color: DS.color.textPri, fontSize: 12, padding: "5px 8px" }}>
          {STATUS_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </div>
      <div style={{ flex: 1 }}></div>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={onApply} style={{ background: DS.color.blue, color: "#fff", border: "none", borderRadius: 5, padding: "6px 16px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Terapkan Filter</button>
        <button onClick={onReset} style={{ background: "transparent", color: DS.color.textSec, border: `1px solid ${DS.color.border}`, borderRadius: 5, padding: "6px 12px", fontSize: 12, cursor: "pointer" }}>Reset</button>
      </div>
    </div>
  </Card>
);

const PagBtn = ({ label, active, disabled, onClick }) => (
  <button
    disabled={disabled}
    onClick={onClick}
    style={{
      minWidth: 28, height: 28, borderRadius: 5, fontSize: 12, fontWeight: active ? 700 : 400,
      background: active ? DS.color.blue : "transparent",
      color: active ? "#fff" : disabled ? DS.glass.riskBarTrack : DS.color.textSec,
      border: active ? "none" : `1px solid ${DS.color.border}`,
      cursor: disabled ? "not-allowed" : "pointer", padding: "0 6px",
    }}
  >
    {label}
  </button>
);

export default function AlertList({ onOpenDetail }) {
  const [sumber, setSumber] = useState("Semua");
  const [kategori, setKategori] = useState("Semua");
  const [status, setStatus] = useState("Semua");
  const [appliedSumber, setAppliedSumber] = useState("Semua");
  const [appliedKategori, setAppliedKategori] = useState("Semua");
  const [appliedStatus, setAppliedStatus] = useState("Semua");
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [hover, setHover] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(false);
    const params = { limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE };
    if (appliedStatus !== "Semua") params.status = appliedStatus;
    // Cascade sumber→kategori. Akun "UNKNOWN" tipologi = disajikan sbg pola AML.
    const k = appliedKategori;
    if (appliedSumber === "AML") {
      params.typology = "UNKNOWN";                  // sumber AML = tanpa tipologi jelas
      if (k !== "Semua") params.aml_pattern = k;
    } else if (appliedSumber === "Tipologi") {
      if (k === "Semua") params.only_known_typology = true;  // semua yg bertipologi
      else params.typology = k;
    } else if (k !== "Semua") {                     // sumber Semua + kategori spesifik
      if (TYPO_KEYS.has(k)) params.typology = k;
      else if (AML_KEYS.has(k)) { params.typology = "UNKNOWN"; params.aml_pattern = k; }
    }
    listAlerts(params)
      .then((d) => { setRows(d.items); setTotal(d.total); })
      .catch((e) => { console.error("Gagal load alerts:", e); setLoadError(true); setRows([]); setTotal(0); })
      .finally(() => setLoading(false));
  }, [page, appliedSumber, appliedKategori, appliedStatus]);

  useEffect(() => { load(); }, [load]);

  const applyFilter = () => { setAppliedSumber(sumber); setAppliedKategori(kategori); setAppliedStatus(status); setPage(1); };
  const resetFilter = () => { setSumber("Semua"); setKategori("Semua"); setStatus("Semua"); setAppliedSumber("Semua"); setAppliedKategori("Semua"); setAppliedStatus("Semua"); setPage(1); };

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const from = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const to = Math.min(page * PAGE_SIZE, total);

  return (
    <div style={{ flex: 1, minHeight: 0, overflow: "hidden", padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: DS.color.textPri, margin: 0 }}>Daftar Alert</h1>
          <p style={{ fontSize: 12, color: DS.color.textSec, margin: "3px 0 0" }}>{total.toLocaleString("id-ID")} alert ditemukan</p>
        </div>
        <button onClick={load} style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(107,174,255,0.12)", color: DS.color.blue, border: `1px solid ${DS.color.blue}44`, borderRadius: 6, padding: "7px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
          <span>{Icons.refresh(12)}</span> Refresh
        </button>
      </div>

      <FilterBar sumber={sumber} setSumber={setSumber} kategori={kategori} setKategori={setKategori} status={status} setStatus={setStatus} onApply={applyFilter} onReset={resetFilter} />

      <Card pad={0} style={{ overflow: "hidden", flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div style={{ overflow: "auto", flex: 1, minHeight: 0 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${DS.glass.thBorder}` }}>
                {["Alert ID", "Rekening", "Kategori", "Risk Score", "Status", "Timestamp", "Aksi"].map((h) => (
                  <th key={h} style={{ padding: "10px 8px", textAlign: "left", fontSize: 10, fontWeight: 700, color: DS.color.textPri, textTransform: "uppercase", letterSpacing: 0.4, whiteSpace: "nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={7} style={{ padding: 20, textAlign: "center", color: DS.color.textSec }}>Memuat…</td></tr>
              ) : loadError ? (
                <tr><td colSpan={7} style={{ padding: 20, textAlign: "center", color: DS.color.riskHigh || "#FF6B6B" }}>
                  Gagal memuat alert — koneksi ke server bermasalah. Coba Refresh. (BUKAN berarti tidak ada alert)
                </td></tr>
              ) : rows.length === 0 ? (
                <tr><td colSpan={7} style={{ padding: 20, textAlign: "center", color: DS.color.textSec }}>Tidak ada alert yang cocok.</td></tr>
              ) : rows.map((row, i) => {
                const isHover = hover === row.alert_id;
                return (
                  <tr
                    key={row.alert_id}
                    onMouseEnter={() => setHover(row.alert_id)}
                    onMouseLeave={() => setHover(null)}
                    style={{ background: isHover ? DS.glass.rowHover : i % 2 === 0 ? "transparent" : DS.glass.rowAlt, borderBottom: `1px solid ${DS.glass.panelBorder}50` }}
                  >
                    <td style={{ padding: "8px 8px" }}><MonoText style={{ color: DS.color.textSec, fontSize: 11 }}>{row.alert_id}</MonoText></td>
                    <td style={{ padding: "8px 8px" }}><MonoText style={{ color: DS.color.blue }}>{row.account_id}</MonoText></td>
                    <td style={{ padding: "8px 8px" }}><KategoriBadge typology={row.typology} aml_pattern={row.aml_pattern} /></td>
                    <td style={{ padding: "8px 8px" }}><RiskBar score={parseFloat(row.risk_score)} width={72} /></td>
                    <td style={{ padding: "8px 8px" }}><StatusBadge status={row.status} /></td>
                    <td style={{ padding: "8px 8px", color: DS.color.textSec, fontSize: 11, whiteSpace: "nowrap" }}>{new Date(row.created_at).toLocaleString("id-ID")}</td>
                    <td style={{ padding: "8px 8px" }}>
                      <button
                        onClick={() => onOpenDetail && onOpenDetail(row.alert_id)}
                        style={{ background: isHover ? DS.color.blue : `${DS.color.blue}22`, color: isHover ? "#fff" : DS.color.blue, border: `1px solid ${DS.color.blue}44`, borderRadius: 5, padding: "4px 12px", fontSize: 11, fontWeight: 600, cursor: "pointer", whiteSpace: "nowrap" }}
                      >
                        Detail
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", borderTop: `1px solid ${DS.color.border}`, flexShrink: 0, background: DS.color.bgCard || "#fff" }}>
          <span style={{ fontSize: 12, color: DS.color.textSec }}>
            Menampilkan {from}–{to} dari <b style={{ color: DS.color.textPri }}>{total}</b> hasil
          </span>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <PagBtn label="←" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))} />
            <span style={{ fontSize: 12, color: DS.color.textSec, padding: "0 8px" }}>{page} / {totalPages}</span>
            <PagBtn label="→" disabled={page >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))} />
          </div>
        </div>
      </Card>
    </div>
  );
}
