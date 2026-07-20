// =============================================================
// CaseList.jsx — Halaman Daftar Case (investigasi yang sudah dibuka)
// Layout & style disamakan dgn AlertList.jsx, data dari cases.py
// (bukan alerts.py — case punya status/assigned_to sendiri, terpisah
// dari status alert asalnya).
// =============================================================
import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { DS, Icons, Card, MonoText, TypologyBadge, RiskBar, StatusBadge, getInputStyle } from "../design/system";
import { listCases } from "../lib/api";

const STATUS_OPTIONS = ["Semua", "NEW", "IN_REVIEW", "CONFIRM", "FP", "CLOSED"];
const PAGE_SIZE = 16;

const FilterBar = ({ status, setStatus, onApply, onReset }) => (
  <Card style={{ marginBottom: 12 }} pad={12}>
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
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

export default function CaseList() {
  const navigate = useNavigate();
  const [status, setStatus] = useState("Semua");
  const [appliedStatus, setAppliedStatus] = useState("Semua");
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [hover, setHover] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    const params = { limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE };
    if (appliedStatus !== "Semua") params.status = appliedStatus;
    listCases(params)
      .then((d) => { setRows(d.items); setTotal(d.total); })
      .catch((e) => console.error("Gagal load cases:", e))
      .finally(() => setLoading(false));
  }, [page, appliedStatus]);

  useEffect(() => { load(); }, [load]);

  const applyFilter = () => { setAppliedStatus(status); setPage(1); };
  const resetFilter = () => { setStatus("Semua"); setAppliedStatus("Semua"); setPage(1); };

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const from = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const to = Math.min(page * PAGE_SIZE, total);

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: DS.color.textPri, margin: 0 }}>Daftar Case</h1>
          <p style={{ fontSize: 12, color: DS.color.textSec, margin: "3px 0 0" }}>{total.toLocaleString("id-ID")} case ditemukan</p>
        </div>
        <button onClick={load} style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(107,174,255,0.12)", color: DS.color.blue, border: `1px solid ${DS.color.blue}44`, borderRadius: 6, padding: "7px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
          <span>{Icons.refresh(12)}</span> Refresh
        </button>
      </div>

      <FilterBar status={status} setStatus={setStatus} onApply={applyFilter} onReset={resetFilter} />

      <Card pad={0} style={{ overflow: "hidden" }}>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${DS.glass.thBorder}` }}>
                {["Case ID", "Rekening", "Typologi", "Risk Score", "Status", "Assigned To", "Dibuat", "Aksi"].map((h) => (
                  <th key={h} style={{ padding: "10px 8px", textAlign: "left", fontSize: 10, fontWeight: 700, color: DS.color.textPri, textTransform: "uppercase", letterSpacing: 0.4, whiteSpace: "nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={8} style={{ padding: 20, textAlign: "center", color: DS.color.textSec }}>Memuat…</td></tr>
              ) : rows.length === 0 ? (
                <tr><td colSpan={8} style={{ padding: 20, textAlign: "center", color: DS.color.textSec }}>Tidak ada case yang cocok.</td></tr>
              ) : rows.map((row, i) => {
                const isHover = hover === row.case_id;
                return (
                  <tr
                    key={row.case_id}
                    onMouseEnter={() => setHover(row.case_id)}
                    onMouseLeave={() => setHover(null)}
                    style={{ background: isHover ? DS.glass.rowHover : i % 2 === 0 ? "transparent" : DS.glass.rowAlt, borderBottom: `1px solid ${DS.glass.panelBorder}50` }}
                  >
                    <td style={{ padding: "8px 8px" }}><MonoText style={{ color: DS.color.textSec, fontSize: 11 }}>#{row.case_id}</MonoText></td>
                    <td style={{ padding: "8px 8px" }}><MonoText style={{ color: DS.color.blue }}>{row.account_id}</MonoText></td>
                    <td style={{ padding: "8px 8px" }}><TypologyBadge type={row.typology} small /></td>
                    <td style={{ padding: "8px 8px" }}><RiskBar score={parseFloat(row.risk_score)} width={72} /></td>
                    <td style={{ padding: "8px 8px" }}><StatusBadge status={row.status} /></td>
                    <td style={{ padding: "8px 8px", color: DS.color.textSec, fontSize: 11 }}>{row.assigned_to || "—"}</td>
                    <td style={{ padding: "8px 8px", color: DS.color.textSec, fontSize: 11, whiteSpace: "nowrap" }}>{new Date(row.created_at).toLocaleString("id-ID")}</td>
                    <td style={{ padding: "8px 8px" }}>
                      <button
                        onClick={() => navigate(`/cases/${row.case_id}`)}
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
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", borderTop: `1px solid ${DS.color.border}` }}>
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
