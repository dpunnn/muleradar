// =============================================================
// CaseDetail.jsx — Halaman 4: Case Detail + LLM Copilot + Modal LTKM
// Diadaptasi dari pagemuleradar/case-detail.jsx — layout dipertahankan,
// data disambungkan ke cases.py + copilot.py (data & narasi ASLI).
// =============================================================
import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { jsPDF } from "jspdf";
import { DS, Icons, Card, SectionHeader, StatusBadge, TypologyBadge, MonoText, getInputStyle } from "../design/system";
import { getCaseDetail, updateCase, escalateCase, generateSummary, generateLtkm, getNodeNeighbors } from "../lib/api";

const fmtIDR = (n) => `Rp ${Number(n || 0).toLocaleString("id-ID")}`;
const fmtDate = (d) => (d ? new Date(d).toLocaleString("id-ID") : "—");

// Fix (QC 15-Jul): sebelumnya object literal module-level — DS.color.*
// (Proxy dinamis) dibaca SEKALI saat import, nilainya ke-freeze ke tema
// yang aktif waktu itu. Jadi fungsi supaya dibaca ulang tiap render →
// warna ikut berubah saat user toggle dark/light.
const getAuditLabel = (eventType) =>
  ({
    ALERT_CREATED: { icon: "●", color: DS.color.riskHigh },
    CASE_CREATED: { icon: "●", color: DS.color.blue },
    ASSIGNED: { icon: "●", color: DS.color.blue },
    STATUS_CHANGED: { icon: "●", color: "#FAAD14" },
    NOTE_ADDED: { icon: "●", color: DS.color.textSec },
    ESCALATED: { icon: "●", color: DS.color.riskMed },
  }[eventType] || { icon: "●", color: DS.color.textSec });

// ── Mini graph — tetangga 1-hop ASLI dari Neo4j (bukan node acak) ──
const MiniGraph = ({ accountId }) => {
  const [neighbors, setNeighbors] = useState(null);

  useEffect(() => {
    if (!accountId) return;
    getNodeNeighbors(accountId, 1, 12)
      .then((d) => setNeighbors(d.neighbors || []))
      .catch(() => setNeighbors([]));
  }, [accountId]);

  if (neighbors === null) {
    return <div style={{ background: "#080B12", borderRadius: 6, height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: DS.color.textSec, fontSize: 11 }}>Memuat graph…</div>;
  }
  if (neighbors.length === 0) {
    return <div style={{ background: "#080B12", borderRadius: 6, height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: DS.color.textSec, fontSize: 11 }}>Tidak ada tetangga terdeteksi di graph untuk akun ini.</div>;
  }

  const n = Math.min(neighbors.length, 10);
  const nodes = neighbors.slice(0, n).map((nb, i) => {
    const angle = (i / n) * Math.PI * 2;
    const risky = (nb.risk_score || 0) >= 0.5;
    return { x: 130 + Math.cos(angle) * 60, y: 90 + Math.sin(angle) * 55, r: 7, risky, id: nb.account_id };
  });
  return (
    <div style={{ background: "#080B12", borderRadius: 6, overflow: "hidden", height: 180 }}>
      <svg viewBox="0 0 260 180" style={{ width: "100%", height: "100%" }}>
        {nodes.map((nd, i) => (
          <line key={i} x1={130} y1={90} x2={nd.x} y2={nd.y} stroke="#2D3250" strokeWidth="1.5" />
        ))}
        <circle cx={130} cy={90} r={14} fill="#141c30" stroke="#FF4D4F" strokeWidth={2.5} />
        <circle cx={130} cy={90} r={5} fill="#FF4D4F" opacity="0.8" />
        {nodes.map((nd, i) => (
          <g key={i}>
            <circle cx={nd.x} cy={nd.y} r={nd.r} fill="#141c30" stroke={nd.risky ? "#FF4D4F" : "#FAAD14"} strokeWidth={1.8} />
            <circle cx={nd.x} cy={nd.y} r={nd.r * 0.35} fill={nd.risky ? "#FF4D4F" : "#FAAD14"} opacity="0.7" />
          </g>
        ))}
      </svg>
    </div>
  );
};

const AuditTrail = ({ events }) => {
  if (!events || events.length === 0) {
    return <div style={{ fontSize: 11, color: DS.color.textSec }}>Belum ada riwayat.</div>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {events.map((ev) => {
        const meta = getAuditLabel(ev.event_type);
        return (
          <div key={ev.log_id} style={{ display: "flex", gap: 8 }}>
            <span style={{ color: meta.color, fontSize: 10, lineHeight: "16px" }}>{meta.icon}</span>
            <div>
              <div style={{ fontSize: 10, color: DS.color.textSec }}>{fmtDate(ev.created_at)}</div>
              <div style={{ fontSize: 11.5, color: DS.color.textPri }}>{ev.detail}</div>
              <div style={{ fontSize: 10, color: DS.color.textSec }}>{ev.actor}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

const FlaggedTxs = ({ flags }) => {
  const [tab, setTab] = useState("Semua");
  const flagTypes = ["Semua", ...new Set(flags.map((f) => f.flag_type))];
  const filtered = tab === "Semua" ? flags : flags.filter((f) => f.flag_type === tab);

  if (flags.length === 0) {
    return <div style={{ fontSize: 12, color: DS.color.textSec, padding: 12 }}>Tidak ada transaksi yang di-flag untuk akun ini.</div>;
  }

  return (
    <Card style={{ flex: 1 }} pad={14}>
      <div style={{ display: "flex", gap: 6, marginBottom: 12, overflowX: "auto" }}>
        {flagTypes.map((t) => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: "4px 10px", borderRadius: 5, fontSize: 10, cursor: "pointer", whiteSpace: "nowrap", background: tab === t ? "rgba(107,174,255,0.2)" : "rgba(255,255,255,0.05)", border: `1px solid ${tab === t ? "rgba(107,174,255,0.4)" : "rgba(255,255,255,0.1)"}`, color: tab === t ? DS.color.blue : DS.color.textSec, fontWeight: tab === t ? 600 : 400 }}>
            {t}
          </button>
        ))}
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${DS.color.border}` }}>
            {["Flag", "Detail", "Severity"].map((h) => (
              <th key={h} style={{ padding: "5px 6px", textAlign: "left", fontSize: 9.5, color: DS.color.textSec, fontWeight: 600, textTransform: "uppercase" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map((f, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${DS.glass.panelBorder}50` }}>
              <td style={{ padding: "6px 6px" }}>
                <span style={{ fontSize: 9.5, padding: "1px 6px", borderRadius: 3, background: `${DS.color.riskHigh}22`, color: DS.color.riskHigh, fontWeight: 600 }}>{f.flag_type}</span>
              </td>
              <td style={{ padding: "6px 6px", color: DS.color.textSec }}>{f.detail}</td>
              <td style={{ padding: "6px 6px", color: DS.color.textSec }}>{f.severity || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
};

const AICopilot = ({ accountId, onDraftLtkm }) => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  const generate = () => {
    setLoading(true);
    generateSummary(accountId)
      .then((d) => setSummary(d.summary))
      .catch((e) => console.error("Gagal generate summary:", e))
      .finally(() => setLoading(false));
  };

  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <div style={{ width: 24, height: 24, borderRadius: 6, background: "linear-gradient(135deg, #4F8EF7, #7C3AED)", display: "flex", alignItems: "center", justifyContent: "center" }}>{Icons.spark(11)}</div>
        <span style={{ fontSize: 13, fontWeight: 600, color: DS.color.textPri }}>AI Copilot</span>
        <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 3, background: "#7C3AED22", color: "#9F6FED", fontWeight: 600 }}>BETA</span>
      </div>

      {!summary ? (
        <button onClick={generate} disabled={loading} style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, background: `${DS.color.blue}18`, color: DS.color.blue, border: `1px solid ${DS.color.blue}44`, borderRadius: 6, padding: 9, fontSize: 12, fontWeight: 600, cursor: loading ? "default" : "pointer" }}>
          {loading ? <span style={{ fontSize: 11, color: DS.color.textSec }}>Generating…</span> : <><span>{Icons.spark(12)}</span> Generate Case Summary</>}
        </button>
      ) : (
        <div>
          <div style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, padding: 10, fontSize: 11, lineHeight: 1.65, color: DS.color.textSec, marginBottom: 8 }}>{summary}</div>
          <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
            <button onClick={() => setSummary(null)} style={{ fontSize: 10, padding: "4px 10px", borderRadius: 4, background: "transparent", color: DS.color.textSec, border: `1px solid ${DS.color.border}`, cursor: "pointer" }}>Regenerate</button>
          </div>
        </div>
      )}

      <button onClick={onDraftLtkm} style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6, background: `${DS.color.riskLow}18`, color: DS.color.riskLow, border: `1px solid ${DS.color.riskLow}44`, borderRadius: 6, padding: 9, fontSize: 12, fontWeight: 600, cursor: "pointer", marginTop: 8 }}>
        <span>{Icons.doc(12)}</span> Draft LTKM
      </button>
    </Card>
  );
};

const LtkmModal = ({ accountId, onClose }) => {
  const [ltkm, setLtkm] = useState(null);
  const [loading, setLoading] = useState(true);
  const textareaRef = useRef(null);

  useEffect(() => {
    generateLtkm(accountId)
      .then((d) => setLtkm(d))
      .catch((e) => console.error("Gagal generate LTKM:", e))
      .finally(() => setLoading(false));
  }, [accountId]);

  // Fix (item 6.6): PDF di-generate CLIENT-SIDE (jsPDF) dari isi textarea
  // SAAT INI — bukan ltkm.ltkm_draft mentah — supaya konsisten dgn "Salin
  // Teks" (keduanya ambil versi yg mungkin sudah diedit analis, bukan draft
  // asli sebelum revisi manual).
  const handleDownloadPdf = () => {
    const text = textareaRef.current?.value || "";
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const marginX = 48;
    let y = 56;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text(`Draft LTKM — ${accountId}`, marginX, y);
    y += 22;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(140, 90, 0);
    doc.text(
      "Dokumen ini adalah draft otomatis. Verifikasi seluruh informasi sebelum submit ke PPATK.",
      marginX, y, { maxWidth: 500 }
    );
    y += 26;

    doc.setTextColor(20, 20, 20);
    doc.setFontSize(10.5);
    const lines = doc.splitTextToSize(text, 500);
    const pageBottom = doc.internal.pageSize.getHeight() - 48;
    for (const line of lines) {
      if (y > pageBottom) {
        doc.addPage();
        y = 56;
      }
      doc.text(line, marginX, y);
      y += 14;
    }

    doc.save(`LTKM-${accountId}.pdf`);
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.70)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: 24 }} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={{ background: "#0d1829", border: "1px solid rgba(255,255,255,0.13)", borderRadius: 12, width: 700, maxWidth: "95vw", maxHeight: "90vh", display: "flex", flexDirection: "column", overflow: "hidden", boxShadow: "0 24px 80px rgba(0,0,0,0.7)" }}>
        <div style={{ padding: "16px 22px", borderBottom: "1px solid rgba(255,255,255,0.10)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h2 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "#fff" }}>Draft LTKM — {accountId}</h2>
          <button onClick={onClose} style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.13)", borderRadius: 7, color: "rgba(255,255,255,0.6)", cursor: "pointer", padding: "6px 8px" }}>{Icons.close(12)}</button>
        </div>
        <div style={{ flex: 1, overflowY: "auto", padding: "18px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 14px", borderRadius: 7, marginBottom: 18, background: "rgba(255,149,0,0.10)", border: "1px solid rgba(255,149,0,0.28)" }}>
            <span style={{ fontSize: 14, color: "#FF9500" }}>⚠</span>
            <span style={{ fontSize: 11, color: "rgba(255,149,0,0.9)" }}>Dokumen ini adalah draft otomatis. Verifikasi seluruh informasi sebelum submit ke PPATK.</span>
          </div>
          {loading ? (
            <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>Menghasilkan narasi LTKM…</div>
          ) : (
            <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.45)", marginBottom: 8 }}>Uraian ({ltkm?.provider === "template" ? "template rule-based" : ltkm?.provider}):</div>
              <textarea ref={textareaRef} defaultValue={ltkm?.ltkm_draft} style={{ width: "100%", minHeight: 140, padding: "10px 12px", boxSizing: "border-box", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 7, color: "#fff", fontSize: 12, lineHeight: 1.75, resize: "vertical", fontFamily: "'Inter',sans-serif" }} />
            </div>
          )}
        </div>
        <div style={{ padding: "12px 22px", borderTop: "1px solid rgba(255,255,255,0.10)", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(0,0,0,0.2)" }}>
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>{ltkm?.note}</span>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => navigator.clipboard?.writeText(textareaRef.current?.value || "")} style={{ background: "transparent", color: "rgba(255,255,255,0.7)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 7, padding: "8px 16px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Salin Teks</button>
            <button
              onClick={handleDownloadPdf}
              disabled={loading || !ltkm}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                background: DS.color.riskLow, color: "#04140c",
                border: "none", borderRadius: 7, padding: "8px 16px",
                fontSize: 12, fontWeight: 700,
                cursor: loading || !ltkm ? "not-allowed" : "pointer",
                opacity: loading || !ltkm ? 0.5 : 1,
              }}
            >
              {Icons.download(12)} Download PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function CaseDetail({ onCaseChanged }) {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [ltkmOpen, setLtkmOpen] = useState(false);

  const load = () => {
    setLoading(true);
    setNotFound(false);
    setLoadError(false);
    getCaseDetail(caseId)
      .then((d) => { setData(d); setNote(d.case.notes || ""); })
      .catch((e) => {
        console.error("Gagal load case detail:", e);
        // Fix (QC 15-Jul): sebelumnya semua error (404 asli ATAU network/
        // timeout) tampil sbg "Case tidak ditemukan" yg sama — kalau backend
        // lambat/down saat demo, juri lihat pesan yg salah (kesannya case-nya
        // memang tak ada, padahal cuma koneksi gagal).
        if (e?.response?.status === 404) setNotFound(true);
        else setLoadError(true);
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [caseId]);

  const saveNote = () => {
    updateCase(caseId, { notes: note }).then(load).catch((e) => console.error(e));
  };

  const setStatus = (status) => {
    updateCase(caseId, { status }).then(() => { load(); onCaseChanged && onCaseChanged(); }).catch((e) => console.error(e));
  };

  const escalate = () => {
    escalateCase(caseId, note || null)
      .then(load)
      .catch((e) => { console.error("Gagal eskalasi:", e); alert("Gagal mengeskalasi case. Coba lagi."); });
  };

  if (loading) return <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: DS.color.textSec }}>Memuat case…</div>;
  if (loadError) return <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: DS.color.riskHigh }}>Gagal memuat case — periksa koneksi ke server, lalu coba lagi.</div>;
  if (notFound || !data) return <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: DS.color.textSec }}>Case tidak ditemukan.</div>;

  const { case: c, alert, transaction_flags, institusi, cluster, dana_terlibat_idr, audit_trail } = data;
  const accountId = alert?.account_id;

  return (
    <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "14px 20px", borderBottom: `1px solid ${DS.glass.thBorder}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontSize: 11, color: DS.color.textSec, marginBottom: 4 }}>
            <span onClick={() => navigate("/alerts")} style={{ cursor: "pointer", color: DS.color.blue }}>Alerts</span>
            <span style={{ margin: "0 6px" }}>{Icons.chevronRight(10)}</span>
            <span>{alert?.alert_id}</span>
            <span style={{ margin: "0 6px" }}>{Icons.chevronRight(10)}</span>
            <span>Case Detail</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <h1 style={{ fontSize: 20, fontWeight: 700, color: DS.color.textPri, margin: 0 }}>Case #{c.case_id}</h1>
            <StatusBadge status={c.status} />
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={escalate} style={{ background: `${DS.color.riskMed}18`, color: DS.color.riskMed, border: `1px solid ${DS.color.riskMed}44`, borderRadius: 6, padding: "6px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Eskalasi</button>
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", gap: 0, overflow: "hidden", minHeight: 0 }}>
        <div style={{ width: 276, flexShrink: 0, borderRight: `1px solid ${DS.glass.panelBorder}`, overflowY: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
          <Card pad={12}>
            <div style={{ fontSize: 10, fontWeight: 600, color: DS.color.textSec, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10 }}>Informasi Case</div>
            {[
              { label: "Rekening Utama", value: accountId, mono: true, color: DS.color.blue },
              { label: "Institusi", value: institusi || "tidak diketahui" },
              { label: "Typologi", badge: alert?.typology },
              { label: "Risk Score", value: alert ? parseFloat(alert.risk_score).toFixed(4) : "—", color: DS.color.riskHigh },
              { label: "Cluster", value: cluster ? `${cluster.size} rekening (${cluster.risk_level})` : "belum terdeteksi" },
              { label: "Dana Terlibat", value: fmtIDR(dana_terlibat_idr) },
              { label: "Assigned To", value: c.assigned_to || "belum di-assign" },
              { label: "Dibuat", value: fmtDate(alert?.created_at) },
              { label: "Diupdate", value: fmtDate(c.updated_at) },
            ].map((r, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, paddingBottom: 6, borderBottom: `1px solid ${DS.color.border}30` }}>
                <span style={{ fontSize: 10, color: DS.color.textSec }}>{r.label}</span>
                {r.badge ? <TypologyBadge type={r.badge} small /> : <span style={{ fontSize: 11, color: r.color || DS.color.textPri, fontFamily: r.mono ? "'JetBrains Mono', monospace" : "inherit", fontWeight: r.color ? 700 : 400 }}>{r.value}</span>}
              </div>
            ))}
          </Card>

          <Card pad={12}>
            <SectionHeader title="Timeline Investigasi" />
            <AuditTrail events={audit_trail} />
          </Card>

          <Card pad={12}>
            <SectionHeader title="Catatan Investigasi" />
            <textarea value={note} onChange={(e) => setNote(e.target.value)} style={{ width: "100%", minHeight: 80, background: DS.color.bgMain, border: `1px solid ${DS.color.border}`, borderRadius: 5, color: DS.color.textPri, fontSize: 11, padding: 8, resize: "vertical", lineHeight: 1.5, fontFamily: "inherit", boxSizing: "border-box" }} />
            <button onClick={saveNote} style={{ marginTop: 8, width: "100%", background: DS.color.blue, color: "#fff", border: "none", borderRadius: 5, padding: 7, fontSize: 11, fontWeight: 600, cursor: "pointer" }}>Simpan Catatan</button>
          </Card>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <Card pad={12}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: DS.color.textPri }}>Graph Cluster</span>
              <button onClick={() => navigate("/graph")} style={{ fontSize: 11, color: DS.color.blue, background: "none", border: "none", cursor: "pointer" }}>Buka di Graph Explorer →</button>
            </div>
            <MiniGraph accountId={accountId} />
          </Card>
          <FlaggedTxs flags={transaction_flags || []} />
        </div>

        <div style={{ width: 296, flexShrink: 0, borderLeft: `1px solid ${DS.glass.panelBorder}`, overflowY: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
          {accountId && <AICopilot accountId={accountId} onDraftLtkm={() => setLtkmOpen(true)} />}
        </div>
      </div>

      <div style={{ padding: "12px 20px", borderTop: `1px solid ${DS.glass.panelBorder}`, display: "flex", gap: 8, background: DS.glass.bgSticky }}>
        <button onClick={() => setStatus("CONFIRM")} style={{ background: DS.color.riskHigh, color: "#fff", border: "none", borderRadius: 6, padding: "9px 20px", fontSize: 13, fontWeight: 700, cursor: "pointer" }}>Confirm Fraud</button>
        <button onClick={() => setStatus("FP")} style={{ background: "transparent", color: DS.color.textSec, border: `1px solid ${DS.color.border}`, borderRadius: 6, padding: "9px 16px", fontSize: 13, cursor: "pointer" }}>Mark False Positive</button>
      </div>

      {ltkmOpen && accountId && <LtkmModal accountId={accountId} onClose={() => setLtkmOpen(false)} />}
    </div>
  );
}
