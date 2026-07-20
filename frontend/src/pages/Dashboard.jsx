// =============================================================
// Dashboard.jsx — Halaman 1: Dashboard Overview
// Diadaptasi dari pagemuleradar/dashboard.jsx — layout/visual dipertahankan,
// SEMUA data diganti dari array hardcode jadi fetch API asli (dashboard.py).
// =============================================================
import { useEffect, useState } from "react";
import { DS, Card, SectionHeader, MonoText, TypologyBadge, RiskBar, StatusBadge } from "../design/system";
import {
  getDashboardOverview,
  getTypologyBreakdown,
  getStatusDistribution,
  getRiskTrend,
  getHeatmap,
  getTopAccounts,
  listAlerts,
} from "../lib/api";

const STATUS_COLORS = { NEW: "#FF4D4F", IN_REVIEW: "#FAAD14", CONFIRM: "#52C41A", FP: "#4D4D6A", CLOSED: "#4D4D6A" };
const TYPO_COLORS = ["#8B5CF6", "#F97316", "#64748B", "#0D9488", "#06B6D4", "#EAB308", "#D97706", "#94A3B8"];

const fmtIDR = (n) => {
  if (n >= 1e9) return `Rp ${(n / 1e9).toFixed(1)}M`;
  if (n >= 1e6) return `Rp ${(n / 1e6).toFixed(1)}Jt`;
  return `Rp ${n.toLocaleString("id-ID")}`;
};

const KPICard = ({ title, value, sub, accentColor }) => (
  <Card style={{ flex: "1 1 0", minWidth: 0, overflow: "hidden" }}>
    <div style={{ fontSize: 11, color: DS.color.textSec, marginBottom: 8, fontWeight: 500, textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
    <div style={{ fontSize: 32, fontWeight: 700, color: accentColor, lineHeight: 1, marginBottom: 6, letterSpacing: -1 }}>{value}</div>
    <div style={{ fontSize: 11, color: DS.color.textSec }}>{sub}</div>
  </Card>
);

// ── Bar Chart (typology breakdown) ──────────────────────────
const BarChart = ({ data }) => {
  if (!data || data.length === 0) return <div style={{ fontSize: 12, color: DS.color.textSec, padding: 20 }}>Belum ada data.</div>;
  const maxVal = Math.max(...data.map((d) => d.count), 1);
  const W = 580, H = 160;
  const padL = 32, padB = 42, padT = 20, padR = 12;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const barW = (chartW / data.length) * 0.55;
  const gap = chartW / data.length;
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * maxVal));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}>
      {yTicks.map((t) => {
        const y = padT + chartH - (t / maxVal) * chartH;
        return (
          <g key={t}>
            <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="3,3" />
            <text x={padL - 5} y={y + 4} textAnchor="end" fill={DS.color.textSec} fontSize="9">{t}</text>
          </g>
        );
      })}
      {data.map((d, i) => {
        const color = TYPO_COLORS[i % TYPO_COLORS.length];
        const bh = (d.count / maxVal) * chartH;
        const bx = padL + i * gap + (gap - barW) / 2;
        const by = padT + chartH - bh;
        return (
          <g key={d.typology}>
            <rect x={bx} y={by} width={barW} height={bh} fill={color} rx="3" opacity="0.85" />
            <text x={bx + barW / 2} y={by - 4} textAnchor="middle" fill={color} fontSize="9.5" fontWeight="700">{d.count}</text>
            <text x={bx + barW / 2} y={padT + chartH + 13} textAnchor="middle" fill={DS.color.textSec} fontSize="8.5">
              {d.typology.length > 10 ? d.typology.slice(0, 9) + "…" : d.typology}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

// ── Donut Chart (status distribution) ───────────────────────
const DonutChart = ({ data }) => {
  const total = data.reduce((s, d) => s + d.count, 0) || 1;
  const cx = 90, cy = 90, ro = 68, ri = 50;
  let cum = 0;
  const slices = data.map((d, i) => {
    const startAngle = (cum / total) * 2 * Math.PI - Math.PI / 2;
    cum += d.count;
    const endAngle = (cum / total) * 2 * Math.PI - Math.PI / 2;
    const x1 = cx + ro * Math.cos(startAngle), y1 = cy + ro * Math.sin(startAngle);
    const x2 = cx + ro * Math.cos(endAngle), y2 = cy + ro * Math.sin(endAngle);
    const xi1 = cx + ri * Math.cos(endAngle), yi1 = cy + ri * Math.sin(endAngle);
    const xi2 = cx + ri * Math.cos(startAngle), yi2 = cy + ri * Math.sin(startAngle);
    const lg = endAngle - startAngle > Math.PI ? 1 : 0;
    const color = STATUS_COLORS[d.status] || "#4D4D6A";
    return { d: `M${x1},${y1} A${ro},${ro} 0 ${lg} 1 ${x2},${y2} L${xi1},${yi1} A${ri},${ri} 0 ${lg} 0 ${xi2},${yi2} Z`, color, status: d.status, count: d.count };
  });
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
      <svg width="180" height="180" viewBox="0 0 180 180">
        {slices.map((s, i) => <path key={i} d={s.d} fill={s.color} opacity="0.9" />)}
        <text x={cx} y={cy - 8} textAnchor="middle" fill={DS.color.textPri} fontSize="22" fontWeight="700">{total}</text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill={DS.color.textSec} fontSize="10">Total Alert</text>
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {data.map((d) => (
          <div key={d.status} style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: STATUS_COLORS[d.status] || "#4D4D6A", flexShrink: 0 }}></span>
            <span style={{ fontSize: 11, color: DS.color.textSec, flex: 1 }}>{d.status}</span>
            <span style={{ fontSize: 12, color: DS.color.textPri, fontWeight: 600 }}>{d.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Heatmap ──────────────────────────────────────────────────
const DAY_LABELS = ["Min", "Sen", "Sel", "Rab", "Kam", "Jum", "Sab"];
const Heatmap = ({ data }) => {
  const grid = Array.from({ length: 7 }, () => Array(24).fill(0));
  let maxCount = 1;
  (data || []).forEach((r) => {
    if (r.day_of_week >= 0 && r.day_of_week <= 6 && r.hour >= 0 && r.hour <= 23) {
      grid[r.day_of_week][r.hour] = r.count;
      if (r.count > maxCount) maxCount = r.count;
    }
  });
  const cellW = 26, cellH = 10, padL = 38, padT = 18;
  const W = padL + 7 * cellW + 4;
  const H = padT + 24 * cellH + 4;
  const interpolateColor = (v) => {
    if (v < 0.3) return `rgba(79,142,247,${0.08 + v * 0.6})`;
    if (v < 0.6) return `rgba(250,140,22,${0.25 + v * 0.6})`;
    return `rgba(255,77,79,${0.45 + v * 0.55})`;
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}>
      {DAY_LABELS.map((d, di) => (
        <text key={d} x={padL + di * cellW + cellW / 2} y={padT - 4} textAnchor="middle" fill={DS.color.textSec} fontSize="8.5">{d}</text>
      ))}
      {Array.from({ length: 24 }, (_, h) => h).filter((h) => h % 2 === 0).map((h) => (
        <text key={h} x={padL - 4} y={padT + h * cellH + cellH / 2 + 3} textAnchor="end" fill={DS.color.textSec} fontSize="7.5">{h.toString().padStart(2, "0")}</text>
      ))}
      {grid.map((dayRow, di) =>
        dayRow.map((val, hi) => (
          <rect key={`${di}-${hi}`} x={padL + di * cellW + 1} y={padT + hi * cellH + 1} width={cellW - 2} height={cellH - 2} fill={interpolateColor(val / maxCount)} rx="1" />
        ))
      )}
    </svg>
  );
};

// ── Line Chart (risk trend) ──────────────────────────────────
const LineChart = ({ data }) => {
  const W = 400, H = 160;
  const padL = 28, padT = 12, padB = 28, padR = 12;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const maxVal = Math.max(...(data || []).map((d) => d.count), 1);
  const n = Math.max((data || []).length, 1);
  const toX = (i) => padL + (i / Math.max(n - 1, 1)) * chartW;
  const toY = (v) => padT + chartH - (v / maxVal) * chartH;
  const path = (data || []).map((d, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(d.count)}`).join(" ");
  const areaPath = data && data.length ? `M${toX(0)},${toY(0)} L${data.map((d, i) => `${toX(i)},${toY(d.count)}`).join(" L")} L${toX(n - 1)},${toY(0)} Z` : "";
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * maxVal));

  if (!data || data.length === 0) return <div style={{ fontSize: 12, color: DS.color.textSec, padding: 20 }}>Belum ada data trend.</div>;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}>
      <defs>
        <linearGradient id="area-trend" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FF3B30" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#FF3B30" stopOpacity="0.01" />
        </linearGradient>
      </defs>
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={padL} y1={toY(t)} x2={W - padR} y2={toY(t)} stroke={DS.color.border} strokeWidth="1" strokeDasharray="3,3" />
          <text x={padL - 4} y={toY(t) + 3} textAnchor="end" fill={DS.color.textSec} fontSize="8">{t}</text>
        </g>
      ))}
      <path d={areaPath} fill="url(#area-trend)" />
      <path d={path} fill="none" stroke="#FF3B30" strokeWidth="2.5" strokeLinejoin="round" />
      {data.map((d, i) => (
        <circle key={i} cx={toX(i)} cy={toY(d.count)} r="2.5" fill="#FF3B30" />
      ))}
    </svg>
  );
};

const TopAccountsTable = ({ rows, onInvestigate }) => (
  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
    <thead>
      <tr>
        {["#", "Rekening", "Typologi", "Risk Score", "Alert Count", "Aksi"].map((h) => (
          <th key={h} style={{ textAlign: "left", padding: "5px 8px", fontSize: 10, fontWeight: 700, color: DS.color.textPri, borderBottom: `1px solid ${DS.glass.thBorder}`, textTransform: "uppercase", letterSpacing: 0.4 }}>{h}</th>
        ))}
      </tr>
    </thead>
    <tbody>
      {rows.map((r, i) => (
        <tr key={r.account_id} style={{ background: i % 2 === 0 ? DS.glass.rowAlt : "transparent", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <td style={{ padding: "6px 8px", color: DS.color.textSec, fontSize: 11 }}>{i + 1}</td>
          <td style={{ padding: "6px 8px" }}><MonoText style={{ color: DS.color.blue }}>{r.account_id}</MonoText></td>
          <td style={{ padding: "6px 8px" }}><TypologyBadge type={r.typologies?.split(",")[0]?.trim()} small /></td>
          <td style={{ padding: "6px 8px" }}><RiskBar score={parseFloat(r.max_risk_score)} width={70} /></td>
          <td style={{ padding: "6px 8px", color: DS.color.textSec, fontSize: 11 }}>{r.alert_count}</td>
          <td style={{ padding: "6px 8px" }}>
            <button onClick={() => onInvestigate && onInvestigate(r.account_id)} style={{ background: `${DS.color.blue}22`, color: DS.color.blue, border: `1px solid ${DS.color.blue}44`, borderRadius: 4, padding: "3px 10px", fontSize: 10, fontWeight: 600, cursor: "pointer" }}>Investigasi</button>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
);

const RecentAlerts = ({ rows }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    {rows.map((a) => (
      <div key={a.alert_id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderRadius: 6, background: DS.glass.rowAlt, border: `1px solid ${DS.color.border}` }}>
        <TypologyBadge type={a.typology} small />
        <div style={{ flex: 1, minWidth: 0 }}>
          <MonoText style={{ color: DS.color.textPri, fontSize: 11 }}>{a.account_id}</MonoText>
          <div style={{ fontSize: 10, color: DS.color.textSec, marginTop: 1 }}>{new Date(a.created_at).toLocaleString("id-ID")}</div>
        </div>
        <StatusBadge status={a.status} />
      </div>
    ))}
  </div>
);

export default function Dashboard({ onInvestigate }) {
  const [overview, setOverview] = useState(null);
  const [typoData, setTypoData] = useState([]);
  const [statusData, setStatusData] = useState([]);
  const [trendData, setTrendData] = useState([]);
  const [heatmapData, setHeatmapData] = useState([]);
  const [topAccounts, setTopAccounts] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fix (QC 15-Jul): Promise.all sebelumnya all-or-nothing — 1 endpoint
    // gagal/lambat (mis. risk-trend) bikin SEMUA card kosong, padahal 6
    // endpoint lain berhasil. Promise.allSettled supaya tiap card independen:
    // yang berhasil tetap tampil, yang gagal cukup log error (state-nya
    // sudah default [] / null, komponen sudah handle itu dgn aman).
    Promise.allSettled([
      getDashboardOverview(),
      getTypologyBreakdown(),
      getStatusDistribution(),
      getRiskTrend(30),
      getHeatmap(),
      getTopAccounts(10),
      listAlerts({ limit: 8 }),
    ]).then(([ov, typo, status, trend, heat, top, alerts]) => {
      if (ov.status === "fulfilled") setOverview(ov.value);
      else console.error("Gagal load overview:", ov.reason);
      if (typo.status === "fulfilled") setTypoData(typo.value.items);
      else console.error("Gagal load typology breakdown:", typo.reason);
      if (status.status === "fulfilled") setStatusData(status.value.items);
      else console.error("Gagal load status distribution:", status.reason);
      if (trend.status === "fulfilled") setTrendData(trend.value.items);
      else console.error("Gagal load risk trend:", trend.reason);
      if (heat.status === "fulfilled") setHeatmapData(heat.value.items);
      else console.error("Gagal load heatmap:", heat.reason);
      if (top.status === "fulfilled") setTopAccounts(top.value.items);
      else console.error("Gagal load top accounts:", top.reason);
      if (alerts.status === "fulfilled") setRecentAlerts(alerts.value.items);
      else console.error("Gagal load recent alerts:", alerts.reason);
      setLoading(false);
    });
  }, []);

  const content = { flex: 1, overflowY: "auto", padding: 20, display: "flex", flexDirection: "column", gap: 16 };
  const row = { display: "flex", gap: 16 };

  if (loading) {
    return <div style={{ ...content, alignItems: "center", justifyContent: "center", color: DS.color.textSec }}>Memuat dashboard…</div>;
  }

  return (
    <div style={content}>
      <div style={row}>
        <KPICard title="Alert Aktif" value={overview?.alert_aktif ?? "—"} sub={`${overview?.total_alert_sepanjang_waktu ?? 0} total sepanjang waktu`} accentColor={DS.color.riskHigh} />
        <KPICard title="Cluster Aktif" value={overview?.cluster_aktif ?? "—"} sub="terdeteksi dari graph" accentColor={DS.color.riskMed} />
        <KPICard title="Dana Berisiko" value={overview ? fmtIDR(overview.dana_berisiko_idr) : "—"} sub="alert NEW/IN_REVIEW" accentColor={DS.color.riskHigh} />
        <KPICard title="Kasus Pending" value={overview?.kasus_pending ?? "—"} sub="belum ditutup" accentColor={DS.color.blue} />
      </div>

      <div style={{ ...row, alignItems: "stretch" }}>
        <Card style={{ flex: 3 }}>
          <SectionHeader title="Typology Breakdown" />
          <BarChart data={typoData} />
        </Card>
        <Card style={{ flex: 2 }}>
          <SectionHeader title="Status Distribution" />
          <DonutChart data={statusData} />
        </Card>
      </div>

      <div style={{ ...row, alignItems: "stretch" }}>
        <Card style={{ flex: 1 }}>
          <SectionHeader title="Alert Activity (Mingguan)" />
          <Heatmap data={heatmapData} />
        </Card>
        <Card style={{ flex: 1 }}>
          <SectionHeader title="Risk Trend 30 Hari" />
          <LineChart data={trendData} />
        </Card>
      </div>

      <div style={{ ...row, alignItems: "stretch" }}>
        <Card style={{ flex: 3 }}>
          <SectionHeader title="Top 10 Rekening Berisiko" />
          <TopAccountsTable rows={topAccounts} onInvestigate={onInvestigate} />
        </Card>
        <Card style={{ flex: 2 }}>
          <SectionHeader title="Alert Terbaru" />
          <RecentAlerts rows={recentAlerts} />
        </Card>
      </div>
    </div>
  );
}
