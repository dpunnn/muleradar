// =============================================================
// GraphExplorer.jsx — Halaman 3: Graph Explorer
// Diadaptasi dari pagemuleradar/graph.jsx — visual dipertahankan, TAPI
// posisi node yg dulu HARDCODE (polar layout demo) diganti d3-force
// (layout otomatis) krn data asli (cluster Neo4j) ukurannya variabel,
// tidak bisa di-hardcode posisinya.
// =============================================================
import { useEffect, useMemo, useRef, useState } from "react";
import { forceSimulation, forceManyBody, forceLink, forceCenter, forceCollide } from "d3-force";
import { DS, Icons, RiskBadge, TypologyBadge, MonoText, getInputStyle, riskScoreToLevel } from "../design/system";
import { listClusters, getClusterDetail, getNodePPR, getNodeFlags } from "../lib/api";

const riskCol = (risk) => ({ high: "#FF4D4F", medium: "#FAAD14", low: "#52C41A" }[risk] || "#8C8C8C");
const bankFill = () => "#141c30";
const amtThick = (amt) => Math.max(1, Math.min(4.5, ((amt || 0) / 2_000_000) * 3.5));

// ── Layout: d3-force, dijalankan sinkron N tick lalu dipakai statis ──
function computeLayout(nodes, edges, width = 1000, height = 590) {
  const simNodes = nodes.map((n) => ({ ...n }));
  const idIndex = new Map(simNodes.map((n, i) => [n.id, i]));
  const simLinks = edges
    .filter((e) => idIndex.has(e.src) && idIndex.has(e.dst))
    .map((e) => ({ source: e.src, target: e.dst, amount: e.amount, is_laundering: e.is_laundering }));

  const sim = forceSimulation(simNodes)
    .force("charge", forceManyBody().strength(-140))
    .force("link", forceLink(simLinks).id((d) => d.id).distance(70))
    .force("center", forceCenter(width / 2, height / 2))
    .force("collide", forceCollide().radius((d) => (d.r || 10) + 6))
    .stop();

  // Fix (QC 15-Jul): 220 tick sinkron di main thread bikin UI freeze terasa
  // (100-300ms) saat ganti cluster besar. 120 tick masih cukup konvergen
  // utk cluster kecil-menengah yg kita render (MAX_RENDER 120 node).
  for (let i = 0; i < 120; i++) sim.tick();

  return { nodes: simNodes, links: simLinks };
}

const GraphLegend = () => (
  <div style={{ position: "absolute", bottom: 16, left: 16, background: "rgba(8,12,30,0.75)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, padding: "8px 12px", display: "flex", flexDirection: "column", gap: 5 }}>
    {[
      { label: "High Risk", color: "#FF4D4F" },
      { label: "Medium", color: "#FAAD14" },
      { label: "Low", color: "#52C41A" },
    ].map((item) => (
      <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: item.color, flexShrink: 0 }}></div>
        <span style={{ fontSize: 10, color: DS.color.textSec }}>{item.label}</span>
      </div>
    ))}
    <div style={{ marginTop: 3, paddingTop: 5, borderTop: `1px solid ${DS.color.border}` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <div style={{ width: 18, height: 2, background: "#3a3f5a", flexShrink: 0 }}></div>
        <span style={{ fontSize: 10, color: DS.color.textSec }}>Normal</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 4 }}>
        <div style={{ width: 18, height: 2, borderTop: "1px dashed #FF4D4F", flexShrink: 0 }}></div>
        <span style={{ fontSize: 10, color: DS.color.textSec }}>Suspicious</span>
      </div>
    </div>
  </div>
);

const GraphSVG = ({ layout, selected, onSelect, showLabels }) => {
  if (!layout || layout.nodes.length === 0) {
    return <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: DS.color.textSec, fontSize: 12 }}>Pilih cluster di panel kiri untuk melihat graph.</div>;
  }
  const { nodes, links } = layout;
  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  return (
    <svg viewBox="0 0 1000 590" style={{ width: "100%", height: "100%", cursor: "crosshair" }} onClick={(e) => { if (e.target === e.currentTarget) onSelect(null); }}>
      <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1a1e30" strokeWidth="0.5" />
        </pattern>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#4D5070" />
        </marker>
        <marker id="arrow-red" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#FF4D4F88" />
        </marker>
      </defs>
      <rect width="1000" height="590" fill="url(#grid)" />

      {links.map((e, i) => {
        const fn = nodeById.get(typeof e.source === "object" ? e.source.id : e.source);
        const tn = nodeById.get(typeof e.target === "object" ? e.target.id : e.target);
        if (!fn || !tn) return null;
        const dx = tn.x - fn.x, dy = tn.y - fn.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const ex = tn.x - (dx / len) * ((tn.r || 10) + 5);
        const ey = tn.y - (dy / len) * ((tn.r || 10) + 5);
        const sx = fn.x + (dx / len) * ((fn.r || 10) + 3);
        const sy = fn.y + (dy / len) * ((fn.r || 10) + 3);
        const sus = !!e.is_laundering;
        return (
          <line key={i} x1={sx} y1={sy} x2={ex} y2={ey} stroke={sus ? "#FF4D4F55" : "#3a3f5a"} strokeWidth={amtThick(e.amount)} strokeDasharray={sus ? "5,4" : "none"} markerEnd={sus ? "url(#arrow-red)" : "url(#arrow)"} opacity={sus ? 0.8 : 0.6} />
        );
      })}

      {nodes.map((n) => {
        const isSel = selected === n.id;
        const strokeC = riskCol(n.risk);
        return (
          <g key={n.id} style={{ cursor: "pointer" }} onClick={(e) => { e.stopPropagation(); onSelect(n.id); }}>
            {n.risk === "high" && <circle cx={n.x} cy={n.y} r={(n.r || 10) + 7} fill="none" stroke={riskCol("high")} strokeWidth="1" opacity="0.2" />}
            {isSel && <circle cx={n.x} cy={n.y} r={(n.r || 10) + 10} fill="none" stroke="#ffffff" strokeWidth="1.5" opacity="0.35" />}
            <circle cx={n.x} cy={n.y} r={n.r || 10} fill={bankFill()} stroke={isSel ? "#ffffff" : strokeC} strokeWidth={isSel ? 2.5 : n.isCollector ? 2.5 : 1.8} opacity={0.95} />
            <circle cx={n.x} cy={n.y} r={(n.r || 10) * 0.35} fill={strokeC} opacity={0.7} />
            {(showLabels || isSel || n.isCollector) && (
              <text x={n.x} y={n.y + (n.r || 10) + 10} textAnchor="middle" fontSize={n.isCollector ? 9 : 8} fill={isSel ? "#ffffff" : "#A0A8C0"} fontFamily="'JetBrains Mono', monospace" style={{ pointerEvents: "none" }}>
                {n.id.slice(-7)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

const Section = ({ title, children }) => (
  <div style={{ marginBottom: 16 }}>
    <div style={{ fontSize: 10, color: DS.color.textSec, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 8, paddingBottom: 4, borderBottom: `1px solid ${DS.glass.thBorder}` }}>{title}</div>
    {children}
  </div>
);
const Row = ({ label, value, valueStyle }) => (
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
    <span style={{ fontSize: 11, color: DS.color.textSec }}>{label}</span>
    <span style={{ fontSize: 11, color: DS.color.textPri, ...valueStyle }}>{value}</span>
  </div>
);

const NodeDetail = ({ nodeId, node, onClose, onOpenCase }) => {
  const [ppr, setPpr] = useState(null);
  const [flags, setFlags] = useState(null);

  useEffect(() => {
    if (!nodeId) return;
    // Fix (QC 15-Jul): tanpa guard ini, klik node A lalu node B dgn cepat
    // bisa bikin response A tiba SETELAH response B (network tak berurutan)
    // dan menimpa state dgn data node yg salah. `current` menjaga hasil
    // cuma dipakai kalau nodeId masih sama saat promise resolve.
    let current = true;
    setPpr(null);
    setFlags(null);
    // Fix (20-Jul, scan produksi): _error flag membedakan "berhasil tapi
    // kosong" dari "GAGAL" — dulu keduanya tampil identik (kosong), analis
    // bisa salah simpul rekening bersih padahal API-nya yg gagal.
    getNodePPR(nodeId).then((d) => { if (current) setPpr(d); }).catch(() => { if (current) setPpr({ scores: {}, _error: true }); });
    getNodeFlags(nodeId).then((d) => { if (current) setFlags(d); }).catch(() => { if (current) setFlags({ flags: [], _error: true }); });
    return () => { current = false; };
  }, [nodeId]);

  if (!nodeId || !node) {
    return (
      <div style={{ flex: "0 0 300px", background: DS.glass.bgPanel, backdropFilter: DS.glass.blurPanel, borderLeft: `1px solid ${DS.glass.panelBorder}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", color: DS.color.textSec }}>
          <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.3 }}>{Icons.network(28)}</div>
          <p style={{ fontSize: 12 }}>Klik node untuk melihat detail</p>
        </div>
      </div>
    );
  }

  const pprScore = ppr?.scores?.[nodeId] ?? node.ppr ?? 0;

  return (
    <div style={{ flex: "0 0 300px", background: DS.glass.bgPanel, backdropFilter: DS.glass.blurPanel, borderLeft: `1px solid ${DS.glass.panelBorder}`, overflowY: "auto", padding: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
        <div>
          <MonoText style={{ fontSize: 13, color: DS.color.textPri, fontWeight: 700, display: "block" }}>{nodeId}</MonoText>
          <div style={{ display: "flex", gap: 6, marginTop: 5, alignItems: "center" }}>
            <RiskBadge level={node.risk} />
          </div>
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", color: DS.color.textSec, cursor: "pointer", padding: 4 }}>{Icons.close(11)}</button>
      </div>

      <div style={{ textAlign: "center", padding: "12px 0 16px", borderBottom: `1px solid ${DS.color.border}`, marginBottom: 14 }}>
        <div style={{ fontSize: 36, fontWeight: 800, color: riskCol(node.risk), letterSpacing: -2, lineHeight: 1 }}>{pprScore.toFixed(2)}</div>
        <div style={{ fontSize: 10, color: DS.color.textSec, marginTop: 4 }}>Risk Score (PPR proxy)</div>
        <div style={{ marginTop: 8, height: 5, background: DS.glass.riskBarTrack, borderRadius: 3, overflow: "hidden" }}>
          <div style={{ width: `${pprScore * 100}%`, height: "100%", background: riskCol(node.risk), borderRadius: 3 }}></div>
        </div>
      </div>

      <Section title="Risk Signals">
        <Row label="In-degree" value={node.in ?? "—"} />
        <Row label="Out-degree" value={node.out ?? "—"} />
      </Section>

      {flags?.flags?.length > 0 && (
        <Section title="Transaction Flags">
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {flags.flags.map((f, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 10, color: DS.color.riskHigh, fontWeight: 600 }}>⚑ {f.flag_type}</span>
                <span style={{ fontSize: 10, color: DS.color.textSec }}>{f.detail || ""}</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Fix (20-Jul): tampilkan error eksplisit, jangan diam-diam kosong */}
      {(ppr?._error || flags?._error) && (
        <div style={{ fontSize: 10, color: DS.color.riskHigh || "#FF6B6B", background: "rgba(255,107,107,0.08)", border: "1px solid rgba(255,107,107,0.25)", borderRadius: 6, padding: "6px 8px", marginTop: 6 }}>
          ⚠ Gagal memuat sebagian sinyal (PPR/flags) — koneksi bermasalah.
          Ini BUKAN berarti rekening bersih; coba pilih ulang node.
        </div>
      )}

      {ppr?.scores && Object.keys(ppr.scores).length > 0 && (
        <Section title="Tetangga Berisiko Tinggi (PPR)">
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {Object.entries(ppr.scores).slice(0, 6).map(([acc, sc]) => (
              <div key={acc} style={{ display: "flex", justifyContent: "space-between" }}>
                <MonoText style={{ fontSize: 10, color: DS.color.blue }}>{acc}</MonoText>
                <span style={{ fontSize: 10, color: DS.color.textSec }}>{sc.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      <div style={{ display: "flex", gap: 8, paddingTop: 8, borderTop: `1px solid ${DS.color.border}`, marginTop: 4 }}>
        <button onClick={() => onOpenCase && onOpenCase(nodeId)} style={{ flex: 1, background: DS.color.blue, color: "#fff", border: "none", borderRadius: 6, padding: 8, fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Buka Case</button>
      </div>
    </div>
  );
};

const ClusterNavigator = ({ clusters, selectedCluster, setSelectedCluster, showLabels, setShowLabels }) => (
  <div style={{ flex: "0 0 280px", background: DS.glass.bgPanel, backdropFilter: DS.glass.blurPanel, borderRight: `1px solid ${DS.glass.panelBorder}`, overflowY: "auto", display: "flex", flexDirection: "column" }}>
    <div style={{ padding: 12, borderBottom: `1px solid ${DS.color.border}` }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: DS.color.textSec, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 8 }}>
        Cluster List <span style={{ fontWeight: 400 }}>({clusters.length})</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {clusters.length === 0 && <div style={{ fontSize: 11, color: DS.color.textSec }}>Belum ada cluster terdeteksi.</div>}
        {clusters.map((c) => (
          <div
            key={c.cluster_id}
            onClick={() => setSelectedCluster(c.cluster_id)}
            style={{ padding: "7px 9px", borderRadius: 6, cursor: "pointer", background: selectedCluster === c.cluster_id ? `${DS.color.blue}18` : "transparent", border: selectedCluster === c.cluster_id ? `1px solid ${DS.color.blue}30` : "1px solid transparent" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: selectedCluster === c.cluster_id ? DS.color.blue : DS.color.textPri, fontWeight: 600 }}>{c.cluster_id}</span>
              <RiskBadge level={(c.risk_level || "low").toLowerCase()} />
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 4, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: DS.color.textSec }}>{c.size} node</span>
            </div>
          </div>
        ))}
      </div>
    </div>
    <div style={{ padding: 12 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: DS.color.textSec, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10 }}>Filter Graph</div>
      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}>
        <span style={{ fontSize: 11, color: DS.color.textSec }}>Tampilkan label node</span>
        <div onClick={() => setShowLabels(!showLabels)} style={{ width: 32, height: 18, borderRadius: 9, cursor: "pointer", background: showLabels ? DS.color.blue : DS.glass.riskBarTrack, position: "relative" }}>
          <div style={{ position: "absolute", top: 2, left: showLabels ? 16 : 2, width: 14, height: 14, borderRadius: "50%", background: "#fff", transition: "left 0.2s" }}></div>
        </div>
      </label>
    </div>
  </div>
);

export default function GraphExplorer({ onOpenCase }) {
  const [clusters, setClusters] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [clusterDetail, setClusterDetail] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showLabels, setShowLabels] = useState(false);
  const [loadingCluster, setLoadingCluster] = useState(false);
  const [clusterLoadError, setClusterLoadError] = useState(false);

  useEffect(() => {
    listClusters(2).then((d) => {
      setClusters(d.items);
      if (d.items.length > 0) setSelectedCluster(d.items[0].cluster_id);
    }).catch((e) => console.error("Gagal load clusters:", e));
  }, []);

  useEffect(() => {
    if (!selectedCluster) return;
    setLoadingCluster(true);
    setClusterLoadError(false);
    setSelectedNode(null);
    getClusterDetail(selectedCluster)
      .then((d) => setClusterDetail(d))
      .catch((e) => {
        console.error("Gagal load cluster detail:", e);
        setClusterDetail(null);
        setClusterLoadError(true);
      })
      .finally(() => setLoadingCluster(false));
  }, [selectedCluster]);

  const layout = useMemo(() => {
    if (!clusterDetail) return null;
    // Batasi jumlah node yg dirender (cluster HIGH-risk bisa ribuan node —
    // render semuanya bikin SVG lambat/tak terbaca). Sample sampai 120 node.
    const MAX_RENDER = 120;
    const nodeIds = clusterDetail.nodes.slice(0, MAX_RENDER);
    const nodeSet = new Set(nodeIds);
    const nodes = nodeIds.map((id, i) => ({
      id, r: i === 0 ? 18 : 9 + Math.random() * 4,
      risk: clusterDetail.risk_level ? clusterDetail.risk_level.toLowerCase() : "medium",
      isCollector: i === 0,
    }));
    const edges = clusterDetail.edges
      .filter((e) => nodeSet.has(e.src) && nodeSet.has(e.dst))
      .map((e) => ({ src: e.src, dst: e.dst, amount: e.amount, is_laundering: e.is_laundering }));
    return computeLayout(nodes, edges);
  }, [clusterDetail]);

  const selectedNodeObj = layout?.nodes.find((n) => n.id === selectedNode) || null;

  return (
    <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
      <ClusterNavigator clusters={clusters} selectedCluster={selectedCluster} setSelectedCluster={setSelectedCluster} showLabels={showLabels} setShowLabels={setShowLabels} />

      <div style={{ flex: 1, position: "relative", background: "#080B12", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 4, background: "rgba(26,29,39,0.92)", border: `1px solid ${DS.color.border}`, borderRadius: 8, padding: "6px 10px", zIndex: 10 }}>
          {clusterDetail && (
            <span style={{ fontSize: 11, color: DS.color.textSec, padding: "5px 8px" }}>
              {clusterDetail.cluster_id} · {clusterDetail.nodes.length} node total{clusterDetail.nodes.length > 120 ? " (120 ditampilkan)" : ""}
            </span>
          )}
        </div>

        {loadingCluster ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: DS.color.textSec, fontSize: 12 }}>Memuat graph…</div>
        ) : clusterLoadError ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: DS.color.riskHigh, fontSize: 12 }}>Gagal memuat detail cluster — coba pilih ulang atau refresh.</div>
        ) : (
          <GraphSVG layout={layout} selected={selectedNode} onSelect={setSelectedNode} showLabels={showLabels} />
        )}
        <GraphLegend />
      </div>

      <NodeDetail nodeId={selectedNode} node={selectedNodeObj} onClose={() => setSelectedNode(null)} onOpenCase={onOpenCase} />
    </div>
  );
}
