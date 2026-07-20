// =============================================================
// system.jsx — MuleRadar Design System (Dark + Light)
// Diadaptasi dari pagemuleradar/MuleRadar.zip (common.jsx) — visual/style
// dipertahankan PERSIS, cuma dikonversi dari global `window.X` (React CDN +
// Babel standalone, tanpa build step) ke ES module biasa (import/export)
// supaya bisa dipakai di project Vite ini.
// =============================================================

const DS_DARK = {
  color: {
    bgMain: "transparent",
    bgCard: "rgba(255,255,255,0.07)",
    bgSidebar: "rgba(8,12,32,0.72)",
    blue: "#6BAEFF",
    riskHigh: "#FF6B72",
    riskMed: "#FFB347",
    riskLow: "#4ADE80",
    riskGray: "rgba(255,255,255,0.38)",
    textPri: "#FFFFFF",
    textSec: "rgba(255,255,255,0.55)",
    border: "rgba(255,255,255,0.13)",
    borderHov: "rgba(255,255,255,0.22)",
    bankA: "#6BAEFF",
    bankB: "#B07EFF",
    bankC: "#2DD4BF",
    bankABg: "rgba(107,174,255,0.13)",
    bankBBg: "rgba(176,126,255,0.13)",
    bankCBg: "rgba(45,212,191,0.13)",
  },
  glass: {
    bg: "rgba(255,255,255,0.07)",
    bgMid: "rgba(255,255,255,0.11)",
    bgPanel: "rgba(8,14,38,0.75)",
    bgInput: "rgba(0,0,0,0.28)",
    bgModal: "rgba(12,18,48,0.84)",
    bgSticky: "rgba(8,14,38,0.82)",
    bgToolbar: "rgba(10,14,30,0.76)",
    blur: "blur(24px) saturate(1.8)",
    blurPanel: "blur(24px)",
    blurInput: "blur(12px)",
    blurHeavy: "blur(40px) saturate(2)",
    blurToolbar: "blur(20px)",
    border: "1px solid rgba(255,255,255,0.14)",
    borderMed: "1px solid rgba(255,255,255,0.10)",
    shadow: "0 8px 32px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.09)",
    panelBorder: "rgba(255,255,255,0.10)",
    riskBarTrack: "rgba(255,255,255,0.10)",
    rowAlt: "rgba(255,255,255,0.04)",
    rowHover: "rgba(255,255,255,0.07)",
    topNavBg: "rgba(255,255,255,0.05)",
    topNavBlur: "blur(24px) saturate(1.8)",
    navBorder: "rgba(255,255,255,0.09)",
    thBorder: "rgba(255,255,255,0.10)",
  },
  typoColor: {
    Scam: "#C084FC",
    Judol: "#FB923C",
    Dormant: "#94A3B8",
    "Vendor Cangkang": "#2DD4BF",
    "QRIS Ring": "#22D3EE",
    Bendahara: "#FCD34D",
    "PEP Network": "#F59E0B",
    UNKNOWN: "#94A3B8",
  },
};

const DS_LIGHT = {
  color: {
    bgMain: "#F0F4FA",
    bgCard: "#FFFFFF",
    bgSidebar: "rgba(8,12,32,0.72)",
    blue: "#4F8EF7",
    riskHigh: "#FF4D4F",
    riskMed: "#FA8C16",
    riskLow: "#52C41A",
    riskGray: "#9CA3AF",
    textPri: "#111827",
    textSec: "#6B7280",
    border: "#E5E8F0",
    borderHov: "#CBD5E1",
    bankA: "#4F8EF7",
    bankB: "#9333EA",
    bankC: "#0D9488",
    bankABg: "rgba(79,142,247,0.10)",
    bankBBg: "rgba(147,51,234,0.10)",
    bankCBg: "rgba(13,148,136,0.10)",
  },
  glass: {
    bg: "#FFFFFF",
    bgMid: "#F8FAFD",
    bgPanel: "#FFFFFF",
    bgInput: "#F0F4FA",
    bgModal: "#FFFFFF",
    bgSticky: "#FFFFFF",
    bgToolbar: "rgba(255,255,255,0.95)",
    blur: "none",
    blurPanel: "none",
    blurInput: "none",
    blurHeavy: "none",
    blurToolbar: "none",
    border: "1px solid #E5E8F0",
    borderMed: "1px solid #E5E8F0",
    shadow: "0 2px 12px rgba(0,0,0,0.06)",
    panelBorder: "#E5E8F0",
    riskBarTrack: "#EEF0F6",
    rowAlt: "#F8FAFD",
    rowHover: "rgba(79,142,247,0.04)",
    topNavBg: "#FFFFFF",
    topNavBlur: "none",
    navBorder: "#E5E8F0",
    thBorder: "#E5E8F0",
  },
  typoColor: {
    Scam: "#8B5CF6",
    Judol: "#F97316",
    Dormant: "#64748B",
    "Vendor Cangkang": "#0D9488",
    "QRIS Ring": "#06B6D4",
    Bendahara: "#D97706",
    "PEP Network": "#B45309",
    UNKNOWN: "#64748B",
  },
};

// ── Theme state (module-level, sama pola dgn window.__MULEX_THEME__ asli) ──
let _theme = "dark";
const _listeners = new Set();

export function setGlobalTheme(t) {
  _theme = t;
  _listeners.forEach((fn) => fn(t));
}
export function getGlobalTheme() {
  return _theme;
}
export function subscribeTheme(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

export const DS = new Proxy(
  {},
  {
    get(_, key) {
      return (_theme === "dark" ? DS_DARK : DS_LIGHT)[key];
    },
  }
);

export const getGlassBase = () => ({
  background: DS.glass.bg,
  backdropFilter: DS.glass.blur,
  WebkitBackdropFilter: DS.glass.blur,
  border: DS.glass.border,
  boxShadow: DS.glass.shadow,
});

export const getInputStyle = (extra) => ({
  background: DS.glass.bgInput,
  backdropFilter: DS.glass.blurInput,
  WebkitBackdropFilter: DS.glass.blurInput,
  border: DS.glass.border,
  borderRadius: 7,
  color: DS.color.textPri,
  outline: "none",
  ...extra,
});

// ── Icons ─────────────────────────────────────────────────
export const Icons = {
  dashboard: (s = 16) => (<svg width={s} height={s} viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="6" height="6" rx="1.5" fill="currentColor" opacity="0.9"/><rect x="9" y="1" width="6" height="6" rx="1.5" fill="currentColor" opacity="0.9"/><rect x="1" y="9" width="6" height="6" rx="1.5" fill="currentColor" opacity="0.9"/><rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor" opacity="0.9"/></svg>),
  bell: (s = 16) => (<svg width={s} height={s} viewBox="0 0 16 16" fill="none"><path d="M8 1.5C5.79 1.5 4 3.29 4 5.5v3.5L2.5 11h11L12 9V5.5C12 3.29 10.21 1.5 8 1.5z" stroke="currentColor" strokeWidth="1.3" fill="none"/><path d="M6.5 11v.5a1.5 1.5 0 003 0V11" stroke="currentColor" strokeWidth="1.3"/></svg>),
  network: (s = 16) => (<svg width={s} height={s} viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="1.8" fill="currentColor"/><circle cx="2.5" cy="4" r="1.5" fill="currentColor" opacity="0.8"/><circle cx="13.5" cy="4" r="1.5" fill="currentColor" opacity="0.8"/><circle cx="2.5" cy="12" r="1.5" fill="currentColor" opacity="0.8"/><circle cx="13.5" cy="12" r="1.5" fill="currentColor" opacity="0.8"/><line x1="4" y1="4" x2="6.2" y2="7" stroke="currentColor" strokeWidth="1.2" opacity="0.7"/><line x1="12" y1="4" x2="9.8" y2="7" stroke="currentColor" strokeWidth="1.2" opacity="0.7"/><line x1="4" y1="12" x2="6.2" y2="9" stroke="currentColor" strokeWidth="1.2" opacity="0.7"/><line x1="12" y1="12" x2="9.8" y2="9" stroke="currentColor" strokeWidth="1.2" opacity="0.7"/></svg>),
  folder: (s = 16) => (<svg width={s} height={s} viewBox="0 0 16 16" fill="none"><path d="M1.5 4.5C1.5 3.67 2.17 3 3 3h3l1.5 2H13c.83 0 1.5.67 1.5 1.5v6c0 .83-.67 1.5-1.5 1.5H3c-.83 0-1.5-.67-1.5-1.5v-8z" stroke="currentColor" strokeWidth="1.3" fill="none"/></svg>),
  settings: (s = 16) => (<svg width={s} height={s} viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="2.2" stroke="currentColor" strokeWidth="1.3"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M2.93 2.93l1.41 1.41M11.66 11.66l1.41 1.41M2.93 13.07l1.41-1.41M11.66 4.34l1.41-1.41" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>),
  search: (s = 14) => (<svg width={s} height={s} viewBox="0 0 14 14" fill="none"><circle cx="6" cy="6" r="4" stroke="currentColor" strokeWidth="1.4"/><line x1="9" y1="9" x2="13" y2="13" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/></svg>),
  chevronRight: (s = 12) => (<svg width={s} height={s} viewBox="0 0 12 12" fill="none"><path d="M4.5 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  refresh: (s = 13) => (<svg width={s} height={s} viewBox="0 0 13 13" fill="none"><path d="M11 6.5A4.5 4.5 0 012 6.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/><path d="M11 6.5V3M11 3H8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  download: (s = 13) => (<svg width={s} height={s} viewBox="0 0 13 13" fill="none"><path d="M6.5 1v7M3.5 5.5l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/><path d="M1.5 10.5h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/></svg>),
  close: (s = 12) => (<svg width={s} height={s} viewBox="0 0 12 12" fill="none"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>),
  spark: (s = 13) => (<svg width={s} height={s} viewBox="0 0 13 13" fill="none"><path d="M6.5 1L8 5h4l-3 2.5L10 12 6.5 9 3 12l1-4.5L1 5h4z" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinejoin="round"/></svg>),
  doc: (s = 13) => (<svg width={s} height={s} viewBox="0 0 13 13" fill="none"><rect x="2" y="1" width="9" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.2"/><line x1="4" y1="4.5" x2="9" y2="4.5" stroke="currentColor" strokeWidth="1"/><line x1="4" y1="7" x2="9" y2="7" stroke="currentColor" strokeWidth="1"/><line x1="4" y1="9.5" x2="7" y2="9.5" stroke="currentColor" strokeWidth="1"/></svg>),
  zoomIn: (s = 14) => (<svg width={s} height={s} viewBox="0 0 14 14" fill="none"><circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.3"/><path d="M4 6h4M6 4v4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/><line x1="9.5" y1="9.5" x2="13" y2="13" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>),
  zoomOut: (s = 14) => (<svg width={s} height={s} viewBox="0 0 14 14" fill="none"><circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.3"/><path d="M4 6h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/><line x1="9.5" y1="9.5" x2="13" y2="13" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>),
  expand: (s = 14) => (<svg width={s} height={s} viewBox="0 0 14 14" fill="none"><path d="M1 1h4M1 1v4M13 1h-4M13 1v4M1 13h4M1 13v-4M13 13h-4M13 13v-4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>),
  sun: (s = 14) => (<svg width={s} height={s} viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="2.5" stroke="currentColor" strokeWidth="1.3"/><path d="M7 1v1.5M7 11.5V13M1 7h1.5M11.5 7H13M2.93 2.93l1 1M10.07 10.07l1 1M2.93 11.07l1-1M10.07 3.93l1-1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>),
  moon: (s = 14) => (<svg width={s} height={s} viewBox="0 0 14 14" fill="none"><path d="M11.5 8A5 5 0 016 2.5a5 5 0 100 9 5 5 0 005.5-3.5z" stroke="currentColor" strokeWidth="1.3" fill="none"/></svg>),
};

// ── Shared Components ─────────────────────────────────────
export const MonoText = ({ children, style }) => (
  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, ...style }}>{children}</span>
);

export const Card = ({ children, style, pad = 16, glow }) => {
  const base = getGlassBase();
  return (
    <div
      style={{
        ...base,
        borderRadius: 12,
        padding: pad,
        boxShadow: glow ? `${DS.glass.shadow}, 0 0 40px ${glow}22` : DS.glass.shadow,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

export const RiskBadge = ({ level }) => {
  const map = {
    high: { color: DS.color.riskHigh, label: "HIGH" },
    medium: { color: DS.color.riskMed, label: "MED" },
    low: { color: DS.color.riskLow, label: "LOW" },
    unknown: { color: DS.color.riskGray, label: "—" },
  };
  const { color, label } = map[level] || map.unknown;
  const isDark = getGlobalTheme() === "dark";
  return (
    <span
      style={{
        display: "inline-flex", alignItems: "center", gap: 4,
        padding: "2px 7px", borderRadius: 20,
        background: `${color}20`, color,
        fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
        border: `1px solid ${color}38`,
        backdropFilter: isDark ? "blur(8px)" : "none",
      }}
    >
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: color, boxShadow: isDark ? `0 0 6px ${color}` : "none" }}></span>
      {label}
    </span>
  );
};

// riskScoreToLevel: helper krn API kita balikin angka 0-1, bukan string level
export const riskScoreToLevel = (score) => {
  if (score == null) return "unknown";
  if (score >= 0.75) return "high";
  if (score >= 0.4) return "medium";
  return "low";
};

export const StatusBadge = ({ status }) => {
  const norm = (status || "").replace("_", " ").toUpperCase();
  const map = {
    NEW: { color: DS.color.riskHigh, pulse: true },
    "IN REVIEW": { color: DS.color.riskMed, pulse: false },
    CONFIRM: { color: DS.color.riskLow, pulse: false },
    FP: { color: DS.color.riskGray, pulse: false },
    CLOSED: { color: DS.color.riskGray, pulse: false },
  };
  const { color, pulse } = map[norm] || map.FP;
  return (
    <span
      style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        padding: "2px 8px", borderRadius: 20,
        background: `${color}18`, color,
        fontSize: 10, fontWeight: 700, letterSpacing: 0.4,
        border: `1px solid ${color}35`,
        whiteSpace: "nowrap",
      }}
    >
      {pulse && <span className="pulse-dot" style={{ background: color }}></span>}
      {norm}
    </span>
  );
};

export const TypologyBadge = ({ type, small }) => {
  const label = type || "UNKNOWN";
  const color = DS.typoColor[label] || DS.color.riskGray;
  return (
    <span
      style={{
        display: "inline-block",
        padding: small ? "1px 7px" : "2px 9px",
        borderRadius: 20,
        background: `${color}18`,
        color: color,
        fontSize: small ? 10 : 11,
        fontWeight: 600,
        border: `1px solid ${color}35`,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
};

export const RiskBar = ({ score, width = 80 }) => {
  const s = score ?? 0;
  const color = s >= 0.75 ? DS.color.riskHigh : s >= 0.4 ? DS.color.riskMed : DS.color.riskLow;
  const isDark = getGlobalTheme() === "dark";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width, height: 5, background: DS.glass.riskBarTrack, borderRadius: 3, overflow: "hidden" }}>
        <div
          style={{
            width: `${s * 100}%`, height: "100%",
            background: isDark ? `linear-gradient(90deg, ${color}88, ${color})` : color,
            borderRadius: 3,
            boxShadow: isDark ? `0 0 8px ${color}88` : "none",
          }}
        ></div>
      </div>
      <span style={{ fontSize: 11, color, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{s.toFixed(2)}</span>
    </div>
  );
};

export const SectionHeader = ({ title, action, onAction }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
    <span style={{ fontSize: 13, fontWeight: 600, color: DS.color.textPri }}>{title}</span>
    {action && (
      <span onClick={onAction} style={{ fontSize: 11, color: DS.color.blue, cursor: "pointer" }}>
        {action}
      </span>
    )}
  </div>
);

export const GlassInput = ({ style, ...props }) => (
  <input style={getInputStyle({ padding: "6px 10px", fontSize: 12, ...style })} {...props} />
);

export const GlassSelect = ({ style, children, ...props }) => (
  <select style={getInputStyle({ padding: "5px 8px", fontSize: 12, cursor: "pointer", ...style })} {...props}>
    {children}
  </select>
);
