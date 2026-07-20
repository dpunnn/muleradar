// =============================================================
// Layout.jsx — Sidebar + TopNav
// Diadaptasi dari pagemuleradar (common.jsx) — visual dipertahankan,
// navigasi diganti dari onNavigate(screen) manual jadi react-router-dom.
// =============================================================
import { useNavigate, useLocation } from "react-router-dom";
import { DS, Icons, getGlobalTheme, getInputStyle } from "./system";
import { clearToken } from "../lib/api";

const NAV_ITEMS = [
  { id: "dashboard", path: "/", label: "Dashboard", icon: Icons.dashboard },
  { id: "alerts", path: "/alerts", label: "Alerts", icon: Icons.bell, badgeKey: "alerts" },
  { id: "graph", path: "/graph", label: "Graph Explorer", icon: Icons.network },
  { id: "cases", path: "/cases", label: "Cases", icon: Icons.folder, badgeKey: "cases" },
];

export const Sidebar = ({ badges = {} }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const isDark = getGlobalTheme() === "dark";

  const SB = isDark
    ? {
        bg: "rgba(8,12,32,0.72)", blur: "blur(40px) saturate(2)",
        border: "1px solid rgba(255,255,255,0.08)", divider: "1px solid rgba(255,255,255,0.07)",
        textPri: "#fff", textSec: "rgba(255,255,255,0.4)",
        activeBg: "rgba(107,174,255,0.18)", activeBord: "1px solid rgba(107,174,255,0.30)",
        activeColor: "#6BAEFF", inactiveColor: "rgba(255,255,255,0.5)",
        badgeActive: "rgba(107,174,255,0.3)", badgeInact: "rgba(255,255,255,0.1)",
        badgeActClr: "#6BAEFF", badgeInClr: "rgba(255,255,255,0.4)",
      }
    : {
        bg: "#FFFFFF", blur: "none",
        border: "1px solid #E5E8F0", divider: "1px solid #E5E8F0",
        textPri: "#111827", textSec: "#9CA3AF",
        activeBg: "rgba(79,142,247,0.08)", activeBord: "1px solid rgba(79,142,247,0.25)",
        activeColor: "#4F8EF7", inactiveColor: "#6B7280",
        badgeActive: "rgba(79,142,247,0.15)", badgeInact: "#F0F4FA",
        badgeActClr: "#4F8EF7", badgeInClr: "#9CA3AF",
      };

  return (
    <div
      style={{
        width: 220, flexShrink: 0, background: SB.bg,
        backdropFilter: SB.blur, WebkitBackdropFilter: SB.blur,
        borderRight: SB.border, display: "flex", flexDirection: "column",
        height: "100vh", position: "sticky", top: 0,
        boxShadow: isDark ? "none" : "2px 0 8px rgba(0,0,0,0.04)",
      }}
    >
      <div style={{ padding: "18px 16px 14px", borderBottom: SB.divider }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div
            style={{
              width: 30, height: 30, borderRadius: 8,
              background: "linear-gradient(135deg, #6BAEFF, #9F6FED)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 800, color: "#fff",
              boxShadow: "0 4px 16px rgba(107,174,255,0.35)",
            }}
          >
            M
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: SB.textPri, letterSpacing: -0.3 }}>MuleRadar</div>
            <div style={{ fontSize: 9, color: SB.textSec, letterSpacing: 1, textTransform: "uppercase", marginTop: 1 }}>Fraud Intelligence</div>
          </div>
        </div>
      </div>
      <nav style={{ flex: 1, padding: "10px 8px" }}>
        {NAV_ITEMS.map((item) => {
          const isActive = location.pathname === item.path;
          const badge = item.badgeKey ? badges[item.badgeKey] : null;
          return (
            <button
              key={item.id}
              onClick={() => navigate(item.path)}
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 10,
                padding: "9px 10px", borderRadius: 9, marginBottom: 2,
                background: isActive ? SB.activeBg : "transparent",
                border: isActive ? SB.activeBord : "1px solid transparent",
                color: isActive ? SB.activeColor : SB.inactiveColor,
                cursor: "pointer", transition: "all 0.15s",
                fontSize: 13, fontWeight: isActive ? 600 : 400,
              }}
            >
              <span style={{ opacity: isActive ? 1 : 0.75 }}>{item.icon(15)}</span>
              <span style={{ flex: 1, textAlign: "left" }}>{item.label}</span>
              {badge != null && (
                <span
                  style={{
                    background: isActive ? SB.badgeActive : SB.badgeInact,
                    color: isActive ? SB.badgeActClr : SB.badgeInClr,
                    fontSize: 10, fontWeight: 700, padding: "1px 6px",
                    borderRadius: 10, minWidth: 20, textAlign: "center",
                  }}
                >
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>
      <div style={{ padding: "12px 14px", borderTop: SB.divider, display: "flex", alignItems: "center", gap: 9 }}>
        <div
          style={{
            width: 30, height: 30, borderRadius: "50%",
            background: "linear-gradient(135deg, #6BAEFF, #9F6FED)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 12, fontWeight: 700, color: "#fff", flexShrink: 0,
            boxShadow: "0 2px 10px rgba(107,174,255,0.35)",
          }}
        >
          R
        </div>
        <div style={{ overflow: "hidden" }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: SB.textPri, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>Rafi Ahmad</div>
          <div style={{ fontSize: 10, color: SB.textSec }}>Fraud Analyst</div>
        </div>
      </div>
    </div>
  );
};

const TITLES = { "/": "Dashboard", "/alerts": "Daftar Alert", "/graph": "Graph Explorer", "/cases": "Cases" };

export const TopNav = ({ theme, onToggleTheme }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const isDark = (theme || "dark") === "dark";
  const title = TITLES[location.pathname] || (location.pathname.startsWith("/cases/") ? "Case Detail" : "MuleRadar");

  // Fix (17-Jul, Prioritas 2 — autentikasi): avatar jadi tombol logout.
  const handleLogout = () => {
    if (window.confirm("Keluar dari MuleRadar?")) {
      clearToken();
      navigate("/login", { replace: true });
    }
  };

  return (
    <div
      style={{
        height: 56, background: DS.glass.topNavBg,
        backdropFilter: DS.glass.topNavBlur, WebkitBackdropFilter: DS.glass.topNavBlur,
        borderBottom: `1px solid ${DS.glass.navBorder}`,
        display: "flex", alignItems: "center",
        padding: "0 20px", gap: 12, flexShrink: 0,
        boxShadow: DS.glass.shadow,
      }}
    >
      <div style={{ flex: 1 }}>
        <span style={{ fontSize: 15, fontWeight: 600, color: DS.color.textPri }}>{title}</span>
      </div>
      {/* Fix (QC 15-Jul): search bar belum ada endpoint pencarian di backend
          (butuh search lintas alerts/cases/graph node — scope Phase 17
          tersendiri). Sebelumnya terlihat aktif tapi diam saat diketik,
          bikin bingung. Non-aktifkan visual + tooltip jujur, drpd terlihat
          rusak. */}
      <div
        title="Pencarian belum tersedia — segera hadir"
        style={{ display: "flex", alignItems: "center", gap: 7, opacity: 0.5, cursor: "not-allowed", ...getInputStyle({ padding: "6px 12px", width: 220, borderRadius: 8 }) }}
      >
        <span style={{ color: DS.color.textSec }}>{Icons.search(13)}</span>
        <input
          placeholder="Cari rekening, cluster... (segera hadir)"
          disabled
          style={{ background: "none", border: "none", outline: "none", color: DS.color.textSec, fontSize: 12, width: "100%", cursor: "not-allowed" }}
        />
      </div>
      <button
        onClick={onToggleTheme}
        title={isDark ? "Switch to Light" : "Switch to Dark"}
        style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          width: 32, height: 32, borderRadius: 8, cursor: "pointer",
          background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)",
          border: isDark ? "1px solid rgba(255,255,255,0.14)" : "1px solid #E5E8F0",
          color: isDark ? "#FCD34D" : "#6B7280",
          transition: "all 0.2s", flexShrink: 0,
        }}
      >
        {isDark ? Icons.sun(14) : Icons.moon(14)}
      </button>
      <div style={{ position: "relative", cursor: "pointer", padding: 6 }}>
        <span style={{ color: DS.color.textSec }}>{Icons.bell(16)}</span>
        <span style={{ position: "absolute", top: 2, right: 2, width: 8, height: 8, borderRadius: "50%", background: DS.color.riskHigh, border: `1.5px solid ${DS.glass.topNavBg}` }}></span>
      </div>
      <div
        onClick={handleLogout}
        title="Logout"
        style={{
          width: 30, height: 30, borderRadius: "50%",
          background: "linear-gradient(135deg, #6BAEFF, #9F6FED)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, fontWeight: 700, color: "#fff", cursor: "pointer",
        }}
      >
        R
      </div>
    </div>
  );
};
