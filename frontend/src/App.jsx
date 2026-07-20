import { useState, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { Sidebar, TopNav } from "./design/Layout";
import { setGlobalTheme } from "./design/system";
import Dashboard from "./pages/Dashboard";
import AlertList from "./pages/AlertList";
import GraphExplorer from "./pages/GraphExplorer";
import CaseDetail from "./pages/CaseDetail";
import CaseList from "./pages/CaseList";
import Login from "./pages/Login";
import { getDashboardOverview, getAlertDetail, createCase, listAlerts, getToken } from "./lib/api";

// Fix (17-Jul, Prioritas 2 — autentikasi): route guard sederhana. Tanpa
// token -> paksa /login. Ini cuma cek KEBERADAAN token (bukan validitas —
// validitas asli dicek backend tiap request; kalau token kedaluwarsa,
// interceptor 401 di lib/api.js yang akan redirect balik ke /login).
function RequireAuth({ children }) {
  if (!getToken()) return <Navigate to="/login" replace />;
  return children;
}

function Shell() {
  const [theme, setThemeState] = useState(() => {
    try {
      return localStorage.getItem("mr_theme") || "dark";
    } catch {
      return "dark";
    }
  });
  const [badges, setBadges] = useState({ alerts: null, cases: null });
  const navigate = useNavigate();

  // Fix (QC 15-Jul): setGlobalTheme HARUS sinkron saat render, bukan di
  // useEffect (yang jalan setelah commit) — kalau di effect, semua child
  // (Sidebar/Dashboard/dst yang baca DS.color.* langsung saat render) sempat
  // render 1x dgn _theme LAMA dulu sebelum effect sempat update, warna jadi
  // tidak ikut berubah saat toggle.
  // CATATAN (QC ronde 2): reviewer sempat usul pindah ke useLayoutEffect
  // supaya "React-idiomatic" — SENGAJA TIDAK dipakai, karena useLayoutEffect
  // (spt useEffect) baru jalan SETELAH seluruh subtree (termasuk Sidebar/
  // Dashboard) selesai di-render pada pass yang sama, artinya children
  // tetap baca _theme LAMA dulu -> bug yang sama muncul lagi. Panggilan
  // langsung di sini aman krn idempotent (cuma assignment var + notify
  // listener kosong) dan React memproses parent SEBELUM children dlm satu
  // render pass yang sama — itu yg justru dibutuhkan di sini.
  setGlobalTheme(theme);

  useEffect(() => {
    document.body.style.background =
      theme === "light"
        ? "#F0F4FA"
        : [
            "radial-gradient(ellipse 80% 60% at 15% 10%, rgba(79,109,247,0.22) 0%, transparent 60%)",
            "radial-gradient(ellipse 60% 50% at 85% 8%,  rgba(139,80,230,0.18) 0%, transparent 55%)",
            "radial-gradient(ellipse 55% 45% at 50% 92%, rgba(29,210,160,0.11) 0%, transparent 55%)",
            "radial-gradient(ellipse 45% 40% at 82% 75%, rgba(79,142,247,0.10) 0%, transparent 50%)",
            "linear-gradient(155deg, #050914 0%, #0C1333 35%, #120A2A 65%, #070F22 100%)",
          ].join(", ");
    document.body.style.color = theme === "light" ? "#111827" : "#FFFFFF";
    document.body.style.backgroundAttachment = "fixed";
  }, [theme]);

  const refreshBadges = useCallback(() => {
    getDashboardOverview()
      .then((d) => setBadges({ alerts: d.alert_aktif, cases: d.kasus_pending }))
      .catch(() => {});
  }, []);
  useEffect(() => {
    refreshBadges();
  }, [refreshBadges]);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("mr_theme", next);
    } catch {
      /* ignore */
    }
    setThemeState(next);
  };

  // Buka (atau buat, kalau belum ada) case dari satu alert_id, lalu navigate.
  const openCaseForAlert = useCallback(
    async (alertId) => {
      try {
        const detail = await getAlertDetail(alertId);
        if (detail.case) {
          navigate(`/cases/${detail.case.case_id}`);
          return;
        }
        const created = await createCase(alertId, null, null);
        refreshBadges();
        navigate(`/cases/${created.case_id}`);
      } catch (e) {
        // Fix (QC 15-Jul): double-click / race condition -> createCase bisa
        // kena 409 (case sudah dibuat request lain di antara getAlertDetail
        // dan createCase). Sebelumnya cuma console.error, user tersangkut
        // tanpa navigasi. Kalau 409, case-nya PASTI sudah ada -> re-fetch
        // & tetap navigate, bukan anggap gagal.
        if (e?.response?.status === 409) {
          try {
            const retry = await getAlertDetail(alertId);
            if (retry.case) {
              // Fix (QC ronde 2): path normal panggil refreshBadges() sebelum
              // navigate, path retry 409 ini sebelumnya lupa -> badge sidebar
              // (mis. "Kasus Pending") bisa stale setelah race condition.
              refreshBadges();
              navigate(`/cases/${retry.case.case_id}`);
              return;
            }
          } catch {
            /* fall through ke alert di bawah */
          }
        }
        console.error("Gagal buka case dari alert:", e);
        alert("Gagal membuka case untuk alert ini. Coba lagi.");
      }
    },
    [navigate, refreshBadges]
  );

  // Dari Dashboard/Graph Explorer kita cuma py account_id — cari alert
  // TERBARU utk akun itu, baru pakai alur yg sama dgn openCaseForAlert.
  const openCaseForAccount = useCallback(
    async (accountId) => {
      try {
        const res = await listAlerts({ account_id: accountId, limit: 1 });
        const target = res.items[0];
        if (target) {
          await openCaseForAlert(target.alert_id);
        } else {
          // Fix (QC 15-Jul): sebelumnya diam-diam pindah ke /graph tanpa
          // penjelasan — user klik "Investigasi" tapi tiba-tiba pindah
          // halaman, terlihat seperti bug (khususnya saat demo langsung).
          alert(`Akun ${accountId} belum punya alert — membuka Graph Explorer untuk investigasi manual.`);
          navigate("/graph");
        }
      } catch (e) {
        console.error("Gagal cari alert utk akun:", e);
        alert("Gagal mencari alert untuk akun ini. Coba lagi.");
      }
    },
    [openCaseForAlert, navigate]
  );

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", position: "relative", zIndex: 1 }}>
      <Sidebar badges={badges} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <TopNav theme={theme} onToggleTheme={toggleTheme} />
        <div className="screen-fade" style={{ flex: 1, display: "flex", overflow: "hidden" }}>
          <Routes>
            <Route path="/" element={<Dashboard onInvestigate={openCaseForAccount} />} />
            <Route path="/alerts" element={<AlertList onOpenDetail={openCaseForAlert} />} />
            <Route path="/graph" element={<GraphExplorer onOpenCase={openCaseForAccount} />} />
            <Route path="/cases/:caseId" element={<CaseDetail onCaseChanged={refreshBadges} />} />
            <Route path="/cases" element={<CaseList />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/*"
          element={
            <RequireAuth>
              <Shell />
            </RequireAuth>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
