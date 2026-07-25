// =============================================================
// OsintIntelligence.jsx — Halaman 0: OSINT Intelligence (Phase 6.1)
// Crawler Control Panel (tombol ON/OFF, Phase 4.5.12) + rekening ditemukan
// + jaringan bandar + seed ke graph. Data dari /osint/* (lihat lib/api.js).
// =============================================================
import { useState, useEffect, useCallback, useRef } from "react";
import {
  getOsintStatus, startOsintCrawl, stopOsintCrawl,
  listOsintAccounts, listOsintNetworks, seedAllOsint, seedOsintAccount,
} from "../lib/api";
import { DS, Card, MonoText } from "../design/system";

// Badge bank kecil (reuse warna dari DS kalau ada, fallback netral)
function BankBadge({ bank }) {
  const b = (bank || "?").toUpperCase();
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 6,
      background: "rgba(107,174,255,0.15)", color: "#6BAEFF",
      letterSpacing: 0.3, whiteSpace: "nowrap",
    }}>{b}</span>
  );
}

function RiskPill({ level }) {
  const map = {
    HIGH: { c: DS.color.riskHigh, bg: "rgba(255,90,90,0.14)" },
    MED: { c: DS.color.riskMed, bg: "rgba(255,180,60,0.14)" },
    LOW: { c: DS.color.riskLow, bg: "rgba(120,200,140,0.14)" },
  };
  const s = map[level] || map.LOW;
  return (
    <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 10, background: s.bg, color: s.c }}>
      {level || "LOW"}
    </span>
  );
}

// Kartu statistik ringkas
function StatCard({ label, value, accent, sub }) {
  return (
    <Card pad={14} style={{ flex: 1, minWidth: 140 }}>
      <div style={{ fontSize: 11, color: DS.color.textSec, marginBottom: 6, letterSpacing: 0.3 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 800, color: accent || DS.color.textPri, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: DS.color.textSec, marginTop: 5 }}>{sub}</div>}
    </Card>
  );
}

export default function OsintIntelligence() {
  const [status, setStatus] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const [networks, setNetworks] = useState([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const pollRef = useRef(null);

  const loadStatus = useCallback(() => {
    getOsintStatus().then(setStatus).catch(() => {});
  }, []);
  const loadData = useCallback(() => {
    listOsintAccounts({ limit: 100 }).then((d) => setAccounts(d.items || [])).catch(() => {});
    listOsintNetworks(false).then((d) => setNetworks(d.items || [])).catch(() => {});
  }, []);

  useEffect(() => {
    loadStatus();
    loadData();
  }, [loadStatus, loadData]);

  // Polling saat crawler ON — refresh status + data tiap 5 detik.
  useEffect(() => {
    const on = status?.crawling === "ON";
    if (on && !pollRef.current) {
      pollRef.current = setInterval(() => { loadStatus(); loadData(); }, 5000);
    } else if (!on && pollRef.current) {
      clearInterval(pollRef.current); pollRef.current = null;
    }
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [status?.crawling, loadStatus, loadData]);

  const crawling = status?.crawling === "ON";
  const playwrightOk = status?.playwright_available !== false;

  const toggleCrawl = async () => {
    setBusy(true); setMsg(null);
    try {
      if (crawling) {
        const r = await stopOsintCrawl();
        setMsg(r.message || "Crawler dihentikan.");
      } else {
        const r = await startOsintCrawl(5);
        setMsg(r.message || "Crawler dimulai.");
      }
      setTimeout(loadStatus, 800);
    } catch (e) {
      const detail = e?.response?.data?.detail || "Gagal mengubah status crawler.";
      setMsg(detail);
    } finally { setBusy(false); }
  };

  const handleSeedAll = async () => {
    setBusy(true); setMsg(null);
    try {
      const r = await seedAllOsint();
      setMsg(`Seed selesai: ${r.seeded ?? r.count ?? 0} rekening masuk graph.`);
      loadData();
    } catch {
      setMsg("Gagal seed ke graph.");
    } finally { setBusy(false); }
  };

  const handleSeedOne = async (rek) => {
    try {
      await seedOsintAccount(rek);
      loadData();
    } catch { /* ignore per-row */ }
  };

  const totalRek = accounts.length;
  const seeded = accounts.filter((a) => a.seeded_to_graph).length;
  const pendingSeed = totalRek - seeded;
  const highNet = networks.filter((n) => n.risk_level === "HIGH").length;

  return (
    <div style={{ flex: 1, overflow: "auto", padding: 22 }}>
      {/* ── Intel Summary Cards ── */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <StatCard label="Rekening Ditemukan" value={totalRek} accent="#6BAEFF" />
        <StatCard label="Jaringan Bandar" value={networks.length} accent={DS.color.riskMed} sub={`${highNet} risiko tinggi`} />
        <StatCard label="Sudah di-Seed" value={seeded} accent={DS.color.riskLow} sub="masuk graph" />
        <StatCard label="Menunggu Seed" value={pendingSeed} accent={DS.color.textPri} />
      </div>

      {/* ── Crawler Control Panel ── */}
      <Card pad={18} style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            {/* Indikator status */}
            <div style={{
              width: 12, height: 12, borderRadius: "50%",
              background: crawling ? "#3BD98A" : "rgba(255,255,255,0.25)",
              boxShadow: crawling ? "0 0 12px #3BD98A" : "none",
              transition: "all 0.3s",
            }} />
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: DS.color.textPri }}>
                OSINT Crawler {crawling ? "AKTIF" : "Nonaktif"}
              </div>
              <div style={{ fontSize: 11, color: DS.color.textSec, marginTop: 2 }}>
                {crawling
                  ? `Menyisir situs judol • ${status?.pending ?? 0} antre • ${status?.done_today ?? 0} selesai hari ini`
                  : "Tekan ON untuk mulai berburu rekening bandar dari situs judol"}
              </div>
            </div>
          </div>

          {/* Toggle ON/OFF */}
          <button
            onClick={toggleCrawl}
            disabled={busy || !playwrightOk}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "10px 22px", borderRadius: 10, cursor: busy || !playwrightOk ? "not-allowed" : "pointer",
              border: "none", fontSize: 14, fontWeight: 700,
              background: crawling ? "rgba(255,90,90,0.15)" : "linear-gradient(135deg, #3BD98A, #1DB574)",
              color: crawling ? DS.color.riskHigh : "#04231A",
              opacity: busy || !playwrightOk ? 0.55 : 1, transition: "all 0.2s",
            }}
          >
            {crawling ? "■ Hentikan (OFF)" : "▶ Mulai Crawl (ON)"}
          </button>
        </div>

        {!playwrightOk && (
          <div style={{ marginTop: 12, fontSize: 11, color: DS.color.riskMed, background: "rgba(255,180,60,0.1)", padding: "8px 12px", borderRadius: 8 }}>
            ⚠ Playwright belum terpasang di server — crawler tidak bisa dijalankan. Jalankan: <MonoText>pip install playwright && python -m playwright install chromium</MonoText>
          </div>
        )}
        {msg && (
          <div style={{ marginTop: 12, fontSize: 12, color: DS.color.textSec, background: "rgba(107,174,255,0.08)", padding: "8px 12px", borderRadius: 8 }}>
            {msg}
          </div>
        )}
      </Card>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* ── Tabel Rekening Ditemukan ── */}
        <div style={{ flex: 2, minWidth: 380 }}>
          <Card pad={0}>
            <div style={{ padding: "14px 16px", borderBottom: `1px solid ${DS.color.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: DS.color.textPri }}>Rekening Bandar Ditemukan</span>
              <button
                onClick={handleSeedAll}
                disabled={busy || pendingSeed === 0}
                style={{
                  fontSize: 11, fontWeight: 600, padding: "6px 12px", borderRadius: 7,
                  background: "rgba(107,174,255,0.15)", color: "#6BAEFF", border: "none",
                  cursor: busy || pendingSeed === 0 ? "not-allowed" : "pointer", opacity: pendingSeed === 0 ? 0.5 : 1,
                }}
              >
                Seed Semua ke Graph ({pendingSeed})
              </button>
            </div>
            {accounts.length === 0 ? (
              <div style={{ padding: 40, textAlign: "center", color: DS.color.textSec, fontSize: 13 }}>
                Belum ada rekening. Nyalakan crawler untuk mulai berburu.
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: DS.color.textSec, textAlign: "left" }}>
                      <th style={{ padding: "9px 16px", fontWeight: 500 }}>Rekening</th>
                      <th style={{ padding: "9px 8px", fontWeight: 500 }}>Bank</th>
                      <th style={{ padding: "9px 8px", fontWeight: 500 }}>Situs</th>
                      <th style={{ padding: "9px 8px", fontWeight: 500 }}>Status</th>
                      <th style={{ padding: "9px 16px", fontWeight: 500 }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {accounts.map((a) => (
                      <tr key={a.rekening} style={{ borderTop: `1px solid ${DS.color.border}` }}>
                        <td style={{ padding: "9px 16px" }}><MonoText style={{ color: DS.color.textPri }}>{a.rekening}</MonoText></td>
                        <td style={{ padding: "9px 8px" }}><BankBadge bank={a.bank} /></td>
                        <td style={{ padding: "9px 8px", color: a.shared_count > 1 ? DS.color.riskHigh : DS.color.textSec, fontWeight: a.shared_count > 1 ? 700 : 400 }}>
                          {a.shared_count}{a.shared_count > 1 ? " situs (bandar!)" : " situs"}
                        </td>
                        <td style={{ padding: "9px 8px" }}>
                          {a.seeded_to_graph
                            ? <span style={{ fontSize: 10, color: DS.color.riskLow }}>✓ di-graph</span>
                            : <span style={{ fontSize: 10, color: DS.color.textSec }}>belum</span>}
                        </td>
                        <td style={{ padding: "9px 16px", textAlign: "right" }}>
                          {!a.seeded_to_graph && (
                            <button onClick={() => handleSeedOne(a.rekening)} style={{ fontSize: 10, fontWeight: 600, padding: "4px 10px", borderRadius: 6, background: "transparent", color: "#6BAEFF", border: "1px solid rgba(107,174,255,0.3)", cursor: "pointer" }}>
                              Seed
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>

        {/* ── Peta Jaringan Bandar (list) ── */}
        <div style={{ flex: 1, minWidth: 260 }}>
          <Card pad={0}>
            <div style={{ padding: "14px 16px", borderBottom: `1px solid ${DS.color.border}` }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: DS.color.textPri }}>Jaringan Bandar</span>
              <div style={{ fontSize: 10, color: DS.color.textSec, marginTop: 2 }}>Rekening yang dipakai lintas situs</div>
            </div>
            {networks.length === 0 ? (
              <div style={{ padding: 30, textAlign: "center", color: DS.color.textSec, fontSize: 12 }}>
                Belum ada jaringan terdeteksi.
              </div>
            ) : (
              <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                {networks.map((n) => (
                  <div key={n.network_id} style={{ padding: "10px 12px", borderRadius: 9, background: "rgba(255,255,255,0.03)", border: `1px solid ${DS.color.border}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                      <span style={{ fontSize: 12, fontWeight: 700, color: DS.color.textPri }}>Jaringan #{n.network_id}</span>
                      <RiskPill level={n.risk_level} />
                    </div>
                    <div style={{ fontSize: 10, color: DS.color.textSec }}>
                      {(n.rekening_list || []).length} rekening • {(n.site_list || []).length} situs terhubung
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
