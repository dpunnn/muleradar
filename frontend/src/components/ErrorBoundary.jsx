// =============================================================
// ErrorBoundary.jsx — jaring pengaman React (fix 20-Jul, scan produksi).
// Sebelumnya TIDAK ADA: 1 komponen throw saat render -> SELURUH app blank
// putih (bukan fallback UI). Untuk tool yang dipakai analis kerja beneran,
// itu risiko UX serius. Class component wajib (getDerivedStateFromError /
// componentDidCatch hanya tersedia di class, bukan hook).
// =============================================================
import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Log ke console (dan nanti bisa disambung ke observability Phase 12).
    console.error("[ErrorBoundary] render error:", error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div style={{
        minHeight: "100vh", display: "flex", alignItems: "center",
        justifyContent: "center", background: "#0a1020", color: "#e6edf7",
        fontFamily: "'Inter', system-ui, sans-serif", padding: 24,
      }}>
        <div style={{
          maxWidth: 460, textAlign: "center", background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.12)", borderRadius: 14, padding: "32px 28px",
        }}>
          <div style={{ fontSize: 34, marginBottom: 8 }}>⚠️</div>
          <h1 style={{ fontSize: 18, fontWeight: 700, margin: "0 0 8px" }}>
            Terjadi kesalahan pada tampilan
          </h1>
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "rgba(230,237,247,0.65)", margin: "0 0 20px" }}>
            Satu bagian antarmuka gagal dimuat. Data kamu aman — ini masalah
            tampilan, bukan kehilangan data. Coba muat ulang halaman.
          </p>
          {this.state.error?.message && (
            <pre style={{
              fontSize: 11, textAlign: "left", color: "rgba(255,143,163,0.9)",
              background: "rgba(255,143,163,0.08)", border: "1px solid rgba(255,143,163,0.25)",
              borderRadius: 8, padding: "8px 10px", overflowX: "auto", marginBottom: 20,
            }}>
              {String(this.state.error.message).slice(0, 300)}
            </pre>
          )}
          <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
            <button
              onClick={() => window.location.reload()}
              style={{ background: "#6BAEFF", color: "#04122b", border: "none", borderRadius: 8, padding: "9px 18px", fontSize: 13, fontWeight: 700, cursor: "pointer" }}
            >
              Muat Ulang Halaman
            </button>
            <button
              onClick={this.handleReset}
              style={{ background: "transparent", color: "rgba(230,237,247,0.7)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "9px 18px", fontSize: 13, fontWeight: 600, cursor: "pointer" }}
            >
              Coba Lagi
            </button>
          </div>
        </div>
      </div>
    );
  }
}
