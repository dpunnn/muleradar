import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { DS, getGlobalTheme, getInputStyle } from "../design/system";
import { login, setToken } from "../lib/api";

export default function Login() {
  const navigate = useNavigate();
  const isDark = getGlobalTheme() === "dark";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await login(username, password);
      setToken(res.access_token);
      navigate("/", { replace: true });
    } catch (err) {
      setError(
        err?.response?.status === 401
          ? "Username atau password salah."
          : "Gagal login — cek koneksi ke server."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        height: "100vh", width: "100%", display: "flex",
        alignItems: "center", justifyContent: "center",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          width: 340, padding: 28, borderRadius: 16,
          background: DS.glass.bgPanel, backdropFilter: DS.glass.blurPanel,
          WebkitBackdropFilter: DS.glass.blurPanel,
          border: `1px solid ${DS.glass.panelBorder}`,
          boxShadow: DS.glass.shadow,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 22 }}>
          <div
            style={{
              width: 36, height: 36, borderRadius: 9,
              background: "linear-gradient(135deg, #6BAEFF, #9F6FED)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16, fontWeight: 800, color: "#fff",
              boxShadow: "0 4px 16px rgba(107,174,255,0.35)",
            }}
          >
            M
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: DS.color.textPri, letterSpacing: -0.3 }}>
              MuleRadar
            </div>
            <div style={{ fontSize: 10, color: DS.color.textSec, letterSpacing: 1, textTransform: "uppercase" }}>
              Fraud Intelligence
            </div>
          </div>
        </div>

        <label style={{ fontSize: 12, color: DS.color.textSec, marginBottom: 5, display: "block" }}>
          Username
        </label>
        <input
          autoFocus
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={{ ...getInputStyle({ padding: "9px 12px", width: "100%" }), marginBottom: 14, boxSizing: "border-box" }}
        />

        <label style={{ fontSize: 12, color: DS.color.textSec, marginBottom: 5, display: "block" }}>
          Password
        </label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ ...getInputStyle({ padding: "9px 12px", width: "100%" }), marginBottom: 18, boxSizing: "border-box" }}
        />

        {error && (
          <div style={{ fontSize: 12, color: DS.color.riskHigh, marginBottom: 14 }}>{error}</div>
        )}

        <button
          type="submit"
          disabled={loading || !username || !password}
          style={{
            width: "100%", padding: "10px 0", borderRadius: 9, border: "none",
            background: "linear-gradient(135deg, #6BAEFF, #9F6FED)",
            color: "#fff", fontSize: 13, fontWeight: 700,
            cursor: loading || !username || !password ? "not-allowed" : "pointer",
            opacity: loading || !username || !password ? 0.6 : 1,
          }}
        >
          {loading ? "Masuk..." : "Masuk"}
        </button>
      </form>
    </div>
  );
}
