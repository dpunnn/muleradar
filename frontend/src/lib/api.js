import axios from "axios";

// Backend FastAPI base URL — override via .env (VITE_API_BASE_URL) saat deploy.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const TOKEN_KEY = "mr_token";
export const getToken = () => {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
};
export const setToken = (token) => {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* ignore (private browsing dsb) */
  }
};
export const clearToken = () => {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
};

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
});

// Fix (17-Jul, Prioritas 2 — autentikasi): semua endpoint selain /auth/login
// sekarang butuh Bearer JWT (lihat backend/api/auth.py). Interceptor ini
// nempelin token dari localStorage ke tiap request otomatis — komponen
// pemanggil (Dashboard, AlertList, dst) TIDAK perlu tau soal token sama
// sekali, tetap panggil getDashboardOverview() dst seperti biasa.
api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// 401 (token tak ada/kedaluwarsa) -> paksa balik ke /login. Reload penuh
// (bukan navigate() react-router) krn interceptor ini di luar komponen React,
// tak punya akses ke instance navigate — reload jg sekalian bersihin state.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err?.response?.status === 401 && window.location.pathname !== "/login") {
      clearToken();
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

// ── Auth ─────────────────────────────────────────────────────
export const login = (username, password) =>
  api.post("/auth/login", { username, password }).then((r) => r.data);

// ── Dashboard ────────────────────────────────────────────────
export const getDashboardOverview = () => api.get("/dashboard/overview").then(r => r.data);
export const getTypologyBreakdown = () => api.get("/dashboard/typology-breakdown").then(r => r.data);
export const getStatusDistribution = () => api.get("/dashboard/status-distribution").then(r => r.data);
export const getRiskTrend = (days = 30) => api.get("/dashboard/risk-trend", { params: { days } }).then(r => r.data);
export const getHeatmap = () => api.get("/dashboard/heatmap").then(r => r.data);
export const getTopAccounts = (limit = 10) => api.get("/dashboard/top-accounts", { params: { limit } }).then(r => r.data);

// ── Alerts ───────────────────────────────────────────────────
export const listAlerts = (params = {}) => api.get("/alerts", { params }).then(r => r.data);
export const getAlertDetail = (alertId) => api.get(`/alerts/${alertId}`).then(r => r.data);
export const updateAlertStatus = (alertId, status) => api.patch(`/alerts/${alertId}/status`, { status }).then(r => r.data);
export const assignAlert = (alertId, assignedTo, notes) => api.post(`/alerts/${alertId}/assign`, { assigned_to: assignedTo, notes }).then(r => r.data);

// ── Graph ────────────────────────────────────────────────────
export const getGraphOverview = () => api.get("/graph/overview").then(r => r.data);
export const listClusters = (minSize = 2) => api.get("/graph/clusters", { params: { min_size: minSize } }).then(r => r.data);
export const getClusterDetail = (clusterId, minSize = 2) => api.get(`/graph/cluster/${clusterId}`, { params: { min_size: minSize } }).then(r => r.data);
export const getNodeNeighbors = (accountId, hops = 1, limit = 50) => api.get(`/graph/node/${accountId}/neighbors`, { params: { hops, limit } }).then(r => r.data);
export const getNodePPR = (accountId, topK = 20) => api.get(`/graph/node/${accountId}/ppr`, { params: { top_k: topK } }).then(r => r.data);
export const getNodeFlags = (accountId) => api.get(`/graph/node/${accountId}/flags`).then(r => r.data);

// ── Cases ────────────────────────────────────────────────────
export const createCase = (alertId, assignedTo, notes) => api.post("/cases", { alert_id: alertId, assigned_to: assignedTo, notes }).then(r => r.data);
export const listCases = (params = {}) => api.get("/cases", { params }).then(r => r.data);
export const getCaseDetail = (caseId) => api.get(`/cases/${caseId}`).then(r => r.data);
export const updateCase = (caseId, body) => api.patch(`/cases/${caseId}`, body).then(r => r.data);
export const escalateCase = (caseId, note) => api.post(`/cases/${caseId}/escalate`, { note }).then(r => r.data);

// ── Copilot ──────────────────────────────────────────────────
export const getTypologyInfo = (name) => api.get(`/copilot/typology/${name}`).then(r => r.data);
export const generateSummary = (accountId) => api.post("/copilot/summary", { account_id: accountId }).then(r => r.data);
export const generateLtkm = (accountId) => api.post("/copilot/ltkm", { account_id: accountId }).then(r => r.data);
