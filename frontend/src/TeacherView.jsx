import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL  = import.meta.env.VITE_WS_URL  || "ws://localhost:8000";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const STATES = {
  Engagement:  { color: "#22D3A5", emoji: "🎯", label: "Engagement"  },
  Boredom:     { color: "#6B7FD4", emoji: "😴", label: "Ennui"       },
  Confusion:   { color: "#F59E3F", emoji: "🤔", label: "Confusion"   },
  Frustration: { color: "#EF4B6C", emoji: "😤", label: "Frustration" },
};

// ── Composants utilitaires ────────────────────────────────────────────────────
function StatusDot({ online }) {
  return (
    <div style={{
      width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
      background: online ? "#22D3A5" : "#2a2a3a",
      boxShadow: online ? "0 0 5px #22D3A5" : "none",
      animation: online ? "pulse 2s infinite" : "none",
    }} />
  );
}

function LevelDots({ level, color }) {
  return (
    <div style={{ display: "flex", gap: 3 }}>
      {[0, 1].map(i => (
        <div key={i} style={{
          width: 7, height: 7, borderRadius: 2,
          background: i <= level ? color : "rgba(255,255,255,0.08)",
          transition: "background .4s",
        }} />
      ))}
    </div>
  );
}

function EngBar({ score }) {
  const color = score >= 60 ? "#22D3A5" : score >= 30 ? "#F59E3F" : "#EF4B6C";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
        <div style={{ height: "100%", background: color, width: `${score}%`, transition: "width .7s", borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: 11, color, fontFamily: "monospace", minWidth: 28 }}>{score}%</span>
    </div>
  );
}

function AlertBanner({ alerts, onAck }) {
  if (!alerts.length) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      {alerts.map(a => (
        <div key={a.id} style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "9px 14px", borderRadius: 10, marginBottom: 6,
          background: "rgba(239,75,108,0.1)", border: "1px solid rgba(239,75,108,0.25)",
          animation: "slideIn .3s ease",
        }}>
          <div>
            <span style={{ fontSize: 12, color: "#EF4B6C", fontWeight: 700 }}>⚠️ {a.studentName}</span>
            <span style={{ fontSize: 11, color: "rgba(239,75,108,0.7)", marginLeft: 8 }}>
              {a.state === "confusion" ? "Confusion" : "Frustration"} élevée · {a.duration}s
            </span>
          </div>
          <button onClick={() => onAck(a.id)} style={{
            padding: "3px 10px", borderRadius: 6, fontSize: 11,
            background: "rgba(239,75,108,0.2)", border: "1px solid rgba(239,75,108,0.3)",
            color: "#EF4B6C", cursor: "pointer", fontFamily: "inherit",
          }}>Acquitter</button>
        </div>
      ))}
    </div>
  );
}

// ── Vue Historique ────────────────────────────────────────────────────────────
function HistoryTab({ token }) {
  const [sessions, setSessions]   = useState([]);
  const [selected, setSelected]   = useState(null);
  const [history,  setHistory]    = useState(null);
  const [loading,  setLoading]    = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/sessions`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.json())
      .then(setSessions)
      .catch(console.error);
  }, [token]);

  const loadSession = async (id) => {
    setLoading(true);
    try {
      const r = await fetch(`${API_URL}/sessions/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await r.json();
      setHistory(data);
      setSelected(id);
    } finally {
      setLoading(false);
    }
  };

  const exportCsv = (id) => {
    window.open(`${API_URL}/sessions/${id}/export`, "_blank");
  };

  const fmtDate = (iso) => new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
  const durMin = (s) => {
    if (!s.ended_at) return "En cours";
    const d = Math.round((new Date(s.ended_at) - new Date(s.started_at)) / 60000);
    return `${d} min`;
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: selected ? "1fr 1.4fr" : "1fr", gap: 16 }}>
      {/* Liste des sessions */}
      <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, overflow: "hidden" }}>
        <div style={{ padding: "12px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", fontSize: 12, color: "rgba(255,255,255,0.4)", letterSpacing: 1, textTransform: "uppercase" }}>
          Sessions enregistrées
        </div>
        {sessions.length === 0
          ? <div style={{ padding: 32, textAlign: "center", color: "rgba(255,255,255,0.2)", fontSize: 13 }}>Aucune session</div>
          : sessions.map(s => (
            <div key={s.id} onClick={() => loadSession(s.id)} style={{
              padding: "11px 16px", borderBottom: "1px solid rgba(255,255,255,0.04)",
              cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center",
              background: selected === s.id ? "rgba(34,211,165,0.05)" : "transparent",
              borderLeft: `2px solid ${selected === s.id ? "#22D3A5" : "transparent"}`,
              transition: "background .15s",
            }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#fff", marginBottom: 2 }}>
                  Session {s.id.slice(0, 8)}
                </div>
                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)" }}>
                  {fmtDate(s.started_at)} · {durMin(s)} · {s.n_predictions} prédictions
                </div>
              </div>
              {s.avg_engagement_score != null && (
                <span style={{
                  fontSize: 12, fontWeight: 700, fontFamily: "monospace",
                  color: s.avg_engagement_score >= 60 ? "#22D3A5" : s.avg_engagement_score >= 30 ? "#F59E3F" : "#EF4B6C",
                }}>
                  {Math.round(s.avg_engagement_score)}%
                </span>
              )}
            </div>
          ))}
      </div>

      {/* Détail session */}
      {selected && history && (
        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, overflow: "hidden" }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", letterSpacing: 1, textTransform: "uppercase" }}>
              Détail — {history.session.n_predictions} prédictions · {history.alerts.length} alertes
            </span>
            <button onClick={() => exportCsv(selected)} style={{
              padding: "4px 12px", borderRadius: 7, fontSize: 11, fontFamily: "inherit",
              background: "rgba(34,211,165,0.1)", border: "1px solid rgba(34,211,165,0.3)",
              color: "#22D3A5", cursor: "pointer",
            }}>↓ Export CSV</button>
          </div>

          {loading
            ? <div style={{ padding: 32, textAlign: "center", color: "rgba(255,255,255,0.3)" }}>Chargement...</div>
            : (
              <div style={{ padding: 16, maxHeight: 400, overflowY: "auto" }}>
                {/* Alertes de la session */}
                {history.alerts.length > 0 && (
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Alertes</div>
                    {history.alerts.map(a => (
                      <div key={a.id} style={{ fontSize: 11, color: "#EF4B6C", padding: "4px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                        {new Date(a.timestamp).toLocaleTimeString("fr-FR")} · {a.state} · {a.acknowledged ? "✓ Acquitté" : "Non acquitté"}
                      </div>
                    ))}
                  </div>
                )}

                {/* Dernières prédictions */}
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
                  Dernières prédictions
                </div>
                {history.predictions.slice(-20).reverse().map(p => (
                  <div key={p.id} style={{ display: "grid", gridTemplateColumns: "80px repeat(4,1fr)", gap: 6, padding: "5px 0", borderBottom: "1px solid rgba(255,255,255,0.03)", fontSize: 10, fontFamily: "monospace" }}>
                    <span style={{ color: "rgba(255,255,255,0.3)" }}>{new Date(p.timestamp).toLocaleTimeString("fr-FR")}</span>
                    {["Engagement","Boredom","Confusion","Frustration"].map(k => {
                      const val = p[k.toLowerCase()];
                      const { color } = STATES[k];
                      return <span key={k} style={{ color: val === 1 ? color : "rgba(255,255,255,0.2)" }}>{k.slice(0,3)} {val === 1 ? "H" : "L"}</span>;
                    })}
                  </div>
                ))}
              </div>
            )}
        </div>
      )}
    </div>
  );
}

// ── Vue principale ────────────────────────────────────────────────────────────
export default function TeacherView({ user, onLogout }) {
  const wsRef      = useRef(null);
  const [wsStatus, setWsStatus] = useState("connecting");
  const [tab,      setTab]      = useState("live");        // "live" | "history"
  const [students, setStudents] = useState({});            // {client_id: studentData}
  const [alerts,   setAlerts]   = useState([]);            // alertes non acquittées
  const [search,   setSearch]   = useState("");
  const [filter,   setFilter]   = useState("all");
  const [selected, setSelected] = useState(null);

  // Token JWT depuis le localStorage (mis par LoginPage après /auth/login)
  const token = localStorage.getItem("edusense_token") || "";

  // ── Connexion WebSocket prof ──────────────────────────────────────────────
  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/teacher`);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus("live");
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);

      if (msg.type === "ping") return;

      if (msg.type === "class_state") {
        // État initial : marquer les étudiants déjà connectés
        msg.online.forEach(clientId => {
          setStudents(prev => ({
            ...prev,
            [clientId]: prev[clientId] || {
              clientId, name: clientId, online: true,
              engScore: 0, predictions: null, updatedAt: Date.now(), sessionId: null,
            },
          }));
        });
      }

      if (msg.type === "student_connected") {
        setStudents(prev => ({
          ...prev,
          [msg.client_id]: {
            clientId: msg.client_id,
            name: msg.client_id,
            online: true, engScore: 0, predictions: null,
            updatedAt: Date.now(), sessionId: msg.session_id,
          },
        }));
      }

      if (msg.type === "student_disconnected") {
        setStudents(prev => {
          const next = { ...prev };
          if (next[msg.client_id]) {
            next[msg.client_id] = { ...next[msg.client_id], online: false, updatedAt: Date.now() };
          }
          return next;
        });
      }

      if (msg.type === "prediction") {
        setStudents(prev => ({
          ...prev,
          [msg.client_id]: {
            ...prev[msg.client_id],
            clientId: msg.client_id,
            online: true,
            engScore: msg.engagement_score,
            predictions: msg.predictions,
            updatedAt: Date.now(),
            sessionId: msg.session_id,
          },
        }));

        // Nouvelles alertes
        if (msg.new_alerts?.length) {
          const studentName = students[msg.client_id]?.name || msg.client_id;
          const newAlerts = msg.new_alerts.map(a => ({
            ...a, studentName, clientId: msg.client_id,
          }));
          setAlerts(prev => [...prev, ...newAlerts]);
        }
      }
    };

    ws.onerror = () => setWsStatus("error");
    ws.onclose = () => setWsStatus("disconnected");

    return () => ws.close();
  }, []);

  // ── Acquittement d'alerte ─────────────────────────────────────────────────
  const ackAlert = useCallback((alertId) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "ack_alert", alert_id: alertId }));
    }
    setAlerts(prev => prev.filter(a => a.id !== alertId));
  }, []);

  // ── Dérivations ───────────────────────────────────────────────────────────
  const studentList = Object.values(students);
  const online      = studentList.filter(s => s.online);
  const classScore  = online.length
    ? Math.round(online.reduce((sum, s) => sum + s.engScore, 0) / online.length)
    : 0;
  const classColor  = classScore >= 60 ? "#22D3A5" : classScore >= 30 ? "#F59E3F" : "#EF4B6C";

  const isCritical = s => s.online && s.predictions && (
    s.predictions.Confusion?.level === 1 ||
    s.predictions.Frustration?.level === 1 ||
    s.engScore < 25
  );

  const filtered = studentList
    .filter(s => {
      if (search && !s.clientId.toLowerCase().includes(search.toLowerCase()) &&
          !s.name.toLowerCase().includes(search.toLowerCase())) return false;
      if (filter === "critical") return isCritical(s);
      if (filter === "engaged")  return s.online && s.engScore >= 60;
      return true;
    })
    .sort((a, b) => b.engScore - a.engScore);

  const selStudent = selected ? students[selected] : null;

  const fmtAge = ms => {
    const s = Math.floor((Date.now() - ms) / 1000);
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}min`;
  };

  const wsCfg = {
    live:          { color: "#22D3A5", dot: true,  text: "En direct"      },
    connecting:    { color: "#6B7FD4", dot: true,  text: "Connexion..."    },
    disconnected:  { color: "#EF4B6C", dot: false, text: "Déconnecté"     },
    error:         { color: "#EF4B6C", dot: false, text: "Erreur WS"      },
  }[wsStatus] || { color: "#555", dot: false, text: wsStatus };

  return (
    <div style={{ minHeight: "100vh", background: "#06090F", color: "#fff", fontFamily: "'DM Sans','Segoe UI',sans-serif" }}>
      <style>{`
        @keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes up      { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
        @keyframes slideIn { from{opacity:0;transform:translateX(10px)} to{opacity:1;transform:none} }
        .row:hover { background:rgba(255,255,255,0.035) !important; cursor:pointer; }
        ::-webkit-scrollbar { width:4px } ::-webkit-scrollbar-thumb { background:rgba(255,255,255,.1);border-radius:2px }
      `}</style>

      {/* Header */}
      <header style={{
        padding: "0 24px", height: 54, display: "flex", alignItems: "center",
        justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(6,9,15,0.95)", backdropFilter: "blur(12px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: "linear-gradient(135deg,#22D3A5,#6B7FD4)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13 }}>🧠</div>
          <span style={{ fontWeight: 700, fontSize: 14 }}>Edu<span style={{ color: "#22D3A5" }}>Sense</span></span>
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", marginLeft: 4 }}>Tableau de bord</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Status WS */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 10px", borderRadius: 20, background: `${wsCfg.color}18`, border: `1px solid ${wsCfg.color}44` }}>
            {wsCfg.dot && <div style={{ width: 5, height: 5, borderRadius: "50%", background: wsCfg.color, animation: "pulse 2s infinite" }} />}
            <span style={{ fontSize: 10, color: wsCfg.color, fontWeight: 700, letterSpacing: 1 }}>{wsCfg.text}</span>
          </div>
          <span style={{ fontSize: 13, color: "rgba(255,255,255,0.4)" }}>👨‍🏫 {user?.name}</span>
          <button onClick={onLogout} style={{ padding: "5px 12px", borderRadius: 7, border: "1px solid rgba(255,255,255,0.1)", background: "transparent", color: "rgba(255,255,255,0.35)", fontSize: 12, cursor: "pointer" }}>
            Déconnexion
          </button>
        </div>
      </header>

      <main style={{ padding: "20px 24px", animation: "up .35s ease" }}>

        {/* Onglets */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, padding: 4, background: "rgba(255,255,255,0.04)", borderRadius: 10, width: "fit-content" }}>
          {[["live", "📡 En direct"], ["history", "📂 Historique"]].map(([v, l]) => (
            <button key={v} onClick={() => setTab(v)} style={{
              padding: "6px 18px", borderRadius: 7, border: "none", cursor: "pointer", fontFamily: "inherit",
              background: tab === v ? "rgba(34,211,165,0.15)" : "transparent",
              color: tab === v ? "#22D3A5" : "rgba(255,255,255,0.4)",
              fontWeight: 600, fontSize: 12, transition: "all .2s",
            }}>{l}</button>
          ))}
        </div>

        {tab === "history" && <HistoryTab token={token} />}

        {tab === "live" && (
          <>
            {/* KPIs */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 16 }}>
              {[
                { label: "Connectés",    value: `${online.length}/${studentList.length}`, color: "#22D3A5" },
                { label: "Eng. classe",  value: `${classScore}%`,                         color: classColor },
                { label: "Alertes",      value: alerts.length,                             color: alerts.length ? "#EF4B6C" : "#555" },
                { label: "Très engagés", value: online.filter(s => s.engScore >= 70).length, color: "#6B7FD4" },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: "14px 16px" }}>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "monospace" }}>{value}</div>
                </div>
              ))}
            </div>

            {/* Alertes */}
            <AlertBanner alerts={alerts} onAck={ackAlert} />

            <div style={{ display: "grid", gridTemplateColumns: selStudent ? "1fr 280px" : "1fr", gap: 16 }}>
              {/* Table étudiants */}
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, overflow: "hidden" }}>
                {/* Toolbar */}
                <div style={{ padding: "12px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                  <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Rechercher..."
                    style={{ padding: "6px 12px", borderRadius: 8, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "#fff", fontSize: 13, outline: "none", fontFamily: "inherit", width: 160 }} />
                  <div style={{ display: "flex", gap: 5 }}>
                    {[["all","Tous"], ["critical","⚠️ Difficulté"], ["engaged","🔥 Engagés"]].map(([v, l]) => (
                      <button key={v} onClick={() => setFilter(v)} style={{
                        padding: "4px 12px", borderRadius: 20, border: `1px solid ${filter===v ? "rgba(34,211,165,0.3)" : "rgba(255,255,255,0.1)"}`,
                        background: filter===v ? "rgba(34,211,165,0.1)" : "transparent",
                        color: filter===v ? "#22D3A5" : "rgba(255,255,255,0.4)",
                        fontSize: 12, cursor: "pointer", fontFamily: "inherit", transition: "all .15s",
                      }}>{l}</button>
                    ))}
                  </div>
                  <span style={{ marginLeft: "auto", fontSize: 11, color: "rgba(255,255,255,0.25)" }}>
                    {filtered.length} étudiant{filtered.length > 1 ? "s" : ""}
                  </span>
                </div>

                {/* En-têtes */}
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 1fr", padding: "8px 16px", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                  {["Étudiant", "Engagement", ...Object.values(STATES).map(s => s.emoji + " " + s.label)].map(h => (
                    <div key={h} style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: 0.8, textTransform: "uppercase" }}>{h}</div>
                  ))}
                </div>

                {/* Lignes */}
                <div style={{ maxHeight: "calc(100vh - 380px)", overflowY: "auto" }}>
                  {studentList.length === 0
                    ? (
                      <div style={{ padding: 40, textAlign: "center", color: "rgba(255,255,255,0.2)", fontSize: 13 }}>
                        {wsStatus === "live" ? "En attente de connexions étudiants..." : "Connexion au serveur..."}
                      </div>
                    )
                    : filtered.length === 0
                      ? <div style={{ padding: 32, textAlign: "center", color: "rgba(255,255,255,0.2)", fontSize: 13 }}>Aucun étudiant</div>
                      : filtered.map(s => {
                        const crit  = isCritical(s);
                        const isSel = s.clientId === selected;
                        return (
                          <div key={s.clientId} className="row"
                            onClick={() => setSelected(isSel ? null : s.clientId)}
                            style={{
                              display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 1fr",
                              padding: "10px 16px", alignItems: "center",
                              borderBottom: "1px solid rgba(255,255,255,0.03)",
                              background: isSel ? "rgba(34,211,165,0.05)" : crit ? "rgba(239,75,108,0.02)" : "transparent",
                              borderLeft: `2px solid ${isSel ? "#22D3A5" : crit ? "rgba(239,75,108,0.35)" : "transparent"}`,
                              transition: "background .15s",
                            }}>

                            {/* Nom */}
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <StatusDot online={s.online} />
                              <div>
                                <div style={{ fontSize: 13, fontWeight: 600, display: "flex", alignItems: "center", gap: 5 }}>
                                  {s.name}
                                  {crit && <span style={{ fontSize: 10, padding: "1px 5px", borderRadius: 5, background: "rgba(239,75,108,0.2)", color: "#EF4B6C", fontWeight: 700 }}>!</span>}
                                </div>
                                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.2)", marginTop: 1 }}>
                                  {s.online ? `il y a ${fmtAge(s.updatedAt)}` : "Hors ligne"}
                                </div>
                              </div>
                            </div>

                            {/* Score engagement */}
                            <div>
                              {s.online && s.predictions
                                ? <EngBar score={s.engScore} />
                                : <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>—</span>}
                            </div>

                            {/* 4 états binaires */}
                            {Object.entries(STATES).map(([k, { color }]) => (
                              <div key={k}>
                                {s.online && s.predictions
                                  ? <LevelDots level={s.predictions[k]?.level ?? 0} color={color} />
                                  : <span style={{ fontSize: 11, color: "rgba(255,255,255,0.15)" }}>—</span>}
                              </div>
                            ))}
                          </div>
                        );
                      })}
                </div>

                <div style={{ padding: "10px 16px", borderTop: "1px solid rgba(255,255,255,0.04)", fontSize: 11, color: "rgba(255,255,255,0.2)", textAlign: "right" }}>
                  {wsStatus === "live" ? "Données en temps réel via WebSocket" : `Statut : ${wsStatus}`}
                </div>
              </div>

              {/* Panneau détail */}
              {selStudent && (
                <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, padding: "18px", animation: "up .2s ease", alignSelf: "start" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <StatusDot online={selStudent.online} />
                      <span style={{ fontWeight: 700, fontSize: 14 }}>{selStudent.name}</span>
                    </div>
                    <button onClick={() => setSelected(null)} style={{ width: 26, height: 26, borderRadius: 7, background: "rgba(255,255,255,0.06)", border: "none", color: "rgba(255,255,255,0.4)", cursor: "pointer", fontSize: 13 }}>✕</button>
                  </div>

                  {/* Score */}
                  <div style={{ padding: "14px", background: "rgba(255,255,255,0.02)", borderRadius: 10, marginBottom: 14 }}>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginBottom: 6 }}>Score d'engagement</div>
                    <div style={{ fontSize: 28, fontWeight: 800, fontFamily: "monospace", color: selStudent.engScore >= 60 ? "#22D3A5" : selStudent.engScore >= 30 ? "#F59E3F" : "#EF4B6C" }}>
                      {selStudent.engScore}%
                    </div>
                  </div>

                  {/* Détail 4 états */}
                  {selStudent.predictions && (
                    <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 14 }}>
                      {Object.entries(STATES).map(([k, { color, emoji, label }]) => {
                        const pred   = selStudent.predictions[k];
                        const isHigh = pred?.level === 1;
                        return (
                          <div key={k}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ fontSize: 12, color: "rgba(255,255,255,0.55)" }}>{emoji} {label}</span>
                              <span style={{ fontSize: 12, fontWeight: 700, color: isHigh ? color : "rgba(255,255,255,0.3)" }}>
                                {isHigh ? "Élevé" : "Faible"}
                              </span>
                            </div>
                            <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
                              <div style={{ height: "100%", background: color, width: isHigh ? "100%" : "12%", transition: "width .5s", borderRadius: 2 }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Alerte critique */}
                  {isCritical(selStudent) && (
                    <div style={{ padding: "10px 12px", borderRadius: 10, background: "rgba(239,75,108,0.08)", border: "1px solid rgba(239,75,108,0.2)" }}>
                      <div style={{ fontSize: 12, color: "#EF4B6C", fontWeight: 700, marginBottom: 3 }}>⚠️ Intervention recommandée</div>
                      <div style={{ fontSize: 11, color: "rgba(239,75,108,0.7)", lineHeight: 1.6 }}>
                        {selStudent.predictions?.Frustration?.level === 1 && "Frustration élevée. "}
                        {selStudent.predictions?.Confusion?.level === 1   && "Confusion détectée. "}
                        {selStudent.engScore < 25 && "Engagement très faible. "}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
