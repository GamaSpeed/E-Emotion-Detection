import { useState, useEffect, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL  = import.meta.env.VITE_WS_URL  || "ws://localhost:8000";
const POLL_MS = 3000; // polling REST si pas de WS dédié prof

const STATES = {
  Engagement:  { color: "#22D3A5", emoji: "🎯", label: "Eng."   },
  Boredom:     { color: "#6B7FD4", emoji: "😴", label: "Ennui"  },
  Confusion:   { color: "#F59E3F", emoji: "🤔", label: "Conf."  },
  Frustration: { color: "#EF4B6C", emoji: "😤", label: "Frus."  },
};

// ── Mock pour démonstration (remplacé par données WebSocket en production) ──
function mockStudents() {
  const names = [
    "Alice Martin", "Baptiste Leroy", "Camille Dubois", "Dylan Moreau",
    "Emma Bernard", "Félix Petit", "Gabrielle Simon", "Hugo Laurent",
    "Inès Thomas", "Julien Robert",
  ];
  return names.map((name, i) => ({
    id: `student_${i}`, name,
    online: Math.random() > 0.25,
    engScore: Math.floor(Math.random() * 100),
    predictions: Object.fromEntries(
      Object.keys(STATES).map(k => [k, { level: Math.random() > 0.5 ? 1 : 0, confidence: 0.5 + Math.random() * 0.5 }])
    ),
    updatedAt: Date.now() - Math.floor(Math.random() * 20000),
  }));
}

function isCritical(s) {
  return s.online && (
    s.predictions.Frustration?.level === 1 ||
    s.predictions.Confusion?.level === 1 ||
    s.engScore < 25
  );
}

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
    <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
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

function ClassDonut({ score }) {
  const color = score >= 60 ? "#22D3A5" : score >= 30 ? "#F59E3F" : "#EF4B6C";
  const r = 28, circ = 2 * Math.PI * r;
  const fill = (score / 100) * circ;
  return (
    <svg width={70} height={70} viewBox="0 0 70 70">
      <circle cx={35} cy={35} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={7} />
      <circle cx={35} cy={35} r={r} fill="none" stroke={color} strokeWidth={7}
        strokeDasharray={`${fill} ${circ - fill}`} strokeLinecap="round"
        transform="rotate(-90 35 35)"
        style={{ transition: "stroke-dasharray .8s cubic-bezier(.4,2,.6,1)" }} />
      <text x={35} y={38} textAnchor="middle" fontSize={13} fontWeight={800} fill="#fff" fontFamily="monospace">{score}%</text>
    </svg>
  );
}

export default function TeacherView({ user, onLogout }) {
  const [students,   setStudents]   = useState(() => mockStudents());
  const [filter,     setFilter]     = useState("all");   // all | critical | engaged
  const [search,     setSearch]     = useState("");
  const [selected,   setSelected]   = useState(null);
  const [wsStatus,   setWsStatus]   = useState("mock");  // mock | live | error

  // ── Simulation mise à jour temps réel (mock) ──────────────────────────────
  // En production, remplacer par un WebSocket vers /ws/teacher ou polling /api/students
  useEffect(() => {
    const t = setInterval(() => {
      setStudents(prev => prev.map(s => {
        if (!s.online) return s;
        const newPreds = {};
        Object.keys(STATES).forEach(k => {
          const cur   = s.predictions[k].level;
          const flip  = Math.random() < 0.1;
          newPreds[k] = { level: flip ? 1 - cur : cur, confidence: 0.5 + Math.random() * 0.5 };
        });
        return {
          ...s, predictions: newPreds, updatedAt: Date.now(),
          engScore: Math.max(0, Math.min(100, s.engScore + Math.round(Math.random() * 8 - 4))),
        };
      }));
    }, POLL_MS);
    return () => clearInterval(t);
  }, []);

  // ── Tentative connexion WebSocket ─────────────────────────────────────────
  // Le back n'expose pas encore /ws/teacher — décommenté quand disponible
  /*
  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/teacher`);
    ws.onopen  = () => setWsStatus("live");
    ws.onerror = () => setWsStatus("error");
    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.students) setStudents(data.students);
    };
    return () => ws.close();
  }, []);
  */

  const filtered = students
    .filter(s => {
      if (search && !s.name.toLowerCase().includes(search.toLowerCase())) return false;
      if (filter === "critical") return isCritical(s);
      if (filter === "engaged")  return s.online && s.engScore >= 60;
      return true;
    })
    .sort((a, b) => b.engScore - a.engScore);

  const online     = students.filter(s => s.online);
  const classScore = online.length ? Math.round(online.reduce((sum, s) => sum + s.engScore, 0) / online.length) : 0;
  const critCount  = students.filter(isCritical).length;
  const classColor = classScore >= 60 ? "#22D3A5" : classScore >= 30 ? "#F59E3F" : "#EF4B6C";

  const fmtAge = ms => {
    const s = Math.floor((Date.now() - ms) / 1000);
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}min`;
  };

  const selStudent = students.find(s => s.id === selected);

  return (
    <div style={{ minHeight: "100vh", background: "#06090F", color: "#fff", fontFamily: "'DM Sans','Segoe UI',sans-serif" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes up    { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
        .row:hover { background:rgba(255,255,255,0.035) !important; cursor:pointer; }
        ::-webkit-scrollbar { width:4px } ::-webkit-scrollbar-thumb { background:rgba(255,255,255,.1); border-radius:2px }
        .filter-btn { padding:5px 14px; border-radius:20px; border:1px solid rgba(255,255,255,0.1);
          background:transparent; color:rgba(255,255,255,0.45); font-size:12px; cursor:pointer;
          transition:all .15s; font-family:inherit; }
        .filter-btn.active { background:rgba(34,211,165,0.12); border-color:rgba(34,211,165,0.3); color:#22D3A5; }
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
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {wsStatus === "live" && (
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#22D3A5", animation: "pulse 2s infinite" }} />
              <span style={{ fontSize: 10, color: "#22D3A5", fontWeight: 700, letterSpacing: 1 }}>LIVE</span>
            </div>
          )}
          <span style={{ fontSize: 13, color: "rgba(255,255,255,0.4)" }}>👨‍🏫 {user?.name}</span>
          <button onClick={onLogout} style={{ padding: "5px 12px", borderRadius: 7, border: "1px solid rgba(255,255,255,0.1)", background: "transparent", color: "rgba(255,255,255,0.35)", fontSize: 12, cursor: "pointer" }}>
            Déconnexion
          </button>
        </div>
      </header>

      <main style={{ padding: "20px 24px", animation: "up .35s ease" }}>

        {/* KPIs */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
          {[
            { label: "Connectés",  value: `${online.length}/${students.length}`, color: "#22D3A5" },
            { label: "Eng. classe",value: `${classScore}%`,                      color: classColor },
            { label: "En difficulté",value: critCount,                            color: critCount ? "#EF4B6C" : "#555" },
            { label: "Très engagés",value: students.filter(s => s.online && s.engScore >= 70).length, color: "#6B7FD4" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: "14px 16px" }}>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "monospace" }}>{value}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: selStudent ? "1fr 280px" : "1fr", gap: 16 }}>

          {/* Table */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, overflow: "hidden" }}>

            {/* Toolbar */}
            <div style={{ padding: "14px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <input value={search} onChange={e => setSearch(e.target.value)}
                placeholder="Rechercher..."
                style={{ padding: "6px 12px", borderRadius: 8, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "#fff", fontSize: 13, outline: "none", fontFamily: "inherit", width: 180 }} />
              <div style={{ display: "flex", gap: 6 }}>
                {[["all", "Tous"], ["critical", "⚠️ Difficulté"], ["engaged", "🔥 Engagés"]].map(([v, l]) => (
                  <button key={v} className={`filter-btn${filter === v ? " active" : ""}`} onClick={() => setFilter(v)}>{l}</button>
                ))}
              </div>
              <span style={{ marginLeft: "auto", fontSize: 11, color: "rgba(255,255,255,0.25)" }}>
                {filtered.length} étudiant{filtered.length > 1 ? "s" : ""}
              </span>
            </div>

            {/* En-tête colonnes */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 1fr", padding: "8px 16px", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
              {["Étudiant", "Engagement", ...Object.values(STATES).map(s => s.emoji + " " + s.label)].map(h => (
                <div key={h} style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: 0.8, textTransform: "uppercase" }}>{h}</div>
              ))}
            </div>

            {/* Lignes */}
            <div style={{ maxHeight: "calc(100vh - 320px)", overflowY: "auto" }}>
              {filtered.length === 0
                ? <div style={{ padding: 32, textAlign: "center", color: "rgba(255,255,255,0.2)", fontSize: 13 }}>Aucun étudiant</div>
                : filtered.map(s => {
                  const crit = isCritical(s);
                  const isSel = s.id === selected;
                  return (
                    <div key={s.id} className="row"
                      onClick={() => setSelected(isSel ? null : s.id)}
                      style={{
                        display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 1fr",
                        padding: "10px 16px", alignItems: "center",
                        borderBottom: "1px solid rgba(255,255,255,0.03)",
                        background: isSel ? "rgba(34,211,165,0.05)" : crit ? "rgba(239,75,108,0.02)" : "transparent",
                        borderLeft: `2px solid ${isSel ? "#22D3A5" : crit ? "rgba(239,75,108,0.3)" : "transparent"}`,
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
                        {s.online ? (
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
                              <div style={{ height: "100%", background: s.engScore >= 60 ? "#22D3A5" : s.engScore >= 30 ? "#F59E3F" : "#EF4B6C", width: `${s.engScore}%`, transition: "width .7s", borderRadius: 2 }} />
                            </div>
                            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontFamily: "monospace", minWidth: 28 }}>{s.engScore}%</span>
                          </div>
                        ) : <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>—</span>}
                      </div>

                      {/* 4 états binaires */}
                      {Object.entries(STATES).map(([k, { color }]) => (
                        <div key={k}>
                          {s.online
                            ? <LevelDots level={s.predictions[k]?.level ?? 0} color={color} />
                            : <span style={{ fontSize: 11, color: "rgba(255,255,255,0.15)" }}>—</span>
                          }
                        </div>
                      ))}
                    </div>
                  );
                })
              }
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

              {/* Donut score */}
              <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16, padding: "12px", background: "rgba(255,255,255,0.02)", borderRadius: 10 }}>
                <ClassDonut score={selStudent.engScore} />
                <div>
                  <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginBottom: 3 }}>Score d'engagement</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: selStudent.engScore >= 60 ? "#22D3A5" : selStudent.engScore >= 30 ? "#F59E3F" : "#EF4B6C" }}>
                    {selStudent.engScore >= 60 ? "Très engagé 🔥" : selStudent.engScore >= 30 ? "Modéré 💡" : "Faible 😴"}
                  </div>
                </div>
              </div>

              {/* Détail 4 états */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 14 }}>
                {Object.entries(STATES).map(([k, { color, emoji, label }]) => {
                  const pred = selStudent.predictions[k];
                  const isHigh = pred?.level === 1;
                  return (
                    <div key={k}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                        <span style={{ fontSize: 12, color: "rgba(255,255,255,0.55)" }}>{emoji} {label}</span>
                        <span style={{ fontSize: 12, fontWeight: 700, color: isHigh ? color : "rgba(255,255,255,0.3)" }}>
                          {isHigh ? "Élevé" : "Faible"}
                        </span>
                      </div>
                      <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
                        <div style={{ height: "100%", background: color, width: isHigh ? "100%" : "15%", transition: "width .5s", borderRadius: 2 }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Alerte critique */}
              {isCritical(selStudent) && (
                <div style={{ padding: "10px 12px", borderRadius: 10, background: "rgba(239,75,108,0.08)", border: "1px solid rgba(239,75,108,0.2)" }}>
                  <div style={{ fontSize: 12, color: "#EF4B6C", fontWeight: 700, marginBottom: 3 }}>⚠️ Intervention recommandée</div>
                  <div style={{ fontSize: 11, color: "rgba(239,75,108,0.7)", lineHeight: 1.6 }}>
                    {selStudent.predictions.Frustration?.level === 1 && "Frustration élevée. "}
                    {selStudent.predictions.Confusion?.level === 1   && "Confusion détectée. "}
                    {selStudent.engScore < 25 && "Engagement très faible. "}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
