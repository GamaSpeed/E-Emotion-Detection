import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || 
  (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host;
const FRAME_MS  = 500;

// Mode binaire : Low=0, High=1
const STATES = {
  Engagement:  { color: "#22D3A5", emoji: "🎯", label: "Engagement"  },
  Boredom:     { color: "#6B7FD4", emoji: "😴", label: "Ennui"       },
  Confusion:   { color: "#F59E3F", emoji: "🤔", label: "Confusion"   },
  Frustration: { color: "#EF4B6C", emoji: "😤", label: "Frustration" },
};

function BinaryGauge({ state, prediction }) {
  const { color, emoji, label } = STATES[state];
  const isHigh   = prediction?.level === 1;
  const conf     = prediction?.confidence ?? 0;

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)", border: `1px solid ${isHigh ? color + "33" : "rgba(255,255,255,0.07)"}`,
      borderRadius: 14, padding: "16px 14px",
      display: "flex", flexDirection: "column", alignItems: "center", gap: 10,
      transition: "border-color .4s",
    }}>
      <div style={{ fontSize: 26 }}>{emoji}</div>
      <div style={{
        fontSize: 13, fontWeight: 700,
        color: isHigh ? color : "rgba(255,255,255,0.3)",
        transition: "color .4s",
      }}>
        {isHigh ? "Élevé" : "Faible"}
      </div>
      <div style={{ width: "100%", height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
        <div style={{
          height: "100%", borderRadius: 2, background: color,
          width: `${conf * 100}%`, transition: "width .5s",
        }} />
      </div>
      <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: 1, textTransform: "uppercase" }}>
        {label}
      </div>
    </div>
  );
}

function Sparkline({ data, color }) {
  if (data.length < 2) return <div style={{ height: 28 }} />;
  const W = 100, H = 28;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - v * (H - 4) - 2;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: H }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5}
        strokeLinecap="round" strokeLinejoin="round" opacity={0.7} />
    </svg>
  );
}

export default function StudentView({ user, onLogout }) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const wsRef       = useRef(null);
  const timerRef    = useRef(null);
  const clientId    = useRef(`student_${Date.now()}`);

  const [status,       setStatus]       = useState("idle");      // idle | connecting | connected | buffering | error
  const [isStreaming,  setIsStreaming]   = useState(false);
  const [predictions,  setPredictions]  = useState(null);
  const [engScore,     setEngScore]     = useState(0);
  const [sessionTime,  setSessionTime]  = useState(0);
  const [bufferPct,    setBufferPct]    = useState(0);
  const [history,      setHistory]      = useState(
    Object.fromEntries(Object.keys(STATES).map(k => [k, []]))
  );

  // Timer session
  useEffect(() => {
    if (!isStreaming) return;
    const t = setInterval(() => setSessionTime(s => s + 1), 1000);
    return () => clearInterval(t);
  }, [isStreaming]);

  const fmt = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  const startSession = useCallback(async () => {
    // Démarrer webcam
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
    } catch {
      setStatus("error"); return;
    }

    // Connecter WebSocket
    setStatus("connecting");
    const ws = new WebSocket(`${WS_URL}/ws/student/${clientId.current}`);
    wsRef.current = ws;

    ws.onopen = () => { setStatus("connected"); setIsStreaming(true); };

    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.status === "buffering") {
        setStatus("buffering");
        setBufferPct((data.frames / data.needed) * 100);
        return;
      }
      if (data.status === "ok" && data.predictions) {
        setStatus("connected");
        setPredictions(data.predictions);
        setEngScore(data.engagement_score ?? 0);
        setHistory(prev => {
          const next = { ...prev };
          Object.keys(STATES).forEach(k => {
            next[k] = [...(prev[k] || []).slice(-39), data.predictions[k]?.level ?? 0];
          });
          return next;
        });
      }
    };

    ws.onerror = () => setStatus("error");
    ws.onclose = () => { setStatus("idle"); setIsStreaming(false); };

    // Envoyer frames
    timerRef.current = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const canvas = document.createElement("canvas");
      canvas.width = 320; canvas.height = 240;
      const ctx = canvas.getContext("2d");
      if (videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        ws.send(JSON.stringify({ frame: canvas.toDataURL("image/jpeg", 0.7) }));
      }
    }, FRAME_MS);
  }, []);

  const stopSession = useCallback(() => {
    clearInterval(timerRef.current);
    wsRef.current?.close();
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false); setStatus("idle");
    setPredictions(null); setEngScore(0); setSessionTime(0);
    setHistory(Object.fromEntries(Object.keys(STATES).map(k => [k, []])));
  }, []);

  const engColor = engScore >= 60 ? "#22D3A5" : engScore >= 30 ? "#F59E3F" : "#EF4B6C";
  const engLabel = engScore >= 60 ? "En flow 🔥" : engScore >= 30 ? "Correct 💡" : "Faible 😴";

  const statusCfg = {
    idle:       { color: "#555",     dot: false, text: "Inactif"     },
    connecting: { color: "#6B7FD4", dot: true,  text: "Connexion..."  },
    connected:  { color: "#22D3A5", dot: true,  text: "En direct"    },
    buffering:  { color: "#F59E3F", dot: true,  text: "Chargement..."  },
    error:      { color: "#EF4B6C", dot: false, text: "Erreur"       },
  }[status] || { color: "#555", dot: false, text: status };

  return (
    <div style={{ minHeight: "100vh", background: "#06090F", color: "#fff", fontFamily: "'DM Sans','Segoe UI',sans-serif" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes up { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
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
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", marginLeft: 4 }}>Étudiant</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Status */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 10px", borderRadius: 20, background: `${statusCfg.color}18`, border: `1px solid ${statusCfg.color}44` }}>
            {statusCfg.dot && <div style={{ width: 5, height: 5, borderRadius: "50%", background: statusCfg.color, animation: "pulse 2s infinite" }} />}
            <span style={{ fontSize: 10, color: statusCfg.color, fontWeight: 700, letterSpacing: 1 }}>{statusCfg.text}</span>
          </div>
          {isStreaming && <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)", fontFamily: "monospace" }}>{fmt(sessionTime)}</span>}
          <button onClick={onLogout} style={{ padding: "5px 12px", borderRadius: 7, border: "1px solid rgba(255,255,255,0.1)", background: "transparent", color: "rgba(255,255,255,0.35)", fontSize: 12, cursor: "pointer" }}>
            Déconnexion
          </button>
        </div>
      </header>

      <main style={{ padding: "24px", maxWidth: 900, margin: "0 auto", animation: "up .35s ease" }}>

        {/* Bouton start/stop */}
        <div style={{ marginBottom: 20, display: "flex", justifyContent: "flex-end" }}>
          {!isStreaming
            ? <button onClick={startSession} style={{ padding: "9px 22px", borderRadius: 10, background: "linear-gradient(135deg,#22D3A5,#6B7FD4)", border: "none", color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer", fontFamily: "inherit" }}>▶ Démarrer</button>
            : <button onClick={stopSession}  style={{ padding: "9px 22px", borderRadius: 10, background: "rgba(239,75,108,0.12)", border: "1px solid rgba(239,75,108,0.3)", color: "#EF4B6C", fontWeight: 600, fontSize: 13, cursor: "pointer", fontFamily: "inherit" }}>⏹ Arrêter</button>
          }
        </div>

        {/* Layout principal */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>

          {/* Webcam */}
          <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 16, overflow: "hidden", aspectRatio: "4/3", position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <video ref={videoRef} autoPlay muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} />
            {!isStreaming && (
              <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(6,9,15,0.85)", gap: 8 }}>
                <div style={{ fontSize: 36 }}>📷</div>
                <div style={{ fontSize: 12, color: "rgba(255,255,255,0.3)" }}>Caméra inactive</div>
              </div>
            )}
            {status === "buffering" && (
              <div style={{ position: "absolute", bottom: 10, left: 10, right: 10, background: "rgba(6,9,15,0.88)", borderRadius: 8, padding: "8px 12px", border: "1px solid rgba(245,158,63,0.25)" }}>
                <div style={{ fontSize: 10, color: "#F59E3F", marginBottom: 4 }}>Initialisation du modèle...</div>
                <div style={{ height: 2, background: "rgba(255,255,255,0.06)", borderRadius: 1 }}>
                  <div style={{ height: "100%", background: "#F59E3F", width: `${bufferPct}%`, transition: "width .3s", borderRadius: 1 }} />
                </div>
              </div>
            )}
          </div>

          {/* Score engagement */}
          <div style={{
            background: "rgba(255,255,255,0.03)", border: `1px solid ${engColor}22`,
            borderRadius: 16, padding: "24px", display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 12,
          }}>
            {/* Arc */}
            <svg width={130} height={130} viewBox="0 0 130 130">
              <defs>
                <linearGradient id="eGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor={engColor} />
                  <stop offset="100%" stopColor="#6B7FD4" />
                </linearGradient>
              </defs>
              {(() => {
                const r = 54, cx = 65, cy = 65;
                const circ = 2 * Math.PI * r;
                const arc  = circ * 0.75;
                const fill = (engScore / 100) * arc;
                return <>
                  <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={9}
                    strokeDasharray={`${arc} ${circ - arc}`} strokeDashoffset={circ * 0.125}
                    transform={`rotate(135 ${cx} ${cy})`} />
                  <circle cx={cx} cy={cy} r={r} fill="none" stroke="url(#eGrad)" strokeWidth={9}
                    strokeDasharray={`${fill} ${circ - fill + circ * 0.25}`} strokeDashoffset={circ * 0.125}
                    strokeLinecap="round" transform={`rotate(135 ${cx} ${cy})`}
                    style={{ transition: "stroke-dasharray .7s cubic-bezier(.4,2,.6,1)" }} />
                  <text x={cx} y={cy - 6} textAnchor="middle" fontSize={26} fontWeight={800} fill="#fff" fontFamily="monospace">{engScore}%</text>
                  <text x={cx} y={cy + 14} textAnchor="middle" fontSize={11} fill="rgba(255,255,255,0.4)">Engagement</text>
                </>;
              })()}
            </svg>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#fff" }}>{engLabel}</div>
          </div>
        </div>

        {/* 4 gauges binaires */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 16 }}>
          {Object.keys(STATES).map(k => (
            <BinaryGauge key={k} state={k} prediction={predictions?.[k]} />
          ))}
        </div>

        {/* Historique sparklines */}
        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 14, padding: "16px 18px" }}>
          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 12 }}>Historique session</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
            {Object.entries(STATES).map(([k, { color, emoji, label }]) => (
              <div key={k}>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 4 }}>{emoji} {label}</div>
                <Sparkline data={history[k]} color={color} />
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
