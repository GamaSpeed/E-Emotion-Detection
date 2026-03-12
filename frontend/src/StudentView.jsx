import { useState, useEffect, useRef, useCallback } from "react";

// ── Config ────────────────────────────────────────────────
const WS_URL      = import.meta.env.VITE_WS_URL  || "ws://localhost:8000";
const API_URL     = import.meta.env.VITE_API_URL || "http://localhost:8000";
const FRAME_RATE  = 500; // ms entre chaque envoi de frame

const EMOTIONS = {
  Boredom:     { color: "#6B7FD4", emoji: "😴", label: "Ennui" },
  Engagement:  { color: "#22D3A5", emoji: "🎯", label: "Engagement" },
  Confusion:   { color: "#F59E3F", emoji: "🤔", label: "Confusion" },
  Frustration: { color: "#EF4B6C", emoji: "😤", label: "Frustration" },
};
const LEVELS = ["Très faible", "Faible", "Élevé", "Très élevé"];

// ── Radial Gauge ──────────────────────────────────────────
function RadialGauge({ value = 0, max = 3, color, label, emoji, confidence = 0 }) {
  const pct   = value / max;
  const r     = 52;
  const circ  = 2 * Math.PI * r;
  const dash  = pct * circ * 0.75;

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:4 }}>
      <svg width={124} height={124} viewBox="0 0 124 124">
        <defs>
          <linearGradient id={`g-${label}`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="0.3"/>
            <stop offset="100%" stopColor={color}/>
          </linearGradient>
        </defs>
        <circle cx={62} cy={62} r={r} fill="none"
          stroke="rgba(255,255,255,0.06)" strokeWidth={9}
          strokeDasharray={`${circ*0.75} ${circ*0.25}`}
          strokeDashoffset={circ*0.125}
          transform="rotate(135 62 62)"/>
        <circle cx={62} cy={62} r={r} fill="none"
          stroke={`url(#g-${label})`} strokeWidth={9}
          strokeDasharray={`${dash} ${circ - dash + circ*0.25}`}
          strokeDashoffset={circ*0.125}
          strokeLinecap="round" transform="rotate(135 62 62)"
          style={{ transition:"stroke-dasharray 0.7s cubic-bezier(.4,2,.6,1)" }}/>
        <text x={62} y={55} textAnchor="middle" fontSize={20}>{emoji}</text>
        <text x={62} y={74} textAnchor="middle" fontSize={11} fontWeight={700}
          fill={color} fontFamily="monospace">{LEVELS[value]}</text>
        <text x={62} y={88} textAnchor="middle" fontSize={9}
          fill="rgba(255,255,255,0.3)" fontFamily="monospace">
          {(confidence * 100).toFixed(0)}%
        </text>
      </svg>
      <span style={{ fontSize:10, color:"rgba(255,255,255,0.4)", letterSpacing:2,
        textTransform:"uppercase", fontFamily:"monospace" }}>{label}</span>
    </div>
  );
}

// ── Connection Status Badge ───────────────────────────────
function StatusBadge({ status }) {
  const cfg = {
    connected:  { color:"#22D3A5", label:"CONNECTÉ",    dot: true  },
    buffering:  { color:"#F59E3F", label:"BUFFERING",   dot: true  },
    connecting: { color:"#6B7FD4", label:"CONNEXION...", dot: true },
    error:      { color:"#EF4B6C", label:"ERREUR",      dot: false },
    idle:       { color:"#666",    label:"INACTIF",     dot: false },
  }[status] || { color:"#666", label: status.toUpperCase(), dot: false };

  return (
    <div style={{ display:"flex", alignItems:"center", gap:6,
      padding:"4px 12px", borderRadius:20,
      background:`${cfg.color}18`, border:`1px solid ${cfg.color}44` }}>
      {cfg.dot && (
        <div style={{ width:6, height:6, borderRadius:"50%",
          background:cfg.color, boxShadow:`0 0 6px ${cfg.color}`,
          animation:"pulse 2s infinite" }}/>
      )}
      <span style={{ fontSize:10, color:cfg.color, fontWeight:700,
        fontFamily:"monospace", letterSpacing:1.5 }}>{cfg.label}</span>
    </div>
  );
}

// ── Sparkline ─────────────────────────────────────────────
function Sparkline({ data, color, width=100, height=32 }) {
  if (data.length < 2) return <div style={{ width, height }}/>;
  const pts = data.map((v,i) => {
    const x = (i / (data.length-1)) * width;
    const y = height - (v/3) * (height-4) - 2;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={width} height={height}>
      <polyline points={pts} fill="none" stroke={color}
        strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.8}/>
    </svg>
  );
}

// ── Main Student View ─────────────────────────────────────
export default function StudentView() {
  const videoRef     = useRef(null);
  const canvasRef    = useRef(null);
  const wsRef        = useRef(null);
  const intervalRef  = useRef(null);
  const clientId     = useRef(`student_${Date.now()}`);

  const [wsStatus,     setWsStatus]     = useState("idle");
  const [cameraMode,   setCameraMode]   = useState("local"); // "local" | "server"
  const [predictions,  setPredictions]  = useState(null);
  const [engScore,     setEngScore]     = useState(0);
  const [sessionTime,  setSessionTime]  = useState(0);
  const [history,      setHistory]      = useState(
    Object.fromEntries(Object.keys(EMOTIONS).map(k => [k, []]))
  );
  const [buffering,    setBuffering]    = useState({ frames:0, needed:8 });
  const [isStreaming,  setIsStreaming]  = useState(false);

  // ── Timer session ─────────────────────────────────────
  useEffect(() => {
    if (!isStreaming) return;
    const t = setInterval(() => setSessionTime(s => s+1), 1000);
    return () => clearInterval(t);
  }, [isStreaming]);

  const fmt = s => `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;

  // ── Démarrer la webcam locale ─────────────────────────
  const startLocalCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width:640, height:480, facingMode:"user" }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (e) {
      console.error("Caméra non disponible :", e);
      setWsStatus("error");
    }
  }, []);

  // ── Connecter WebSocket ───────────────────────────────
  const connectWS = useCallback(() => {
    const url = cameraMode === "local"
      ? `${WS_URL}/ws/student/${clientId.current}`
      : `${WS_URL}/ws/camera`;

    setWsStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus("connected");
      setIsStreaming(true);
    };

    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);

      if (data.status === "buffering") {
        setWsStatus("buffering");
        setBuffering({ frames: data.frames, needed: data.needed });
        return;
      }

      if (data.status === "ok" && data.predictions) {
        setWsStatus("connected");
        setPredictions(data.predictions);
        setEngScore(data.engagement_score || 0);

        // Historique sparklines
        setHistory(prev => {
          const next = { ...prev };
          Object.keys(EMOTIONS).forEach(k => {
            next[k] = [...(prev[k] || []).slice(-29), data.predictions[k]?.level ?? 0];
          });
          return next;
        });

        // Afficher frame serveur si mode serveur
        if (cameraMode === "server" && data.frame && canvasRef.current) {
          const img = new Image();
          img.onload = () => {
            const ctx = canvasRef.current.getContext("2d");
            ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
          };
          img.src = `data:image/jpeg;base64,${data.frame}`;
        }
      }
    };

    ws.onerror = () => setWsStatus("error");
    ws.onclose = () => {
      setWsStatus("idle");
      setIsStreaming(false);
      clearInterval(intervalRef.current);
    };
  }, [cameraMode]);

  // ── Envoyer frames (mode local) ───────────────────────
  const startSendingFrames = useCallback(() => {
    if (cameraMode !== "local") return;
    const canvas  = document.createElement("canvas");
    canvas.width  = 320;
    canvas.height = 240;
    const ctx = canvas.getContext("2d");

    intervalRef.current = setInterval(() => {
      if (!videoRef.current || !wsRef.current ||
          wsRef.current.readyState !== WebSocket.OPEN) return;

      ctx.drawImage(videoRef.current, 0, 0, 320, 240);
      const b64 = canvas.toDataURL("image/jpeg", 0.7);
      wsRef.current.send(JSON.stringify({ frame: b64 }));
    }, FRAME_RATE);
  }, [cameraMode]);

  // ── Démarrer session ──────────────────────────────────
  const startSession = useCallback(async () => {
    if (cameraMode === "local") await startLocalCamera();
    connectWS();
    if (cameraMode === "local") setTimeout(startSendingFrames, 500);
  }, [cameraMode, startLocalCamera, connectWS, startSendingFrames]);

  // ── Arrêter session ───────────────────────────────────
  const stopSession = useCallback(() => {
    clearInterval(intervalRef.current);
    wsRef.current?.close();
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
    }
    setIsStreaming(false);
    setWsStatus("idle");
  }, []);

  useEffect(() => () => stopSession(), []);

  // ── Engagement color ──────────────────────────────────
  const engColor = engScore >= 67 ? "#22D3A5" : engScore >= 33 ? "#F59E3F" : "#EF4B6C";

  return (
    <div style={{
      minHeight:"100vh", background:"#090E1A", color:"#fff",
      fontFamily:"'DM Sans','Segoe UI',sans-serif",
    }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
      `}</style>

      {/* Background glows */}
      <div style={{ position:"fixed", top:-200, left:-200, width:500, height:500,
        borderRadius:"50%", pointerEvents:"none",
        background:"radial-gradient(circle, #22D3A511 0%, transparent 70%)" }}/>
      <div style={{ position:"fixed", bottom:-200, right:-200, width:400, height:400,
        borderRadius:"50%", pointerEvents:"none",
        background:"radial-gradient(circle, #6B7FD411 0%, transparent 70%)" }}/>

      {/* Header */}
      <header style={{ padding:"0 28px", height:60, display:"flex",
        alignItems:"center", justifyContent:"space-between",
        borderBottom:"1px solid rgba(255,255,255,0.06)",
        background:"rgba(9,14,26,0.85)", backdropFilter:"blur(20px)",
        position:"sticky", top:0, zIndex:100 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ width:30, height:30, borderRadius:9,
            background:"linear-gradient(135deg,#22D3A5,#6B7FD4)",
            display:"flex", alignItems:"center", justifyContent:"center", fontSize:15 }}>🧠</div>
          <span style={{ fontWeight:700, fontSize:15, letterSpacing:-0.5 }}>
            EduSense <span style={{ color:"#22D3A5" }}>AI</span>
          </span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <StatusBadge status={wsStatus}/>
          <span style={{ fontSize:13, color:"rgba(255,255,255,0.3)",
            fontFamily:"monospace" }}>{fmt(sessionTime)}</span>
        </div>
      </header>

      <main style={{ padding:"24px 28px", maxWidth:960, margin:"0 auto",
        animation:"fadeUp 0.4s ease" }}>

        {/* Mode selector + Start */}
        {!isStreaming && (
          <div style={{ marginBottom:28, display:"flex", gap:12, alignItems:"center",
            flexWrap:"wrap" }}>
            <div style={{ display:"flex", gap:4, padding:4,
              background:"rgba(255,255,255,0.05)", borderRadius:10 }}>
              {[["local","💻 Webcam locale"],["server","🖥 Caméra serveur"]].map(([m,l]) => (
                <button key={m} onClick={() => setCameraMode(m)} style={{
                  padding:"6px 14px", borderRadius:7, border:"none", cursor:"pointer",
                  background: cameraMode===m ? "rgba(34,211,165,0.2)" : "transparent",
                  color: cameraMode===m ? "#22D3A5" : "rgba(255,255,255,0.4)",
                  fontWeight:600, fontSize:12, transition:"all 0.2s",
                }}>{l}</button>
              ))}
            </div>
            <button onClick={startSession} style={{
              padding:"10px 24px", borderRadius:12,
              background:"linear-gradient(135deg,#22D3A5,#6B7FD4)",
              border:"none", color:"#fff", fontWeight:700, fontSize:14,
              cursor:"pointer", letterSpacing:0.5,
            }}>▶ Démarrer la session</button>
          </div>
        )}

        {isStreaming && (
          <div style={{ marginBottom:20, display:"flex", justifyContent:"flex-end" }}>
            <button onClick={stopSession} style={{
              padding:"8px 20px", borderRadius:10,
              background:"rgba(239,75,108,0.15)",
              border:"1px solid rgba(239,75,108,0.4)",
              color:"#EF4B6C", fontWeight:600, fontSize:13, cursor:"pointer",
            }}>⏹ Arrêter</button>
          </div>
        )}

        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr",
          gap:20, marginBottom:20 }}>

          {/* Video feed */}
          <div style={{ background:"rgba(255,255,255,0.03)",
            border:"1px solid rgba(255,255,255,0.07)",
            borderRadius:20, overflow:"hidden", aspectRatio:"4/3",
            display:"flex", alignItems:"center", justifyContent:"center",
            position:"relative" }}>
            {cameraMode === "local" ? (
              <video ref={videoRef} autoPlay muted playsInline
                style={{ width:"100%", height:"100%", objectFit:"cover",
                  transform:"scaleX(-1)" }}/>
            ) : (
              <canvas ref={canvasRef} width={640} height={480}
                style={{ width:"100%", height:"100%", objectFit:"cover" }}/>
            )}
            {!isStreaming && (
              <div style={{ position:"absolute", inset:0, display:"flex",
                flexDirection:"column", alignItems:"center", justifyContent:"center",
                background:"rgba(9,14,26,0.8)", gap:8 }}>
                <div style={{ fontSize:40 }}>📷</div>
                <div style={{ fontSize:13, color:"rgba(255,255,255,0.4)" }}>
                  Caméra inactive
                </div>
              </div>
            )}
            {wsStatus === "buffering" && (
              <div style={{ position:"absolute", bottom:12, left:12, right:12,
                background:"rgba(9,14,26,0.85)", borderRadius:8, padding:"8px 12px",
                border:"1px solid rgba(245,158,63,0.3)" }}>
                <div style={{ fontSize:11, color:"#F59E3F", marginBottom:4,
                  fontFamily:"monospace" }}>
                  Chargement frames... {buffering.frames}/{buffering.needed}
                </div>
                <div style={{ height:3, background:"rgba(255,255,255,0.06)",
                  borderRadius:2, overflow:"hidden" }}>
                  <div style={{ height:"100%", borderRadius:2,
                    background:"#F59E3F",
                    width:`${(buffering.frames/buffering.needed)*100}%`,
                    transition:"width 0.3s" }}/>
                </div>
              </div>
            )}
          </div>

          {/* Engagement hero */}
          <div style={{ background:"linear-gradient(135deg,rgba(34,211,165,0.08),rgba(107,127,212,0.08))",
            border:`1px solid ${engColor}22`,
            borderRadius:20, padding:"24px",
            display:"flex", flexDirection:"column",
            alignItems:"center", justifyContent:"center", gap:16 }}>
            <svg width={150} height={150} viewBox="0 0 150 150">
              <defs>
                <linearGradient id="eGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor={engColor}/>
                  <stop offset="100%" stopColor="#6B7FD4"/>
                </linearGradient>
              </defs>
              <circle cx={75} cy={75} r={65} fill="none"
                stroke="rgba(255,255,255,0.06)" strokeWidth={11}
                strokeDasharray={`${2*Math.PI*65*0.75} ${2*Math.PI*65*0.25}`}
                strokeDashoffset={2*Math.PI*65*0.125}
                transform="rotate(135 75 75)"/>
              <circle cx={75} cy={75} r={65} fill="none"
                stroke="url(#eGrad)" strokeWidth={11}
                strokeDasharray={`${(engScore/100)*2*Math.PI*65*0.75} ${2*Math.PI*65}`}
                strokeDashoffset={2*Math.PI*65*0.125}
                strokeLinecap="round" transform="rotate(135 75 75)"
                style={{ transition:"stroke-dasharray 0.8s cubic-bezier(.4,2,.6,1)" }}/>
              <text x={75} y={68} textAnchor="middle" fontSize={30} fontWeight={800}
                fill="#fff" fontFamily="monospace">{engScore}%</text>
              <text x={75} y={88} textAnchor="middle" fontSize={12}
                fill="rgba(255,255,255,0.4)">Engagement</text>
            </svg>
            <div style={{ textAlign:"center" }}>
              <div style={{ fontSize:18, fontWeight:700, marginBottom:4 }}>
                {engScore >= 67 ? "🔥 Tu es dans le flow !" :
                 engScore >= 33 ? "💡 Continue comme ça" : "😴 Ressaisis-toi !"}
              </div>
              <div style={{ fontSize:12, color:"rgba(255,255,255,0.4)" }}>
                {predictions
                  ? `Confiance : ${(predictions.Engagement?.confidence*100||0).toFixed(0)}%`
                  : "En attente de données..."}
              </div>
            </div>
          </div>
        </div>

        {/* 4 Gauges */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)",
          gap:14, marginBottom:20 }}>
          {Object.entries(EMOTIONS).map(([key, { color, emoji, label }]) => (
            <div key={key} style={{
              background:"rgba(255,255,255,0.03)",
              border:`1px solid ${color}18`,
              borderRadius:18, padding:"18px 10px",
              display:"flex", justifyContent:"center",
            }}>
              <RadialGauge
                value={predictions?.[key]?.level ?? 0}
                color={color} label={label} emoji={emoji}
                confidence={predictions?.[key]?.confidence ?? 0}/>
            </div>
          ))}
        </div>

        {/* Sparklines historique */}
        <div style={{ background:"rgba(255,255,255,0.03)",
          border:"1px solid rgba(255,255,255,0.06)",
          borderRadius:18, padding:"18px 22px" }}>
          <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)",
            textTransform:"uppercase", letterSpacing:1.5, marginBottom:14 }}>
            Historique session
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12 }}>
            {Object.entries(EMOTIONS).map(([key, { color, emoji, label }]) => (
              <div key={key}>
                <div style={{ fontSize:11, color:"rgba(255,255,255,0.5)",
                  marginBottom:6 }}>{emoji} {label}</div>
                <Sparkline data={history[key]} color={color} width={160} height={36}/>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
