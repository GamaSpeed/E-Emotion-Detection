import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "";

// Comptes démo — remplis automatiquement le formulaire (mot de passe non exposé)
const DEMO_ACCOUNTS = [
  { label: "👨‍🏫 Professeur", email: "prof@edusense.ai",     password: "prof123"     },
  { label: "👨‍🎓 Étudiant",   email: "etudiant@edusense.ai", password: "etudiant123" },
];

export default function LoginPage({ onLogin }) {
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState("");
  const [loading,  setLoading]  = useState(false);

  const fill = (e, p) => { setEmail(e); setPassword(p); setError(""); };

  const handleSubmit = async (ev) => {
    ev.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ email: email.toLowerCase().trim(), password }),
      });

      const data = await res.json();

      if (!res.ok) {
        // FastAPI renvoie { detail: "..." } en cas d'erreur
        setError(data.detail || "Erreur de connexion.");
        setLoading(false);
        return;
      }

      // Stocker le token JWT — utilisé par TeacherView (historique REST)
      localStorage.setItem("edusense_token", data.access_token);

      // Remonter l'utilisateur au composant parent (App.jsx)
      onLogin({
        id:    data.user.id,
        email: data.user.email,
        name:  data.user.name,
        role:  data.user.role,
      });

    } catch (err) {
      // Erreur réseau (backend injoignable, CORS, etc.)
      setError("Impossible de joindre le serveur. Vérifiez que le backend est démarré.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#06090F",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    }}>
      <style>{`
        @keyframes up   { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:none} }
        @keyframes spin { to{transform:rotate(360deg)} }
        .inp {
          width: 100%; box-sizing: border-box; padding: 11px 14px; border-radius: 8px;
          background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
          color: #fff; font-size: 14px; font-family: inherit; outline: none;
          transition: border-color .15s;
        }
        .inp:focus { border-color: #22D3A5; }
        .inp::placeholder { color: rgba(255,255,255,0.25); }
        .dbtn {
          flex: 1; padding: 4px; border-radius: 8px; cursor: pointer; text-align: left;
          background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
          color: rgba(255,255,255,0.55); font-size: 12px; transition: background .15s;
          font-family: inherit;
        }
        .dbtn:hover { background: rgba(255,255,255,0.08); }
        .btn-submit {
          width: 100%; padding: 12px; border-radius: 10px;
          background: linear-gradient(135deg,#22D3A5,#6B7FD4);
          border: none; color: #fff; font-weight: 700; font-size: 14px;
          cursor: pointer; font-family: inherit; transition: opacity .15s;
          display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .btn-submit:disabled { opacity: 0.65; cursor: not-allowed; }
        .btn-submit:hover:not(:disabled) { opacity: 0.9; }
      `}</style>

      <div style={{ width: "100%", maxWidth: 380, padding: "0 20px", animation: "up .4s ease" }}>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 12, margin: "0 auto 12px",
            background: "linear-gradient(135deg,#22D3A5,#6B7FD4)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20,
          }}>🧠</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "#fff", letterSpacing: -0.5 }}>
            Edu<span style={{ color: "#22D3A5" }}>Sense</span>
          </div>
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 3, letterSpacing: 1.5, textTransform: "uppercase" }}>
            Plateforme e-learning IA
          </div>
        </div>

        {/* Formulaire */}
        <div style={{
          background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16, padding: "28px 24px",
        }}>
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: 14 }}>
              <label style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", letterSpacing: 0.8, textTransform: "uppercase", display: "block", marginBottom: 6 }}>
                Email
              </label>
              <input
                className="inp" type="email" required
                value={email} onChange={e => setEmail(e.target.value)}
                placeholder="vous@edusense.ai"
              />
            </div>

            <div style={{ marginBottom: 22 }}>
              <label style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", letterSpacing: 0.8, textTransform: "uppercase", display: "block", marginBottom: 6 }}>
                Mot de passe
              </label>
              <input
                className="inp" type="password" required
                value={password} onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>

            {/* Erreur */}
            {error && (
              <div style={{
                background: "rgba(239,75,108,0.1)", border: "1px solid rgba(239,75,108,0.25)",
                borderRadius: 8, padding: "9px 12px", marginBottom: 16,
                fontSize: 13, color: "#EF4B6C", lineHeight: 1.5,
              }}>
                {error}
              </div>
            )}

            <button className="btn-submit" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <div style={{
                    width: 15, height: 15, borderRadius: "50%",
                    border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff",
                    animation: "spin .6s linear infinite",
                  }} />
                  Connexion...
                </>
              ) : "Se connecter"}
            </button>
          </form>
        </div>

        {/* Comptes démo */}
        <div style={{
          marginTop: 14, background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "14px 16px",
        }}>
          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 8 }}>
            Comptes démo
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {DEMO_ACCOUNTS.map(({ label, email: e, password: p }) => (
              <button key={e} className="dbtn" onClick={() => fill(e, p)}>
                <div style={{ fontWeight: 600, marginBottom: 2, padding: "4px 6px" }}>{label}</div>
                <div style={{ color: "rgba(255,255,255,0.25)", fontSize: 10, padding: "0 6px 4px" }}>{e}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Info serveur */}
        <div style={{ marginTop: 10, textAlign: "center", fontSize: 10, color: "rgba(255,255,255,0.15)" }}>
          Serveur : {API_URL}
        </div>
      </div>
    </div>
  );
}
