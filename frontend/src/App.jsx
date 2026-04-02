import { useState } from "react";
import LoginPage   from "./LoginPage.jsx";
import StudentView from "./StudentView.jsx";
import TeacherView from "./TeacherView.jsx";

export default function App() {
  const [user, setUser] = useState(null);

  const handleLogin = (u) => setUser(u);

  const handleLogout = () => {
    // Supprimer le token JWT au logout
    localStorage.removeItem("edusense_token");
    setUser(null);
  };

  if (!user) return <LoginPage onLogin={handleLogin} />;

  if (user.role === "teacher")
    return <TeacherView user={user} onLogout={handleLogout} />;

  return <StudentView user={user} onLogout={handleLogout} />;
}
