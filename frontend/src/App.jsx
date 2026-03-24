import { useState } from "react";
import LoginPage   from "./LoginPage.jsx";
import StudentView from "./StudentView.jsx";
import TeacherView from "./TeacherView.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  if (!user)               return <LoginPage   onLogin={u => setUser(u)}/>;
  if (user.role==="teacher") return <TeacherView user={user} onLogout={()=>setUser(null)}/>;
  return                          <StudentView  user={user} onLogout={()=>setUser(null)}/>;
}
