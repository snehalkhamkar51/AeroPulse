import { NavLink } from "react-router-dom";
import { Gauge, Info, Layers, Cpu } from "lucide-react";
import { ReactNode } from "react";

const navItems = [
  { to: "/", label: "Landing", icon: Cpu },
  { to: "/inputs", label: "Engine Inputs", icon: Layers },
  { to: "/results", label: "Results", icon: Gauge },
  { to: "/about", label: "About", icon: Info },
];

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">AP</div>
          <div>
            <div className="brand-title">AeroPulse</div>
            <div className="brand-sub">AI Engine Diagnostics</div>
          </div>
        </div>
        <nav className="nav">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              <Icon size={16} /> <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="main">{children}</main>
      <footer className="footer">
        Educational predictive-maintenance prototype — not a certified aviation diagnostic system.
      </footer>
    </div>
  );
}
