import { NavLink, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "../lib/api";

/** Shell header: brand, route nav, live operating-point pill. */
export function Header() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    retry: false,
    staleTime: 60_000,
  });

  const link = ({ isActive }: { isActive: boolean }) =>
    `min-h-[44px] inline-flex items-center px-3 font-mono text-xs tracking-[0.08em] transition-colors ${
      isActive ? "text-teal" : "text-dim hover:text-ink"
    }`;

  return (
    <header className="border-b border-hairline">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-4 md:px-6">
        <Link
          to="/"
          className="font-display text-lg font-bold tracking-tight text-ink"
        >
          SIGNAL LAB <span className="text-dim">// VOICE AI DETECTOR</span>
        </Link>
        <nav aria-label="Primary" className="flex items-center gap-1">
          <NavLink to="/" end className={link}>
            CONSOLE
          </NavLink>
          <NavLink to="/model" className={link}>
            MODEL INTEL
          </NavLink>
        </nav>
        <span className="ml-auto rounded-full border border-hairline bg-panel px-3 py-1 font-mono text-[11px] tracking-wider text-dim">
          OP-POINT: {(health?.threshold ?? 0.85).toFixed(2)}
        </span>
      </div>
    </header>
  );
}
