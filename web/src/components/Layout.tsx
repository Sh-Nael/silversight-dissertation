// App shell: navigation rail on the left, status bar on top, the current page below.
import { NavLink, Outlet } from "react-router";
import { PAGES } from "../pages";
import { useApi, type Status } from "../lib/api";
import { useFigureMode } from "../lib/figureMode";
import { FigureContext } from "../lib/theme";

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true" className="flex-none">
      <circle cx="15" cy="15" r="13" fill="none" stroke="currentColor" strokeOpacity=".35" />
      <g stroke="var(--signal)" strokeWidth="1.4">
        <line x1="15" y1="15" x2="15" y2="3.5" />
        <line x1="15" y1="15" x2="25.5" y2="10" />
        <line x1="15" y1="15" x2="23" y2="24.5" />
        <line x1="15" y1="15" x2="6.5" y2="23.5" />
        <line x1="15" y1="15" x2="4.5" y2="10" />
      </g>
      <g fill="var(--signal)">
        <circle cx="15" cy="3.5" r="1.8" />
        <circle cx="25.5" cy="10" r="1.8" />
        <circle cx="23" cy="24.5" r="1.8" />
        <circle cx="6.5" cy="23.5" r="1.8" />
        <circle cx="4.5" cy="10" r="1.8" />
      </g>
      <circle cx="15" cy="15" r="4.5" fill="var(--silver)" />
    </svg>
  );
}

function Chip({ ok, children }: { ok?: boolean; children: React.ReactNode }) {
  const dot = ok === undefined ? "bg-faint" : ok ? "bg-good" : "bg-bad";
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel px-2.5 py-0.5 font-mono text-[11.5px] text-muted">
      <i className={`inline-block size-[7px] rounded-full ${dot}`} />
      {children}
    </span>
  );
}

function StatusChips() {
  const { data, error, loading } = useApi<Status>("/api/status");
  if (loading) return <Chip>checking data…</Chip>;
  if (error || !data) return <Chip ok={false}>API unreachable: is `xag ui` running?</Chip>;
  return (
    <>
      {data.snapshots.map((s) => (
        <Chip key={s.name} ok={s.verified}>
          {s.name} · {s.verified ? `${s.files} files verified` : "CHECKSUM FAILURE"}
        </Chip>
      ))}
      <Chip ok={!data.code.dirty}>
        HEAD {data.code.commit}
        {data.code.dirty ? " (uncommitted changes)" : ""}
      </Chip>
      <Chip>{data.runs} runs</Chip>
    </>
  );
}

export default function Layout() {
  const [fig, toggleFig] = useFigureMode();
  return (
    <div className="grid min-h-full grid-cols-1 md:grid-cols-[220px_1fr]">
      <aside className="flex flex-wrap items-center gap-4 border-b border-line bg-panel px-4 py-3 md:sticky md:top-0 md:h-screen md:flex-col md:flex-nowrap md:items-stretch md:gap-5 md:border-r md:border-b-0 md:px-3 md:py-5">
        <div className="flex items-center gap-2.5 px-1.5">
          <Logo />
          <div>
            <b className="block font-display text-[17px] leading-none font-semibold tracking-[.06em] uppercase">
              SilverSight
            </b>
            <small className="mt-1 block font-mono text-[10px] tracking-[.08em] text-muted uppercase">
              research console
            </small>
          </div>
        </div>
        <nav aria-label="Pages" className="flex w-full gap-1 overflow-x-auto md:flex-col md:gap-0.5">
          {PAGES.map((p) => (
            <NavLink
              key={p.path}
              to={p.path}
              end={p.path === "/"}
              className={({ isActive }) =>
                `flex items-center justify-between rounded-md px-2.5 py-2 font-medium whitespace-nowrap no-underline ${
                  isActive
                    ? "bg-signal-dim text-text shadow-[inset_2px_0_0_var(--signal)]"
                    : "text-muted hover:bg-panel-2 hover:text-text"
                }`
              }
            >
              {p.label}
              {p.arrives && <em className="hidden font-mono text-[10px] text-faint not-italic md:inline">{p.arrives}</em>}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto hidden px-1.5 font-mono text-[11px] leading-relaxed text-faint md:block">
          XAG/USD · daily
          <br />
          2006-01-03 → 2026-06-30
        </div>
      </aside>

      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-line bg-bg px-4 py-3 md:px-6">
          <div className="flex min-w-0 flex-1 flex-wrap gap-2">
            <StatusChips />
          </div>
          <button
            type="button"
            onClick={toggleFig}
            aria-pressed={fig}
            className="inline-flex flex-none cursor-pointer items-center gap-2 rounded-lg border border-line bg-panel px-3 py-1.5 font-medium hover:border-signal"
          >
            <span className={`relative h-4 w-7 rounded-full transition-colors ${fig ? "bg-signal" : "bg-line"}`}>
              <span
                className={`absolute top-0.5 size-3 rounded-full transition-[left] ${fig ? "left-3.5 bg-white" : "left-0.5 bg-text"}`}
              />
            </span>
            Figure mode
          </button>
        </header>
        <main className="w-full max-w-[1360px] p-4 md:p-6">
          <FigureContext.Provider value={fig}>
            <Outlet />
          </FigureContext.Provider>
        </main>
      </div>
    </div>
  );
}
