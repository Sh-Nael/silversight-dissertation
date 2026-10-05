// Small shared building blocks, styled only through the design tokens.
import type { ReactNode } from "react";

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="font-mono text-[11px] tracking-[.12em] text-signal uppercase">{eyebrow}</div>
      <h1 className="m-0 font-display text-[22px] leading-tight font-semibold tracking-[.02em] text-balance">{title}</h1>
      {children && <p className="m-0 mt-1 max-w-[75ch] text-muted">{children}</p>}
    </div>
  );
}

export function Panel({ title, sub, children, tools, className = "" }: { title: string; sub?: ReactNode; children: ReactNode; tools?: ReactNode; className?: string }) {
  return (
    <section className={`flex min-w-0 flex-col gap-3 rounded-[10px] border border-line bg-panel px-[18px] py-4 ${className}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="m-0 text-sm font-semibold">{title}</h2>
          {sub && <p className="m-0 mt-0.5 max-w-[80ch] text-[12.5px] text-muted">{sub}</p>}
        </div>
        {tools && <div className="flex flex-wrap gap-1.5">{tools}</div>}
      </div>
      {children}
    </section>
  );
}

export function Pill({ tone, children }: { tone: "good" | "bad" | "neutral"; children: ReactNode }) {
  const cls = {
    good: "bg-good/15 text-good",
    bad: "bg-bad/15 text-bad",
    neutral: "bg-panel-2 text-muted",
  }[tone];
  return <span className={`inline-block rounded-full px-2 py-px font-mono text-[11px] font-medium ${cls}`}>{children}</span>;
}

/** Table shell with horizontal scrolling on narrow screens. Cells use TH / TD below. */
export function Table({ head, children }: { head: ReactNode; children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-line-soft">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="bg-panel-2 font-mono text-[11px] tracking-[.06em] text-muted uppercase">{head}</tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export const TH = "px-2.5 py-2 font-medium whitespace-nowrap text-right first:text-left";
export const TD = "border-t border-line-soft px-2.5 py-2 whitespace-nowrap text-right first:text-left";
export const NUM = "font-mono";
/** For text columns after the first (tables right-align by default, for numbers). */
export const TXT = "text-left!";

export function Loading({ what }: { what: string }) {
  return <p className="m-0 text-muted">Loading {what}…</p>;
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <p className="m-0 rounded-md border border-bad/40 px-3 py-2 text-[13px] text-bad">
      {error}. Is <code className="font-mono">xag ui</code> running?
    </p>
  );
}

export function Kpi({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-line bg-panel px-4 py-3.5">
      <span className="font-mono text-[11px] tracking-[.08em] text-muted uppercase">{label}</span>
      <b className="font-mono text-2xl leading-tight font-medium">{value}</b>
      {note && <small className="text-xs text-muted">{note}</small>}
    </div>
  );
}
