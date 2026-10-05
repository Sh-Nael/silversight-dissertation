// Form controls in the console style.
import { useContext } from "react";
import { FigureContext, modelColor } from "../lib/theme";

/** Segmented switch: a small set of mutually exclusive options. */
export function Seg<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <span role="group" aria-label={label} className="inline-flex overflow-hidden rounded-md border border-line">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={`cursor-pointer border-0 px-2.5 py-1 text-xs font-medium ${
            o.value === value ? "bg-signal-dim text-text" : "bg-transparent text-muted hover:text-text"
          }`}
        >
          {o.label}
        </button>
      ))}
    </span>
  );
}

export function Select({
  id,
  label,
  value,
  options,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <label htmlFor={id} className="inline-flex items-center gap-2 text-xs text-muted">
      {label}
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-line bg-panel-2 px-2 py-1 font-mono text-xs text-text"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Model name with its fixed colour swatch. */
export function ModelName({ model }: { model: string }) {
  const fig = useContext(FigureContext);
  return (
    <span className="inline-flex items-center gap-2">
      <span className="inline-block size-[9px] rounded-[2px]" style={{ background: modelColor(model, fig) }} />
      {model}
    </span>
  );
}
