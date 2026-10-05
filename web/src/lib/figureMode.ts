// Figure mode: switches the whole app to the white print palette used for
// dissertation figures, by toggling the "fig" class on <html> (see styles/index.css).
// The class is set in a *layout* effect, which React runs before any child's normal
// effect, so charts that read the CSS colours afterwards always see the right palette.
// The choice is remembered per browser; storage may be unavailable, so it is optional.
import { useCallback, useLayoutEffect, useState } from "react";

const KEY = "silversight.figureMode";

function read(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function useFigureMode(): [boolean, () => void] {
  const [on, setOn] = useState<boolean>(read);
  useLayoutEffect(() => {
    document.documentElement.classList.toggle("fig", on);
    try {
      localStorage.setItem(KEY, on ? "1" : "0");
    } catch {
      /* storage unavailable: the toggle still works for this session */
    }
  }, [on]);
  const toggle = useCallback(() => setOn((v) => !v), []);
  return [on, toggle];
}
