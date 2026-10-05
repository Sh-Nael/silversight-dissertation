// The single list of pages. The navigation rail and the router are both built from it,
// so adding a page means adding one entry here plus its component in App.tsx.
export interface PageDef {
  path: string;
  label: string;
  title: string;
  purpose: string;
  arrives?: string; // the plan step that builds it; absent once it exists
}

export const PAGES: PageDef[] = [
  {
    path: "/",
    label: "Overview",
    title: "Where the evidence stands",
    purpose: "Snapshot integrity, code version, test status, latest runs and the headline standing of every model.",
  },
  {
    path: "/runs",
    label: "Runs",
    title: "Every run, traceable to code and data",
    purpose: "All walk-forward runs with their manifests, timings and tuning logs; later, starting new runs with live progress.",
  },
  {
    path: "/compare",
    label: "Compare",
    title: "Scoreboard and significance",
    purpose: "Pick runs and a reference model: scoreboard, Diebold–Mariano / Wilcoxon / permutation tests, per-fold and cumulative charts, development scores.",
  },
  {
    path: "/calibration",
    label: "Calibration",
    title: "Do the probabilities mean what they say?",
    purpose: "Reliability diagrams for direction forecasts, and volatility forecasts against realised volatility.",
  },
  {
    path: "/signals",
    label: "Signals",
    title: "Can the forecasts be traded?",
    purpose: "BUY / SELL on confident days with a stop and a target from the volatility forecast, backtested on daily bars; buy and sell trades separately; account equity.",
  },
  {
    path: "/folds",
    label: "Folds",
    title: "The frozen walk-forward calendar",
    purpose: "The 8 folds, 200 monthly refits and 17 tuning points on one timeline.",
  },
  {
    path: "/data",
    label: "Data",
    title: "Markets, features and data quality",
    purpose: "Every node and feature with regime bands and the event calendar, plus the data-quality audit.",
  },
  {
    path: "/graph",
    label: "Driver graph",
    title: "Which drivers move silver, and when",
    purpose: "AIM-DG's edge on/off timeline with reliability, the regime tests, persistence per edge and the driver graph of any month.",
  },
];
