// Routes: one per entry in PAGES. Built pages get their own component; the rest show
// a placeholder until their plan step lands.
import { createBrowserRouter, RouterProvider } from "react-router";
import Layout from "./components/Layout";
import Calibration from "./pages/Calibration";
import Compare from "./pages/Compare";
import Data from "./pages/Data";
import DriverGraph from "./pages/DriverGraph";
import Folds from "./pages/Folds";
import Overview from "./pages/Overview";
import Placeholder from "./pages/Placeholder";
import Runs from "./pages/Runs";
import Signals from "./pages/Signals";
import { PAGES } from "./pages";

const BUILT: Record<string, React.ComponentType> = {
  "/": Overview,
  "/runs": Runs,
  "/compare": Compare,
  "/calibration": Calibration,
  "/signals": Signals,
  "/folds": Folds,
  "/data": Data,
  "/graph": DriverGraph,
};

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      ...PAGES.map((p) => {
        const Page = BUILT[p.path];
        const element = Page ? <Page /> : <Placeholder page={p} />;
        return p.path === "/" ? { index: true, element } : { path: p.path.slice(1), element };
      }),
      { path: "*", element: <Placeholder page={{ path: "", label: "Not found", title: "No such page", purpose: "Use the navigation on the left.", arrives: "never" }} /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
