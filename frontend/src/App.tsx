import { useEffect } from "react";
import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { Header } from "./components/Header.tsx";
import { Footer } from "./components/Footer.tsx";
import { Console } from "./pages/Console.tsx";
import { Model } from "./pages/Model.tsx";

const TITLES: Array<[RegExp, string]> = [
  [/^\/model\/?$/, "Model Intel — Signal Lab"],
  [/^\/?$/, "Signal Lab — Voice AI Detector"],
];

function RouteTitle() {
  const { pathname } = useLocation();
  useEffect(() => {
    const hit = TITLES.find(([re]) => re.test(pathname));
    document.title = hit ? hit[1] : "Not found — Signal Lab";
  }, [pathname]);
  return null;
}

function NotFound() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 md:px-6">
      <div className="max-w-[65ch] rounded-[20px] border border-hairline bg-panel p-8">
        <p className="font-mono text-xs tracking-[0.08em] text-dim">
          404 // OFF THE GRID
        </p>
        <h2 className="mt-2 font-display text-2xl font-bold text-ink">
          No bench at this address
        </h2>
        <Link
          to="/"
          className="mt-4 inline-block min-h-11 rounded-xl bg-teal px-5 py-2.5 font-semibold leading-7 text-void"
        >
          Back to the console
        </Link>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <RouteTitle />
      <div className="flex min-h-dvh flex-col bg-void text-ink">
        <Header />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Console />} />
            <Route path="/model" element={<Model />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
