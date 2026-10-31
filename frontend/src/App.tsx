import { Suspense, lazy } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { PredictionProvider } from "@/state/PredictionContext";
import { Layout } from "@/components/Layout";

const Landing = lazy(() => import("@/pages/Landing").then((m) => ({ default: m.Landing })));
const EngineInputs = lazy(() => import("@/pages/EngineInputs").then((m) => ({ default: m.EngineInputs })));
const Results = lazy(() => import("@/pages/Results").then((m) => ({ default: m.Results })));
const About = lazy(() => import("@/pages/About").then((m) => ({ default: m.About })));

function PageFallback() {
  return (
    <div className="container" role="status" aria-live="polite">
      <p>Loading page…</p>
    </div>
  );
}

export default function App() {
  return (
    <PredictionProvider>
      <Layout>
        <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/inputs" element={<EngineInputs />} />
            <Route path="/results" element={<Results />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </Layout>
    </PredictionProvider>
  );
}

