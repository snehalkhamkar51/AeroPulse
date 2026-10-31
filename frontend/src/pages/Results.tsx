import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";
// history result type already inferred from context
import { usePrediction } from "@/state/PredictionContext";
import { RulDial } from "@/components/RulDial";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Play, Pause, RotateCcw } from "lucide-react";

function classify(rul: number): "Normal" | "Warning" | "Critical" {
  if (rul > 30) return "Normal";
  if (rul >= 10) return "Warning";
  return "Critical";
}

const STATUS_TEXT = {
  Normal: "Engine appears healthy. Continue routine monitoring.",
  Warning: "Plan inspection soon. Degradation is progressing.",
  Critical: "Immediate maintenance recommended.",
};

export function Results() {
  const navigate = useNavigate();
  const { request, response, selectedEngine, selectedCycleIndex, setSelectedCycleIndex, historyResult, clear } = usePrediction();
  const [predictions, setPredictions] = useState<number[] | null>(null);
  const [loadingPreds, setLoadingPreds] = useState(false);
  const [progress, setProgress] = useState(0);
  const predictionsCacheRef = useRef<Map<number, number[]>>(new Map());
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [historyEngineIndex, setHistoryEngineIndex] = useState(0);

  const selectedCycle = selectedCycleIndex >= 0 ? selectedCycleIndex : (selectedEngine ? selectedEngine.features.length - 1 : -1);
  const actualRul = useMemo(() => {
    if (!selectedEngine || selectedCycle < 0) return null;
    return selectedEngine.true_rul[selectedCycle] ?? null;
  }, [selectedEngine, selectedCycle]);

  useEffect(() => {
    if (!selectedEngine) {
      setPredictions(null);
      setLoadingPreds(false);
      setProgress(0);
      return;
    }

    const cached = predictionsCacheRef.current.get(selectedEngine.id);
    if (cached) {
      setPredictions(cached);
      setLoadingPreds(false);
      setProgress(100);
      setCurrentIndex(selectedCycle >= 0 ? selectedCycle : cached.length - 1);
      return;
    }

    // Fast path: sample engines already carry model predictions in temporal order.
    if (Array.isArray(selectedEngine.predicted_rul) && selectedEngine.predicted_rul.length === selectedEngine.features.length) {
      const out = selectedEngine.predicted_rul.map((value) => Number(value));
      predictionsCacheRef.current.set(selectedEngine.id, out);
      setPredictions(out);
      setLoadingPreds(false);
      setProgress(100);
      setCurrentIndex(selectedCycle >= 0 ? selectedCycle : out.length - 1);
      return;
    }

    // Compatibility fallback for older sample payloads without precomputed predictions.
    let cancelled = false;
    setLoadingPreds(true);
    setProgress(0);
    const rows = selectedEngine.features;
    const names = selectedEngine.feature_names;
    const out: number[] = new Array(rows.length);
    const concurrency = 4;
    let next = 0;
    let done = 0;
    const workers = Array.from({ length: Math.min(concurrency, rows.length) }, () =>
      (async () => {
        while (true) {
          const i = next++;
          if (i >= rows.length) break;
          const payload = { features: Object.fromEntries(names.map((n, j) => [n, rows[i][j]])) };
          try {
            const r = await api.predict(payload);
            if (cancelled) return;
            out[i] = r.predicted_rul;
          } catch {
            if (cancelled) return;
            out[i] = NaN;
          }
          done++;
          if (!cancelled) setProgress(Math.round((done / rows.length) * 100));
        }
      })()
    );
    Promise.all(workers).then(() => {
      if (cancelled) return;
      predictionsCacheRef.current.set(selectedEngine.id, out);
      setPredictions(out);
      setLoadingPreds(false);
      setCurrentIndex(selectedCycle >= 0 ? selectedCycle : rows.length - 1);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedEngine]);

  useEffect(() => {
    if (!playing || !predictions) return;
    const id = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= predictions.length - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 350);
    return () => clearInterval(id);
  }, [playing, predictions]);

  const replayPredicted = predictions && predictions[currentIndex] != null && Number.isFinite(predictions[currentIndex])
    ? predictions[currentIndex]
    : (response?.predicted_rul ?? 0);
  const replayStatus = classify(replayPredicted);
  const displayRul = selectedEngine && predictions ? replayPredicted : (response?.predicted_rul ?? 0);
  const displayStatus = selectedEngine && predictions ? replayStatus : (response?.health_status ?? "Normal");

  const chartData = useMemo(() => {
    if (!selectedEngine || !predictions) return [];
    return selectedEngine.cycles.map((c, i) => ({
      cycle: c,
      predicted: Number.isFinite(predictions[i]) ? predictions[i] : null,
      actual: selectedEngine.true_rul[i],
    }));
  }, [selectedEngine, predictions]);

  const retry = async () => {
    setRetrying(true);
    try {
      await api.health();
      setFetchError(null);
      window.location.reload();
    } catch {
      setFetchError("Backend still unavailable.");
    } finally {
      setRetrying(false);
    }
  };

  if (!response) {
    return (
      <div className="container">
        <div className="glass card">
          <h2>No prediction yet</h2>
          <p>Run an engine diagnostics prediction first.</p>
          <button className="btn btn-primary" onClick={() => navigate("/inputs")}>Go to Engine Inputs</button>
        </div>
      </div>
    );
  }

  if (fetchError) {
    return (
      <div className="container">
        <div className="glass card">
          <h2>Backend unavailable</h2>
          <p className="error-text">{fetchError}</p>
          <button className="btn btn-primary" onClick={retry} disabled={retrying}>
            {retrying ? "Retrying…" : "Retry"}
          </button>
        </div>
      </div>
    );
  }

  const statusClass = displayStatus === "Normal" ? "badge-normal" : displayStatus === "Warning" ? "badge-warning" : "badge-critical";

  return (
    <div className="container">
      <div className="section">
        <h1>Results Dashboard</h1>
        <p>Prediction returned by the AeroPulse backend.</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "center" }}>
        <div className="glass card" style={{ display: "grid", placeItems: "center" }}>
          <div className="card-title">RUL countdown</div>
          <RulDial value={displayRul} status={displayStatus} />
        </div>
        <div className="glass card">
          <div className="card-title">Traffic-light status</div>
          <div style={{ marginTop: 8 }}>
            <span className={`badge ${statusClass}`}>{displayStatus}</span>
          </div>
          <p style={{ marginTop: 12 }}>{STATUS_TEXT[displayStatus]}</p>
          <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-muted)" }}>
            Thresholds: Normal &gt; 30 cycles, Warning 10–30 cycles, Critical &lt; 10 cycles.
          </div>
        </div>
      </div>

      <div className="kpi-grid" style={{ marginTop: 20 }}>
        <div className="glass card">
          <div className="card-title">Regression</div>
          <div className="stat" style={{ color: "var(--cyan)" }}>{displayRul.toFixed(1)}</div>
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>predicted RUL (cycles)</div>
          <div style={{ fontSize: 12, marginTop: 8, color: "var(--text-dim)" }}>Likely range: {response.rul_low.toFixed(1)}–{response.rul_high.toFixed(1)} cycles</div>
        </div>
        <div className="glass card">
          <div className="card-title">Classification</div>
          <div style={{ marginTop: 6 }}>
            <span className={`badge ${displayStatus === "Normal" ? "badge-normal" : displayStatus === "Warning" ? "badge-warning" : "badge-critical"}`}>
              {displayStatus}
            </span>
          </div>
          <p style={{ marginTop: 10, fontSize: 13 }}>Rule-based from predicted RUL thresholds (not a separate classifier).</p>
        </div>
        {actualRul != null && (
          <div className="glass card">
            <div className="card-title">Validation sample</div>
            <div style={{ fontSize: 13 }}>
              Actual RUL: {actualRul.toFixed(1)} cycles<br />
              Predicted: {response.predicted_rul.toFixed(1)} cycles<br />
              Error: {(response.predicted_rul - actualRul).toFixed(1)} cycles
            </div>
          </div>
        )}
      </div>

      {historyResult && historyResult.engines.length > 0 && (
        <div className="section" style={{ marginTop: 24 }}>
          <div className="glass card">
            <div className="card-title">Uploaded history</div>
            {historyResult.engines.length > 1 && (
              <select
                value={historyEngineIndex}
                onChange={(e) => setHistoryEngineIndex(Number(e.target.value))}
                style={{ marginBottom: 12, background: "rgba(5,10,20,0.6)", border: "1px solid var(--border)", color: "var(--text)", padding: "8px 10px", borderRadius: 8 }}
              >
                {historyResult.engines.map((e, i) => (
                  <option key={e.engine_id} value={i}>
                    Engine {e.engine_id}
                  </option>
                ))}
              </select>
            )}
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyResult.engines[historyEngineIndex]?.rul_by_cycle.map((p) => ({ cycle: p.cycle, predicted: p.predicted_rul })) ?? []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="cycle" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip />
                  <Line type="monotone" dataKey="predicted" stroke="#22d3ee" dot={false} isAnimationActive={false} name="Predicted RUL" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      <div className="section" style={{ marginTop: 24 }}>
        <div className="glass card">
          <div className="card-title">Submitted features</div>
          {request?.features && (
            <div style={{ maxHeight: 220, overflow: "auto", fontSize: 13 }}>
              {Object.entries(request.features).map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border)", padding: "4px 0" }}>
                  <span style={{ fontFamily: "monospace" }}>{k}</span>
                  <span>{v}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {selectedEngine && (
        <div className="section">
          <div className="glass card">
            <div className="card-title">Replay this engine</div>
            {loadingPreds && <p>Loading per-cycle predictions… {progress}%</p>}
            {!loadingPreds && predictions && (
              <>
                <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
                  <button className="btn btn-ghost" onClick={() => setPlaying((p) => !p)}>
                    {playing ? <Pause size={16} /> : <Play size={16} />} {playing ? "Pause" : "Play"}
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={predictions.length - 1}
                    value={currentIndex}
                    onChange={(e) => {
                      setPlaying(false);
                      setCurrentIndex(Number(e.target.value));
                      setSelectedCycleIndex(Number(e.target.value));
                    }}
                    style={{ flex: 1, minWidth: 200 }}
                  />
                  <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    Cycle {selectedEngine.cycles[currentIndex]} / {predictions.length} cycles
                  </span>
                </div>
                <div style={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="cycle" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Line type="monotone" dataKey="predicted" stroke="#22d3ee" dot={false} isAnimationActive={false} name="Predicted RUL" />
                      <Line type="monotone" dataKey="actual" stroke="#f59e0b" dot={false} isAnimationActive={false} name="True RUL" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 16 }}>
        <button className="btn btn-ghost" onClick={() => { clear(); navigate("/inputs"); }}>
          <RotateCcw size={16} /> Run Another Prediction
        </button>
      </div>
    </div>
  );
}

