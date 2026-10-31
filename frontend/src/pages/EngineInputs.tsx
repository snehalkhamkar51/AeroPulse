import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";
import type { PredictHistoryResponse, SampleEnginesResponse, SampleEngine } from "@/types/api";
import { usePrediction } from "@/state/PredictionContext";
import { RefreshCw, Send, AlertTriangle } from "lucide-react";

const FEATURE_META: Record<string, { label: string; unit: string; group: string }> = {
  cycle: { label: "Operational cycle (model input)", unit: "cycles", group: "cycle" },
  sensor_2: { label: "LPC outlet temperature", unit: "deg R", group: "temperature" },
  sensor_3: { label: "HPC outlet temperature", unit: "deg R", group: "temperature" },
  sensor_4: { label: "LPT outlet temperature", unit: "deg R", group: "temperature" },
  sensor_6: { label: "Total pressure at bypass duct", unit: "psia", group: "pressure" },
  sensor_7: { label: "HPC outlet pressure", unit: "psia", group: "pressure" },
  sensor_8: { label: "Physical fan speed", unit: "rpm", group: "speed" },
  sensor_9: { label: "Physical core speed", unit: "rpm", group: "speed" },
  sensor_11: { label: "Static pressure at HPC outlet", unit: "psia", group: "pressure" },
  sensor_12: { label: "Fuel flow / Ps30 ratio", unit: "pps/psi", group: "other" },
  sensor_13: { label: "Corrected fan speed", unit: "rpm", group: "speed" },
  sensor_14: { label: "Corrected core speed", unit: "rpm", group: "speed" },
  sensor_15: { label: "Bypass ratio", unit: "dimensionless", group: "other" },
  sensor_17: { label: "Bleed enthalpy", unit: "raw sensor units", group: "other" },
  sensor_20: { label: "HPT coolant bleed", unit: "lbm/s", group: "other" },
  sensor_21: { label: "LPT coolant bleed", unit: "lbm/s", group: "other" },
};

const GROUPS: { key: string; title: string; features: string[] }[] = [
  { key: "cycle", title: "Engine condition (model input)", features: ["cycle"] },
  { key: "temperature", title: "Temperatures", features: ["sensor_2", "sensor_3", "sensor_4"] },
  { key: "pressure", title: "Pressures", features: ["sensor_6", "sensor_7", "sensor_11"] },
  { key: "speed", title: "Rotation speeds", features: ["sensor_8", "sensor_9", "sensor_13", "sensor_14"] },
  { key: "other", title: "Other model features", features: ["sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"] },
];

export function EngineInputs() {
  const navigate = useNavigate();
  const { setPrediction, setHistoryResult } = usePrediction();
  const [samples, setSamples] = useState<SampleEnginesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEngineId, setSelectedEngineId] = useState<number | "">("");
  const [selectedCycleIndex, setSelectedCycleIndex] = useState<number>(-1);
  const [engineId, setEngineId] = useState<string>("");
  const [cycle, setCycle] = useState<string>("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [allTouched, setAllTouched] = useState(false);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvResult, setCsvResult] = useState<PredictHistoryResponse | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    api
      .sampleEngines()
      .then((data) => {
        setSamples(data);
        const init: Record<string, string> = {};
        data.feature_names.forEach((f) => {
          init[f] = String(data.feature_stats[f]?.median ?? "");
        });
        setValues(init);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const features = samples?.feature_names ?? [];
  const stats = samples?.feature_stats ?? {};


  const loadEngine = (engine: SampleEngine, cycleIndex?: number) => {
    const idx = cycleIndex !== undefined && cycleIndex >= 0 ? cycleIndex : engine.features.length - 1;
    const row = engine.features[idx];
    const next: Record<string, string> = { ...values };
    engine.feature_names.forEach((f, i) => {
      next[f] = String(row[i]);
    });
    setValues(next);
    setEngineId(String(engine.id));
    setCycle(String(engine.cycles[idx]));
    setTouched({});
    setAllTouched(false);
    setSelectedEngineId(engine.id);
    setSelectedCycleIndex(idx);
  };

  const loadQuick = (category: string) => {
    const eng = samples?.engines.find((e) => e.category === category);
    if (eng) loadEngine(eng);
  };

  const handleChange = (k: string, v: string) => {
    setValues((prev) => ({ ...prev, [k]: v }));
    setTouched((prev) => ({ ...prev, [k]: true }));
  };

  const handleReset = () => {
    if (!samples) return;
    const init: Record<string, string> = {};
    samples.feature_names.forEach((f) => {
      init[f] = String(samples.feature_stats[f]?.median ?? "");
    });
    setValues(init);
    setEngineId("");
    setCycle("");
    setTouched({});
    setAllTouched(false);
    setSelectedEngineId("");
    setSelectedCycleIndex(-1);
    setSubmitError(null);
  };

  const handleRandomize = () => {
    if (!samples) return;
    const next: Record<string, string> = { ...values };
    samples.feature_names.forEach((f) => {
      const st = samples.feature_stats[f];
      if (!st) return;
      const span = st.max - st.min;
      const value = st.min + Math.random() * span;
      next[f] = String(value);
    });
    setValues(next);
    setTouched({});
    setAllTouched(false);
  };

  const handleUpload = async () => {
    if (!csvFile) return;
    setUploading(true);
    setCsvError(null);
    try {
      const result = await api.predictHistory(csvFile);
      setCsvResult(result);
      setHistoryResult(result);
    } catch (err) {
      setCsvError(err instanceof Error ? err.message : "Upload failed");
      setCsvResult(null);
    } finally {
      setUploading(false);
    }
  };

  const downloadTemplate = () => {
    if (!samples) return;
    const header = ["engine_id", "cycle", ...samples.feature_names].join(",");
    const medianRow = ["1", "5", ...samples.feature_names.map((f) => samples.feature_stats[f]?.median ?? "")].join(",");
    const blob = new Blob([header + "\n" + medianRow + "\n"], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "aeropulse_history_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadResults = () => {
    if (!csvResult) return;
    const header = ["engine_id", "latest_cycle", "predicted_rul", "health_status", "rul_low", "rul_high"].join(",");
    const rows = csvResult.engines.map((e) =>
      [e.engine_id, e.latest_cycle, e.predicted_rul, e.health_status, e.rul_low, e.rul_high].join(",")
    );
    const blob = new Blob([header + "\n" + rows.join("\n") + "\n"], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "aeropulse_batch_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAllTouched(true);
    if (!allValid) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const payload: Record<string, number> = {};
      features.forEach((k) => (payload[k] = Number(values[k])));
      const res = await api.predict({ features: payload, unit_id: engineId ? Number(engineId) : undefined });
      const selected = samples?.engines.find((e) => e.id === (engineId ? Number(engineId) : -1)) ?? null;
      setPrediction({ features: payload, unit_id: engineId ? Number(engineId) : undefined }, res, selected, selectedCycleIndex >= 0 ? selectedCycleIndex : undefined);
      navigate("/results");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setSubmitting(false);
    }
  };

  const showError = (k: string) => allTouched || touched[k];

  const validate = useMemo(() => {
    const errs: Record<string, string> = {};
    features.forEach((k) => {
      const raw = values[k]?.trim() ?? "";
      if (raw === "") {
        errs[k] = "Required";
        return;
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) errs[k] = "Must be a finite number";
    });
    return errs;
  }, [features, values]);

  const allValid = features.length > 0 && Object.keys(validate).length === 0;

  if (loading) return <div className="container"><p>Loading sample engines and training statistics…</p></div>;
  if (error) {
    return (
      <div className="container">
        <div className="glass card">
          <h2>Backend unavailable</h2>
          <p className="error-text">{error}</p>
          <p>Start the FastAPI backend on port 8000 and retry.</p>
        </div>
      </div>
    );
  }

  const selectedEngine = samples?.engines.find((e) => e.id === selectedEngineId);

  return (
    <div className="container">
      <div className="section">
        <h1>Engine Inputs</h1>
        <p>
          Fill in the engine readings below, or load a sample engine. Values are sent in the exact trained order. Engine ID and cycle are optional metadata.
        </p>
      </div>

      <div className="glass card" style={{ marginBottom: 16 }}>
        <div className="card-title">Load sample engine</div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <select
            value={selectedEngineId}
            onChange={(e) => {
              const id = Number(e.target.value);
              const eng = samples?.engines.find((x) => x.id === id);
              if (eng) loadEngine(eng);
            }}
            style={{ background: "rgba(5,10,20,0.6)", border: "1px solid var(--border)", color: "var(--text)", padding: "10px 12px", borderRadius: 10 }}
          >
            <option value="">Select an engine…</option>
            {samples?.engines.map((e) => (
              <option key={e.id} value={e.id}>
                Engine {e.id} — {e.category} (final RUL {e.final_rul.toFixed(0)})
              </option>
            ))}
          </select>
          <button type="button" className="btn btn-ghost" onClick={() => loadQuick("healthy")}>Healthy example</button>
          <button type="button" className="btn btn-ghost" onClick={() => loadQuick("mid-life")}>Mid-life example</button>
          <button type="button" className="btn btn-ghost" onClick={() => loadQuick("near-failure")}>Near-failure example</button>
        </div>
        {selectedEngine && (
          <div style={{ marginTop: 10, fontSize: 13, color: "var(--text-muted)" }}>
            Loaded engine {selectedEngine.id}; cycle selector:
            <select
              value={selectedCycleIndex}
              onChange={(e) => loadEngine(selectedEngine, Number(e.target.value))}
              style={{ marginLeft: 8, background: "rgba(5,10,20,0.6)", border: "1px solid var(--border)", color: "var(--text)", padding: "6px 8px", borderRadius: 8 }}
            >
              {selectedEngine.cycles.map((c, i) => (
                <option key={c} value={i}>
                  Cycle {c}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {submitError && (
        <div className="glass card" style={{ marginBottom: 16, borderColor: "var(--red)" }}>
          <div className="error-text">{submitError}</div>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="glass card" style={{ marginBottom: 16 }}>
          <div className="card-title">A. Identification (not used by the model)</div>
          <div className="input-grid">
            <div className="field">
              <label>Engine ID</label>
              <input value={engineId} onChange={(e) => setEngineId(e.target.value)} placeholder="optional metadata" />
            </div>
            <div className="field">
              <label>Operational cycle</label>
              <input value={cycle} onChange={(e) => setCycle(e.target.value)} placeholder="optional metadata" />
            </div>
          </div>
        </div>

        {GROUPS.map((group) => (
          <div className="glass card" style={{ marginBottom: 16 }} key={group.key}>
            <div className="card-title">{group.title}</div>
            <div className="input-grid">
              {group.features.map((k) => {
                const meta = FEATURE_META[k];
                const st = stats[k];
                const raw = values[k] ?? "";
                const n = Number(raw);
                const outOfRange = !!st && raw !== "" && Number.isFinite(n) && (n < st.min || n > st.max);
                return (
                  <div className="field" key={k}>
                    <label>{meta.label}</label>
                    <span className="key">{k} · {meta.unit}</span>
                    <input
                      type="number"
                      step="any"
                      value={raw}
                      placeholder={st ? st.median.toFixed(3) : ""}
                      title={st ? `Typical range: ${st.min.toFixed(3)} to ${st.max.toFixed(3)}` : ""}
                      onChange={(e) => handleChange(k, e.target.value)}
                      onBlur={() => setTouched((prev) => ({ ...prev, [k]: true }))}
                    />
                    {showError(k) && validate[k] && <span className="error-text">{validate[k]}</span>}
                    {outOfRange && <span style={{ fontSize: 11, color: "var(--amber)" }}>Outside observed training range (soft warning)</span>}
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
          <button type="submit" className="btn btn-primary" disabled={!allValid || submitting}>
            <Send size={16} /> {submitting ? "Predicting…" : "Predict RUL"}
          </button>
          <button type="button" className="btn btn-ghost" onClick={handleReset}>
            <RefreshCw size={16} /> Reset to medians
          </button>
          <button type="button" className="btn btn-ghost" onClick={handleRandomize}>
            Randomize within typical range
          </button>
        </div>
        {!allValid && allTouched && (
          <p className="error-text" style={{ marginTop: 12 }}>
            <AlertTriangle size={14} style={{ verticalAlign: "middle" }} /> Complete all fields with finite numeric values to enable prediction.
          </p>
        )}
      </form>

      <div className="glass card" style={{ marginTop: 24 }}>
        <div className="card-title">Upload engine history (CSV)</div>
        <p style={{ fontSize: 13, marginBottom: 12 }}>
          Required columns: <code>cycle</code> and {features.join(", ")}. Optional: <code>engine_id</code> for multiple engines.
        </p>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <input type="file" accept=".csv,text/csv" onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)} />
          <button type="button" className="btn btn-ghost" onClick={downloadTemplate}>Download template CSV</button>
          <button type="button" className="btn btn-primary" onClick={handleUpload} disabled={!csvFile || uploading}>
            {uploading ? "Uploading…" : "Predict history"}
          </button>
        </div>
        {csvError && <p className="error-text" style={{ marginTop: 12 }}>{csvError}</p>}
        {csvResult && (
          <div style={{ marginTop: 16 }}>
            <p style={{ fontSize: 13 }}>Predictions for {csvResult.engines.length} engine(s). Results are also shown on the Results page.</p>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "6px 4px" }}>Engine</th>
                    <th style={{ textAlign: "right", padding: "6px 4px" }}>Latest cycle</th>
                    <th style={{ textAlign: "right", padding: "6px 4px" }}>Predicted RUL</th>
                    <th style={{ textAlign: "left", padding: "6px 4px" }}>Status</th>
                    <th style={{ textAlign: "right", padding: "6px 4px" }}>Range</th>
                  </tr>
                </thead>
                <tbody>
                  {csvResult.engines.map((e) => (
                    <tr key={e.engine_id} style={{ borderTop: "1px solid var(--border)" }}>
                      <td style={{ padding: "6px 4px" }}>{e.engine_id}</td>
                      <td style={{ padding: "6px 4px", textAlign: "right" }}>{e.latest_cycle}</td>
                      <td style={{ padding: "6px 4px", textAlign: "right" }}>{e.predicted_rul.toFixed(1)}</td>
                      <td style={{ padding: "6px 4px" }}>{e.health_status}</td>
                      <td style={{ padding: "6px 4px", textAlign: "right" }}>{e.rul_low.toFixed(1)}–{e.rul_high.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button type="button" className="btn btn-ghost" style={{ marginTop: 12 }} onClick={downloadResults}>
              Download results CSV
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

