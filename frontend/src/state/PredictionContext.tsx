import React, { createContext, useCallback, useContext, useState } from "react";
import type { PredictHistoryResponse, PredictRequest, PredictResponse, SampleEngine } from "@/types/api";

interface PredictionState {
  request: PredictRequest | null;
  response: PredictResponse | null;
  selectedEngine: SampleEngine | null;
  selectedCycleIndex: number;
  historyResult: PredictHistoryResponse | null;
  setPrediction: (req: PredictRequest, res: PredictResponse, engine?: SampleEngine | null, cycleIndex?: number) => void;
  setHistoryResult: (result: PredictHistoryResponse | null) => void;
  setSelectedCycleIndex: (index: number) => void;
  clear: () => void;
}

const Ctx = createContext<PredictionState | undefined>(undefined);

export function PredictionProvider({ children }: { children: React.ReactNode }) {
  const [request, setRequest] = useState<PredictRequest | null>(null);
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [selectedEngine, setSelectedEngine] = useState<SampleEngine | null>(null);
  const [selectedCycleIndex, setSelectedCycleIndexState] = useState<number>(-1);
  const [historyResult, setHistoryResultState] = useState<PredictHistoryResponse | null>(null);

  const setPrediction = useCallback((r: PredictRequest, s: PredictResponse, engine?: SampleEngine | null, cycleIndex?: number) => {
    setRequest(r);
    setResponse(s);
    setHistoryResultState(null);
    if (engine !== undefined) setSelectedEngine(engine);
    if (cycleIndex !== undefined) setSelectedCycleIndexState(cycleIndex);
  }, []);
  const setHistoryResult = useCallback((result: PredictHistoryResponse | null) => {
    setHistoryResultState(result);
    if (result && result.engines.length > 0) {
      const latest = result.engines[0];
      setResponse({
        predicted_rul: latest.predicted_rul,
        health_status: latest.health_status,
        explanation: "Prediction from uploaded engine history.",
        features_used: result.feature_names.length,
        feature_contributions: [],
        model_ready: true,
        thresholds: { normal_gt: 30, warning_min: 10, warning_max: 30, critical_lt: 10 },
        model_version: result.model_version,
        rul_low: latest.rul_low,
        rul_high: latest.rul_high,
      });
      setRequest(null);
      setSelectedEngine(null);
      setSelectedCycleIndexState(-1);
    } else {
      setResponse(null);
    }
  }, []);
  const setSelectedCycleIndex = useCallback((index: number) => {
    setSelectedCycleIndexState(index);
  }, []);
  const clear = useCallback(() => {
    setRequest(null);
    setResponse(null);
    setSelectedEngine(null);
    setSelectedCycleIndexState(-1);
    setHistoryResultState(null);
  }, []);

  return (
    <Ctx.Provider value={{ request, response, selectedEngine, selectedCycleIndex, historyResult, setPrediction, setHistoryResult, setSelectedCycleIndex, clear }}>
      {children}
    </Ctx.Provider>
  );
}

export function usePrediction() {
  const v = useContext(Ctx);
  if (!v) throw new Error("usePrediction must be used within PredictionProvider");
  return v;
}
