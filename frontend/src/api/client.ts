import type { EvaluationDataResponse, HealthResponse, ModelInfoResponse, PredictHistoryResponse, PredictRequest, PredictResponse, SampleEnginesResponse } from "@/types/api";

const rawBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";
const API_BASE = rawBase.replace(/\/+$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  const isForm = init?.body instanceof FormData;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: isForm ? { ...(init?.headers ?? {}) } : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new Error("Backend is unavailable. Please start the FastAPI server.");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  modelInfo: () => request<ModelInfoResponse>("/model-info"),
  sampleEngines: () => request<SampleEnginesResponse>("/sample-engines"),
  predictHistory: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<PredictHistoryResponse>("/predict-history", { method: "POST", body: form, headers: {} });
  },
  evaluationData: () => request<EvaluationDataResponse>("/evaluation-data"),
  predict: (payload: PredictRequest) =>
    request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify(payload) }),
};

export { API_BASE };
