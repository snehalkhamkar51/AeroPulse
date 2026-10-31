import React from "react";
import { describe, it, expect, vi } from "vitest";
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("@/three/EngineScene", () => ({
  EngineCanvas: () => <div data-testid="engine-canvas" />,
}));
vi.mock("framer-motion", () => ({
  motion: {
    div: (props: React.ComponentProps<"div">) => <div {...props} />,
    section: (props: React.ComponentProps<"section">) => <section {...props} />,
  },
}));
vi.mock("@/api/client", () => ({
  api: {
    health: vi.fn().mockResolvedValue({ status: "ok", model_ready: true }),
    modelInfo: vi.fn().mockResolvedValue({
      status: "ready",
      feature_names: ["op_setting_1", "sensor_2"],
      feature_importances: [{ feature: "sensor_2", importance: 0.5 }],
      n_features: 2,
    }),
    sampleEngines: vi.fn().mockResolvedValue({
      feature_names: ["op_setting_1", "sensor_2"],
      feature_stats: { op_setting_1: { median: 0, min: -1, max: 1 }, sensor_2: { median: 1, min: 0, max: 2 } },
      engines: [{ id: 1, category: "healthy", final_rul: 100, cycles: [1], feature_names: ["op_setting_1", "sensor_2"], features: [[0, 1]], true_rul: [100] }],
    }),
    predict: vi.fn(),
  },
}));

import App from "@/App";

describe("AeroPulse app", () => {
  it("renders landing page", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.getAllByText(/AeroPulse/i).length).toBeGreaterThan(0);
    expect(await screen.findByText(/AI-Powered Engine Degradation/i)).toBeInTheDocument();
  });

  it("renders engine inputs page", async () => {
    render(
      <MemoryRouter initialEntries={["/inputs"]}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByText(/Load sample engine/i)).toBeInTheDocument();
  });

  it("removes the explainability route", async () => {
    render(
      <MemoryRouter initialEntries={["/explainability"]}>
        <App />
      </MemoryRouter>,
    );
    const matches = await screen.findAllByText(/AI-Powered Engine Degradation/i);
    expect(matches.length).toBeGreaterThan(0);
    expect(screen.queryByText(/Explainability/i)).not.toBeInTheDocument();
  });
});

vi.mock("@/state/PredictionContext", () => ({
  PredictionProvider: ({ children }: { children: React.ReactNode }) => children,
  usePrediction: () => ({
    request: { features: {} },
    response: {
      predicted_rul: 42.5,
      health_status: "Warning",
      explanation: "test",
      features_used: 7,
      feature_contributions: [],
      model_ready: true,
      thresholds: {},
      model_version: "test",
      rul_low: 30,
      rul_high: 55,
    },
    selectedEngine: null,
    selectedCycleIndex: -1,
    historyResult: null,
    setPrediction: vi.fn(),
    setHistoryResult: vi.fn(),
    setSelectedCycleIndex: vi.fn(),
    clear: vi.fn(),
  }),
}));

import { Results } from "@/pages/Results";

describe("Results consistency", () => {
  it("uses one predicted RUL for dial, status and cards", () => {
    render(
      <MemoryRouter initialEntries={["/results"]}>
        <Results />
      </MemoryRouter>,
    );
    expect(screen.getAllByText(/42\.5/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Warning").length).toBeGreaterThan(0);
    expect(screen.getByText(/Likely range: 30.0–55.0/)).toBeInTheDocument();
    expect(screen.queryByText(/Typical error/)).not.toBeInTheDocument();
  });
});
