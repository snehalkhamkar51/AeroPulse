// RulDial


interface Props {
  value: number;
  max?: number;
  status: "Normal" | "Warning" | "Critical";
}

const COLORS = {
  Normal: "#22c55e",
  Warning: "#f59e0b",
  Critical: "#ef4444",
};

export function RulDial({ value, max = 125, status }: Props) {
  const clamped = Math.max(0, Math.min(max, value));
  const radius = 78;
  const stroke = 14;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped / max);
  const color = COLORS[status];

  return (
    <div style={{ display: "grid", placeItems: "center" }}>
      <svg width="200" height="200" viewBox="0 0 200 200" role="img" aria-label={`RUL dial ${clamped.toFixed(1)} cycles`}>
        <circle cx="100" cy="100" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={stroke} />
        <circle
          cx="100"
          cy="100"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 100 100)"
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.3s ease" }}
        />
        <text x="100" y="92" textAnchor="middle" fill="var(--text)" fontSize="34" fontWeight="800">
          {clamped.toFixed(0)}
        </text>
        <text x="100" y="118" textAnchor="middle" fill="var(--text-muted)" fontSize="12">
          cycles
        </text>
      </svg>
    </div>
  );
}
