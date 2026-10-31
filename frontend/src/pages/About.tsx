import { DegradationStory } from "@/three/DegradationStory";

export function About() {
  return (
    <div className="container">
      <div className="section">
        <h1>About AeroPulse</h1>
        <p>AI-Powered Engine Degradation & RUL Countdown Tracker — an educational prototype.</p>
      </div>

      <div className="glass card section">
        <div className="card-title">Problem statement & objectives</div>
        <p>
          Aircraft engines degrade with use. AeroPulse estimates Remaining Useful Life (RUL) in cycles from
          operating settings and sensor measurements, then maps the estimate to a traffic-light health status.
        </p>
      </div>

      <div className="glass card section">
        <div className="card-title">NASA C-MAPSS</div>
        <p>
          The model uses the FD001 subset: one operating condition, one fault mode. FD002–FD004 add more
          operating conditions and fault modes; this prototype intentionally uses FD001 only for a controlled baseline.
        </p>
      </div>

      <div className="glass card section">
        <div className="card-title">ML pipeline</div>
        <p>
          RandomForestRegressor on 16 selected features. RUL labels are max cycle minus current cycle, capped at 125.
          Validation is split by engine. Metrics: validation MAE 12.89 / RMSE 18.30; official test MAE 13.12 / RMSE 18.01.
        </p>
      </div>

      <div className="glass card section">
        <div className="card-title">Tech stack</div>
        <p>Python, pandas, NumPy, scikit-learn, FastAPI, React, TypeScript, Vite, React Three Fiber, Recharts.</p>
      </div>

      <div className="glass card section">
        <div className="card-title">Degradation story</div>
        <p>
          Healthy turbofan → gradual erosion and heat → warning → critical. The RUL countdown is synchronized to the animation.
        </p>
        <div style={{ marginTop: 12 }}>
          <DegradationStory />
        </div>
      </div>

      <div className="glass card section">
        <div className="card-title">Limitations & future work</div>
        <p>
          FD001 only; global feature importance, not per-prediction attribution; simulated data; not certified for maintenance decisions.
        </p>
      </div>
    </div>
  );
}
