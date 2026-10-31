import { useEffect, useRef, useState, type MouseEvent } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Rocket, Activity, ShieldCheck, Layers, ArrowRight } from "lucide-react";
import { api } from "@/api/client";
import type { ModelInfoResponse } from "@/types/api";

const HOTSPOTS = [
  { id: "fan", label: "Fan", x: 19, y: 52 },
  { id: "compressor", label: "Compressor", x: 40, y: 49 },
  { id: "combustor", label: "Combustor", x: 58, y: 42 },
  { id: "turbine", label: "Turbine", x: 78, y: 32 },
];

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55 } },
};

export function Landing() {
  const [imageFailed, setImageFailed] = useState(false);
  const [parallax, setParallax] = useState({ x: 0, y: 0 });
  const target = useRef({ x: 0, y: 0 });
  const reducedMotion =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (reducedMotion) return;
    let frame = 0;
    const tick = () => {
      setParallax((current) => ({
        x: current.x + (target.current.x - current.x) * 0.08,
        y: current.y + (target.current.y - current.y) * 0.08,
      }));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [reducedMotion]);

  const handleMove = (event: MouseEvent<HTMLDivElement>) => {
    if (reducedMotion) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width - 0.5;
    const ny = (event.clientY - rect.top) / rect.height - 0.5;
    target.current = { x: -nx * 12, y: -ny * 12 };
  };

  const handleLeave = () => {
    target.current = { x: 0, y: 0 };
  };

  const [featureLabel, setFeatureLabel] = useState("Multi-sensor health indicators");
  useEffect(() => {
    api
      .modelInfo()
      .then((info: ModelInfoResponse) => {
        if (typeof info.n_features === "number") setFeatureLabel(`${info.n_features} selected features`);
      })
      .catch(() => setFeatureLabel("Multi-sensor health indicators"));
  }, []);

  return (
    <div className="container">
      <section className="hero hero-first">
        <motion.div initial="hidden" animate="show" variants={fadeUp}>
          <div className="badge badge-normal" style={{ marginBottom: 16 }}>Educational prototype</div>
          <h1 className="hero-title">AeroPulse</h1>
          <p className="hero-sub">AI-Powered Engine Degradation & RUL Countdown Tracker</p>
          <p>
            AeroPulse estimates an aircraft turbofan engine's Remaining Useful Life (RUL) in operational cycles,
            tracks degradation, and communicates an estimated health status using a trained Random Forest model
            on the NASA C-MAPSS dataset.
          </p>
          <div style={{ display: "flex", gap: 12, marginTop: 20, flexWrap: "wrap" }}>
            <Link to="/inputs" className="btn btn-primary">
              <Rocket size={18} /> Launch Engine Diagnostics
            </Link>
            <Link to="/about" className="btn btn-ghost">Learn more</Link>
          </div>
          <div className="kpi-grid" style={{ marginTop: 24 }}>
            {[
              { icon: <Activity size={18} />, title: "RUL Estimation", desc: "Predicted remaining cycles" },
              { icon: <Layers size={18} />, title: "Sensor Analysis", desc: featureLabel },
              { icon: <ShieldCheck size={18} />, title: "Health Status", desc: "Normal / Warning / Critical" },
            ].map((k) => (
              <div key={k.title} className="glass card">
                <div style={{ color: "var(--cyan)", marginBottom: 6 }}>{k.icon}</div>
                <div style={{ fontWeight: 700 }}>{k.title}</div>
                <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{k.desc}</div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          className="glass hero-visual"
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7 }}
          onMouseMove={handleMove}
          onMouseLeave={handleLeave}
          style={{ x: parallax.x, y: parallax.y }}
        >
          <div className="hero-glow" />
          <motion.div
            className="hero-float"
            animate={reducedMotion ? {} : { y: [0, -10, 0], scale: [1, 1.03, 1] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          >
            {!imageFailed ? (
              <img
                src="/media/engine-hero.png"
                alt="Holographic turbofan engine cutaway used as the AeroPulse visual centerpiece"
                loading="eager"
                width={960}
                height={720}
                onError={() => {
                  setImageFailed(true);
                  if (import.meta.env.DEV) {
                    console.warn("Engine hero image missing at /public/media/engine-hero.png; using placeholder.");
                  }
                }}
              />
            ) : (
              <div className="hero-placeholder" role="img" aria-label="Placeholder for the AeroPulse engine image" />
            )}
          </motion.div>
          <div className="hero-scan" />
          <div className="hero-hotspots">
            {HOTSPOTS.map((spot) => (
              <button
                key={spot.id}
                type="button"
                className="hotspot"
                style={{ left: `${spot.x}%`, top: `${spot.y}%` }}
                aria-label={spot.label}
              >
                <span className="hotspot-dot" />
                <span className="hotspot-tooltip">{spot.label}</span>
              </button>
            ))}
          </div>
        </motion.div>
      </section>

      <motion.section className="section" initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.25 }} variants={fadeUp}>
        <div className="glass card">
          <h2>The Problem</h2>
          <p>
            Turbofan engines degrade over time. Estimating Remaining Useful Life helps plan maintenance before failure,
            but predictions are statistical estimates and not certified maintenance decisions.
          </p>
        </div>
      </motion.section>

      <motion.section className="section" initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.25 }} variants={fadeUp}>
        <div className="glass card">
          <h2>How it works</h2>
          <div className="kpi-grid" style={{ marginTop: 12 }}>
            {[
              { n: "1", t: "Collect sensors", d: "Operating settings and 21 sensor channels per cycle." },
              { n: "2", t: "Preprocess", d: "Drop constant features, cap RUL at 125, split by engine." },
              { n: "3", t: "Predict", d: "Random Forest estimates RUL; thresholds set status." },
            ].map((s) => (
              <div key={s.n} className="glass card">
                <div style={{ color: "var(--cyan)", fontWeight: 800, fontSize: 22 }}>{s.n}</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>{s.t}</div>
                <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>{s.d}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section className="section" initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.25 }} variants={fadeUp}>
        <div className="glass card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <div>
            <h2>Try it</h2>
            <p>Load a sample engine, predict RUL, and explore the countdown dashboard.</p>
          </div>
          <Link to="/inputs" className="btn btn-primary">
            <ArrowRight size={18} /> Launch Engine Diagnostics
          </Link>
        </div>
      </motion.section>

      <motion.section className="section" initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.25 }} variants={fadeUp}>
        <div className="glass card">
          <h3>What is Remaining Useful Life?</h3>
          <p>
            RUL is the estimated number of operational cycles an engine can continue running before failure.
            AeroPulse uses multivariate sensor and operating-setting data to estimate it, then classifies health
            using transparent thresholds (Normal &gt; 30, Warning 10–30, Critical &lt; 10 cycles).
          </p>
          <p style={{ marginTop: 8, fontSize: 13 }}>
            This is an educational predictive-maintenance prototype, not a certified aviation diagnostic or maintenance system.
          </p>
        </div>
      </motion.section>
    </div>
  );
}

