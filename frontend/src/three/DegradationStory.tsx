import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows, Html } from "@react-three/drei";
import * as THREE from "three";

const DURATION = 18;

function DegradationEngine({ playing, reducedMotion }: { playing: boolean; reducedMotion: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const fanRef = useRef<THREE.Group>(null);
  const bodyMat = useRef<THREE.MeshStandardMaterial>(null);
  const coreMat = useRef<THREE.MeshStandardMaterial>(null);
  const nozzleMat = useRef<THREE.MeshStandardMaterial>(null);
  const progressRef = useRef(0);
  const [rul, setRul] = useState(125);

  useFrame((_, delta) => {
    if (playing && !reducedMotion) {
      progressRef.current = (progressRef.current + delta / DURATION) % 1;
    }
    const t = progressRef.current;
    const currentRul = 125 * (1 - t);
    if (Math.abs(currentRul - rul) > 0.5) setRul(currentRul);

    if (fanRef.current) fanRef.current.rotation.z += delta * (2.4 * (1 - t * 0.7));
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.18;

    const roughness = 0.2 + t * 0.65;
    const metalness = 0.85 - t * 0.45;
    if (bodyMat.current) {
      bodyMat.current.roughness = roughness;
      bodyMat.current.metalness = metalness;
      if (currentRul > 30) {
        bodyMat.current.color.set("#b8c2d6");
        bodyMat.current.emissive.set("#0f172a");
        bodyMat.current.emissiveIntensity = 0.08;
      } else if (currentRul >= 10) {
        bodyMat.current.color.set("#d6a35a");
        bodyMat.current.emissive.set("#b45309");
        bodyMat.current.emissiveIntensity = 0.25 + (30 - currentRul) / 20 * 0.35;
      } else {
        bodyMat.current.color.set("#9f3a3a");
        bodyMat.current.emissive.set("#dc2626");
        bodyMat.current.emissiveIntensity = 0.5 + (10 - currentRul) / 10 * 0.5;
      }
    }
    if (coreMat.current) {
      coreMat.current.emissiveIntensity = 0.2 + t * 0.8;
      if (currentRul > 30) coreMat.current.emissive.set("#0e7490");
      else if (currentRul >= 10) coreMat.current.emissive.set("#b45309");
      else coreMat.current.emissive.set("#dc2626");
    }
    if (nozzleMat.current) {
      nozzleMat.current.color.set(t > 0.85 ? "#7f1d1d" : "#334155");
    }
  });

  const status = rul > 30 ? "Normal" : rul >= 10 ? "Warning" : "Critical";
  const statusColor = status === "Normal" ? "#22c55e" : status === "Warning" ? "#f59e0b" : "#ef4444";

  return (
    <>
      <group ref={groupRef}>
        <mesh material={bodyMat.current ?? new THREE.MeshStandardMaterial({ color: "#b8c2d6" })}>
          <cylinderGeometry args={[1.7, 1.5, 4.6, 64, 1, true]} />
        </mesh>
        <mesh position={[0, 0, 1.9]}>
          <group ref={fanRef}>
            <mesh material={new THREE.MeshStandardMaterial({ color: "#94a3b8", metalness: 0.8, roughness: 0.25 })}>
              <cylinderGeometry args={[0.55, 0.55, 0.18, 32]} />
            </mesh>
            {Array.from({ length: 18 }).map((_, i) => {
              const a = (i / 18) * Math.PI * 2;
              return (
                <mesh key={i} material={bodyMat.current ?? new THREE.MeshStandardMaterial({ color: "#b8c2d6" })} position={[Math.cos(a) * 1.05, Math.sin(a) * 1.05, 0]} rotation={[0.3, 0, a]}>
                  <boxGeometry args={[1.25, 0.05, 0.28]} />
                </mesh>
              );
            })}
          </group>
        </mesh>
        <mesh material={coreMat.current ?? new THREE.MeshStandardMaterial({ color: "#1e293b", emissive: "#0e7490", emissiveIntensity: 0.2 })} position={[0, 0, 0]}>
          <cylinderGeometry args={[0.75, 0.75, 4.2, 48]} />
        </mesh>
        <mesh material={nozzleMat.current ?? new THREE.MeshStandardMaterial({ color: "#334155" })} position={[0, 0, -2.6]}>
          <cylinderGeometry args={[1.15, 0.85, 1.2, 48]} />
        </mesh>
      </group>
      <Html position={[0, 2.6, 0]} center>
        <div style={{ textAlign: "center", color: "#e6edf7", fontWeight: 800, fontSize: 22 }}>
          RUL {rul.toFixed(0)}
        </div>
        <div style={{ textAlign: "center", color: statusColor, fontWeight: 700, fontSize: 13, marginTop: 2 }}>
          {status}
        </div>
      </Html>
    </>
  );
}

export function DegradationStory() {
  const reducedMotion = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [playing, setPlaying] = useState(!reducedMotion);
  const [videoFailed, setVideoFailed] = useState(false);

  useEffect(() => {
    if (reducedMotion) setPlaying(false);
  }, [reducedMotion]);

  if (!videoFailed) {
    return (
      <div style={{ position: "relative" }}>
        <video
          src="/media/about.mp4"
          autoPlay
          muted
          loop
          playsInline
          controls
          onError={() => setVideoFailed(true)}
          style={{ width: "100%", borderRadius: 16, border: "1px solid var(--border)", display: videoFailed ? "none" : "block" }}
        />
        {!videoFailed && (
          <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 6 }}>
            Optional MP4 slot: place a file at <code>public/media/about.mp4</code> to use it instead of the 3D story.
          </p>
        )}
        {videoFailed && (
          <div className="canvas-wrap" style={{ height: 420 }}>
            <Canvas camera={{ position: [4.8, 1.8, 4.8], fov: 44 }} gl={{ antialias: true, alpha: true }}>
              <color attach="background" args={["#050a14"]} />
              <ambientLight intensity={0.45} />
              <directionalLight position={[6, 6, 6]} intensity={1.1} />
              <directionalLight position={[-4, -2, -4]} intensity={0.45} color="#22d3ee" />
              <DegradationEngine playing={playing} reducedMotion={reducedMotion} />
              <ContactShadows position={[0, -2.2, 0]} opacity={0.4} scale={8} blur={2.5} />
              <OrbitControls enablePan={false} enableZoom={false} />
              <Environment preset="city" />
            </Canvas>
            <div style={{ position: "absolute", bottom: 12, left: 12, display: "flex", gap: 8 }}>
              <button type="button" className="btn btn-ghost" onClick={() => setPlaying((p) => !p)}>
                {playing ? "Pause" : "Play"}
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <div className="canvas-wrap" style={{ height: 420 }}>
        <Canvas camera={{ position: [4.8, 1.8, 4.8], fov: 44 }} gl={{ antialias: true, alpha: true }}>
          <color attach="background" args={["#050a14"]} />
          <ambientLight intensity={0.45} />
          <directionalLight position={[6, 6, 6]} intensity={1.1} />
          <directionalLight position={[-4, -2, -4]} intensity={0.45} color="#22d3ee" />
          <DegradationEngine playing={playing} reducedMotion={reducedMotion} />
          <ContactShadows position={[0, -2.2, 0]} opacity={0.4} scale={8} blur={2.5} />
          <OrbitControls enablePan={false} enableZoom={false} />
          <Environment preset="city" />
        </Canvas>
        <div style={{ position: "absolute", bottom: 12, left: 12, display: "flex", gap: 8 }}>
          <button type="button" className="btn btn-ghost" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 6 }}>
        Optional MP4 slot: place a file at <code>public/media/about.mp4</code> to use it instead of the 3D story.
      </p>
    </div>
  );
}

