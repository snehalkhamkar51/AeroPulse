import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import { ErrorBoundary } from "@/components/ErrorBoundary";

const XRAY_VERTEX = `
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vWorldPos;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vec4 worldPosition = modelMatrix * vec4(position, 1.0);
  vViewDir = normalize(cameraPosition - worldPosition.xyz);
  vWorldPos = worldPosition.xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const XRAY_FRAGMENT = `
uniform vec3 colorA;
uniform vec3 colorB;
uniform float intensity;
uniform float baseAlpha;
uniform float uOpacity;
uniform float uFresnelPower;
uniform float scanPos;
uniform float scanWidth;
uniform float pulse;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vWorldPos;
void main() {
  float fresnel = pow(1.0 - max(dot(normalize(vNormal), normalize(vViewDir)), 0.0), uFresnelPower);
  vec3 color = mix(colorA, colorB, fresnel);
  float scan = smoothstep(scanPos - scanWidth, scanPos, vWorldPos.z) * (1.0 - smoothstep(scanPos, scanPos + scanWidth, vWorldPos.z));
  color += vec3(0.25, 0.55, 0.95) * scan * 0.55;
  float alpha = uOpacity + fresnel * intensity + scan * 0.08;
  alpha = clamp(alpha, 0.0, 0.55);
  gl_FragColor = vec4(color * (0.9 + pulse * 0.1), alpha);
}
`;

function createXRayMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      colorA: { value: new THREE.Color("#0a3cff") },
      colorB: { value: new THREE.Color("#3de0ff") },
      intensity: { value: 0.42 },
      uOpacity: { value: 0.055 },
      uFresnelPower: { value: 3.0 },
      scanPos: { value: -4.0 },
      scanWidth: { value: 0.42 },
      pulse: { value: 1.0 },
    },
    vertexShader: XRAY_VERTEX,
    fragmentShader: XRAY_FRAGMENT,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
}

const XRAY_MATERIAL_REF = { current: createXRayMaterial() };

const EDGE_MATERIAL = new THREE.LineBasicMaterial({
  color: "#7dd3fc",
  transparent: true,
  opacity: 0.32,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

interface XRayMeshProps {
  geometry: THREE.BufferGeometry;
  position?: [number, number, number];
  rotation?: [number, number, number];
  scale?: [number, number, number] | number;
  edges?: boolean;
  xray?: boolean;
}

const SOLID_MATERIAL = new THREE.MeshStandardMaterial({
  color: "#1e6bff",
  metalness: 0.72,
  roughness: 0.32,
  emissive: "#0b2f6b",
  emissiveIntensity: 0.18,
});

function XRayMesh({ geometry, position = [0, 0, 0], rotation = [0, 0, 0], scale = 1, edges = true, xray = true }: XRayMeshProps) {
  const edgeGeometry = useMemo(() => new THREE.EdgesGeometry(geometry, 30), [geometry]);
  return (
    <group position={position} rotation={rotation} scale={scale}>
      <mesh geometry={geometry} material={xray ? XRAY_MATERIAL_REF.current : SOLID_MATERIAL} />
      {edges && <lineSegments geometry={edgeGeometry} material={EDGE_MATERIAL} />}
    </group>
  );
}

interface EngineProps {
  autoRotate?: boolean;
  reducedMotion?: boolean;
  paused?: boolean;
  cutaway?: boolean;
  exploded?: boolean;
  showLabels?: boolean;
  lowPerf?: boolean;
  onLowPerf?: (value: boolean) => void;
  bodyOpacity?: number;
  fresnelPower?: number;
  edgeOpacity?: number;
  bloomIntensity?: number;
  bloomThreshold?: number;
  bloomEnabled?: boolean;
  mode?: string;
  playing?: boolean;
  onProgress?: (value: number) => void;
  onPhase?: (value: string) => void;
}

function HolographicEngine({ autoRotate = true, reducedMotion = false, paused = false, cutaway = true, exploded = false, showLabels = true, lowPerf = false, onLowPerf, bodyOpacity = 0.055, fresnelPower = 3.0, edgeOpacity = 0.32 }: EngineProps) {
  const rootRef = useRef<THREE.Group>(null);
  const fanRef = useRef<THREE.Group>(null);
  const stageRefs = useRef<THREE.Group[]>([]);
  const pointsRef = useRef<THREE.Points>(null);
  const materialRef = XRAY_MATERIAL_REF;
  const explodeRef = useRef(0);
  const introRef = useRef(0);
  const frameTimes = useRef<number[]>([]);
  const { pointer } = useThree();

  const particleCount = lowPerf ? 500 : 1500;
  const particles = useMemo(() => {
    const arr = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const r = Math.random() * 0.35;
      const a = Math.random() * Math.PI * 2;
      arr[i * 3] = Math.cos(a) * r;
      arr[i * 3 + 1] = Math.sin(a) * r;
      arr[i * 3 + 2] = -2.8 + Math.random() * 0.4;
    }
    return arr;
  }, [particleCount]);

  const fanBladeGeo = useMemo(() => new THREE.BoxGeometry(1.25, 0.025, 0.22), []);
  const fanBladeEdges = useMemo(() => new THREE.EdgesGeometry(fanBladeGeo, 20), [fanBladeGeo]);
  const fanCount = 24;
  const fanMatrices = useMemo(() => {
    const matrices: THREE.Matrix4[] = [];
    const dummy = new THREE.Object3D();
    for (let i = 0; i < fanCount; i++) {
      const a = (i / fanCount) * Math.PI * 2;
      dummy.position.set(Math.cos(a) * 1.05, Math.sin(a) * 1.05, 0);
      dummy.rotation.set(0.45, 0, a);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      matrices.push(dummy.matrix.clone());
    }
    return matrices;
  }, []);

  const nacelleGeo = useMemo(() => {
    const points: THREE.Vector2[] = [];
    for (let i = 0; i <= 24; i++) {
      const z = -2.7 + (i / 24) * 5.4;
      const r = 1.85 + Math.sin((i / 24) * Math.PI) * 0.22 + (z > 2.3 ? (z - 2.3) * 0.05 : 0);
      points.push(new THREE.Vector2(Math.max(0.4, r), z));
    }
    return new THREE.LatheGeometry(points, 96);
  }, []);

  const coreGeo = useMemo(() => new THREE.CylinderGeometry(0.72, 0.72, 4.8, 64), []);
  const nozzleGeo = useMemo(() => new THREE.CylinderGeometry(1.18, 0.82, 1.35, 64), []);
  const spinnerGeo = useMemo(() => new THREE.ConeGeometry(0.48, 1.0, 48), []);
  const stageDiskGeo = useMemo(() => new THREE.CylinderGeometry(0.95, 0.95, 0.12, 48), []);
  const bladeGeo = useMemo(() => new THREE.BoxGeometry(0.5, 0.025, 0.14), []);
  const bladeEdges = useMemo(() => new THREE.EdgesGeometry(bladeGeo, 20), [bladeGeo]);

  const stageCount = 7;
  const stageZ = useMemo(() => Array.from({ length: stageCount }, (_, i) => 1.0 - i * 0.62), []);
  const stageRadii = useMemo(() => Array.from({ length: stageCount }, (_, i) => 0.95 - i * 0.045), []);
  const stageBladeCounts = useMemo(() => Array.from({ length: stageCount }, (_, i) => (i < 4 ? 12 : 10)), []);

  const stageMatrices = useMemo(() => {
    return stageBladeCounts.map((count, stage) => {
      const matrices: THREE.Matrix4[] = [];
      const dummy = new THREE.Object3D();
      const radius = stageRadii[stage];
      for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2;
        dummy.position.set(Math.cos(a) * radius, Math.sin(a) * radius, 0);
        dummy.rotation.set(0.28, 0, a);
        dummy.updateMatrix();
        matrices.push(dummy.matrix.clone());
      }
      return matrices;
    });
  }, [stageBladeCounts, stageRadii]);

  const turbineCount = 3;
  const turbineZ = useMemo(() => Array.from({ length: turbineCount }, (_, i) => -2.35 - i * 0.62), []);
  const turbineRadii = [1.08, 1.0, 0.92];
  const turbineBladeCounts = [18, 14, 12];
  const turbineMatrices = useMemo(() => {
    return turbineBladeCounts.map((count, stage) => {
      const matrices: THREE.Matrix4[] = [];
      const dummy = new THREE.Object3D();
      const radius = turbineRadii[stage];
      for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2;
        dummy.position.set(Math.cos(a) * radius, Math.sin(a) * radius, 0);
        dummy.rotation.set(-0.22, 0, a);
        dummy.updateMatrix();
        matrices.push(dummy.matrix.clone());
      }
      return matrices;
    });
  }, []);

  const pipeCurve = useMemo(() => {
    const pts = [
      new THREE.Vector3(0.85, 0, 2.0),
      new THREE.Vector3(1.05, 0.35, 0.8),
      new THREE.Vector3(0.95, -0.2, -0.6),
      new THREE.Vector3(0.8, 0.15, -2.0),
    ];
    return new THREE.CatmullRomCurve3(pts);
  }, []);
  const pipeGeo = useMemo(() => new THREE.TubeGeometry(pipeCurve, 48, 0.035, 8, false), [pipeCurve]);

  const rimTicks = useMemo(() => Array.from({ length: 72 }), []);


  useFrame((state, delta) => {
    const dt = Math.min(delta, 0.05);
    const t = state.clock.getElapsedTime();
    const reduced = reducedMotion;

    introRef.current = Math.min(1, introRef.current + (reduced ? 0 : dt / 2.0));

    if (rootRef.current) {
      const intro = introRef.current;
      rootRef.current.scale.setScalar(0.72 + intro * 0.28);
      if (!reduced) {
        rootRef.current.position.y = Math.sin(t * 0.9) * 0.05;
        rootRef.current.rotation.x = 0.14 + pointer.y * 0.07;
        rootRef.current.rotation.y = -0.32 + pointer.x * 0.08 + (autoRotate && !paused ? t * 0.06 : 0);
      }
    }

    if (!reduced && !paused) {
      const explodeTarget = exploded ? 1 : 0;
      explodeRef.current += (explodeTarget - explodeRef.current) * Math.min(1, dt * 4.5);
      const ex = explodeRef.current;

      if (fanRef.current) fanRef.current.rotation.z += dt * (autoRotate ? 2.6 : 0);
      stageRefs.current.forEach((g, i) => {
        if (!g) return;
        g.rotation.z += dt * (autoRotate ? 1.5 - i * 0.12 : 0) * (i % 2 === 0 ? 1 : -0.7);
        const baseZ = 1.0 - i * 0.62;
        g.position.z = baseZ + ex * (i < 4 ? -0.55 * (i + 1) : 0.45 * (i - 3));
      });

      const mat = materialRef.current;
      mat.uniforms.uOpacity.value = bodyOpacity;
      mat.uniforms.uFresnelPower.value = fresnelPower;
      mat.uniforms.scanPos.value = -4.2 + ((t % 6) / 6) * 8.4;
      mat.uniforms.pulse.value = 1.0 + Math.sin(t * 1.35) * 0.35;
      EDGE_MATERIAL.opacity = edgeOpacity;

      if (pointsRef.current) {
        const attr = pointsRef.current.geometry.attributes.position as THREE.BufferAttribute;
        const arr = attr.array as Float32Array;
        for (let i = 0; i < particleCount; i++) {
          const idx = i * 3;
          arr[idx + 2] += (0.025 + (i % 7) * 0.004) * dt * 60;
          if (arr[idx + 2] > 4.6) {
            const r = Math.random() * 0.3;
            const a = Math.random() * Math.PI * 2;
            arr[idx] = Math.cos(a) * r;
            arr[idx + 1] = Math.sin(a) * r;
            arr[idx + 2] = -2.9;
          }
        }
        attr.needsUpdate = true;
      }
    }

    const now = performance.now();
    frameTimes.current.push(now);
    if (frameTimes.current.length > 45) frameTimes.current.shift();
    if (frameTimes.current.length === 45 && onLowPerf) {
      const elapsed = now - frameTimes.current[0];
      const fps = (frameTimes.current.length - 1) / (elapsed / 1000);
      if (fps < 40) onLowPerf(true);
    }
  });

  const fanInstancedRef = useRef<THREE.InstancedMesh>(null);
  const fanEdgeInstancedRef = useRef<THREE.InstancedMesh>(null);
  useEffect(() => {
    if (fanInstancedRef.current) {
      fanMatrices.forEach((m, i) => fanInstancedRef.current!.setMatrixAt(i, m));
      fanInstancedRef.current.instanceMatrix.needsUpdate = true;
    }
    if (fanEdgeInstancedRef.current) {
      fanMatrices.forEach((m, i) => fanEdgeInstancedRef.current!.setMatrixAt(i, m));
      fanEdgeInstancedRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [fanMatrices]);


  return (
    <group ref={rootRef}>
      <XRayMesh xray={cutaway} geometry={nacelleGeo} />
      {rimTicks.map((_, i) => {
        const a = (i / rimTicks.length) * Math.PI * 2;
        return (
          <XRayMesh
            key={`tick-${i}`}
            geometry={new THREE.BoxGeometry(0.025, 0.12, 0.04)}
            position={[Math.cos(a) * 1.92, Math.sin(a) * 1.92, 2.55]}
            rotation={[0, 0, a]}
            edges={false}
          />
        );
      })}

      <group ref={fanRef} position={[0, 0, 1.9]}>
        <XRayMesh xray={cutaway} geometry={new THREE.CylinderGeometry(0.58, 0.58, 0.2, 48)} rotation={[Math.PI / 2, 0, 0]} />
        <instancedMesh ref={fanInstancedRef} args={[fanBladeGeo, XRAY_MATERIAL_REF.current, fanCount]} />
        <instancedMesh ref={fanEdgeInstancedRef} args={[fanBladeEdges, EDGE_MATERIAL, fanCount]} />
        <XRayMesh xray={cutaway} geometry={spinnerGeo} position={[0, 0, 0.55]} rotation={[Math.PI / 2, 0, 0]} />
      </group>

      <group position={[0, 0, 1.15]}>
        {Array.from({ length: 14 }).map((_, i) => {
          const a = (i / 14) * Math.PI * 2;
          return (
            <XRayMesh
              key={`stator-${i}`}
              geometry={new THREE.BoxGeometry(0.03, 0.04, 0.7)}
              position={[Math.cos(a) * 1.25, Math.sin(a) * 1.25, 0]}
              rotation={[0, 0, a]}
              edges={false}
            />
          );
        })}
      </group>

      <XRayMesh xray={cutaway} geometry={new THREE.CylinderGeometry(1.48, 1.48, 4.9, 64, 1, true)} rotation={[Math.PI / 2, 0, 0]} edges={false} />
      <XRayMesh xray={cutaway} geometry={coreGeo} rotation={[Math.PI / 2, 0, 0]} />

      {stageZ.map((z, stage) => (
        <group key={`stage-${stage}`} ref={(el) => { if (el) stageRefs.current[stage] = el; }} position={[0, 0, z]}>
          <XRayMesh xray={cutaway} geometry={stageDiskGeo} rotation={[Math.PI / 2, 0, 0]} />
          <instancedMesh args={[bladeGeo, XRAY_MATERIAL_REF.current, stageBladeCounts[stage]]}>
            {stageMatrices[stage].map((m, i) => (
              <primitive key={i} object={new THREE.Mesh(bladeGeo, XRAY_MATERIAL_REF.current)} matrix={m} matrixAutoUpdate={false} />
            ))}
          </instancedMesh>
          <instancedMesh args={[bladeEdges, EDGE_MATERIAL, stageBladeCounts[stage]]}>
            {stageMatrices[stage].map((m, i) => (
              <primitive key={i} object={new THREE.LineSegments(bladeEdges, EDGE_MATERIAL)} matrix={m} matrixAutoUpdate={false} />
            ))}
          </instancedMesh>
        </group>
      ))}

      <XRayMesh xray={cutaway} geometry={new THREE.CylinderGeometry(0.92, 0.92, 1.25, 48)} position={[0, 0, -1.55]} rotation={[Math.PI / 2, 0, 0]} />

      {turbineZ.map((z, stage) => (
        <group key={`turb-${stage}`} ref={(el) => { if (el) stageRefs.current[stage + stageCount] = el; }} position={[0, 0, z]}>
          <XRayMesh xray={cutaway} geometry={new THREE.CylinderGeometry(turbineRadii[stage], turbineRadii[stage], 0.14, 48)} rotation={[Math.PI / 2, 0, 0]} />
          <instancedMesh args={[bladeGeo, XRAY_MATERIAL_REF.current, turbineBladeCounts[stage]]}>
            {turbineMatrices[stage].map((m, i) => (
              <primitive key={i} object={new THREE.Mesh(bladeGeo, XRAY_MATERIAL_REF.current)} matrix={m} matrixAutoUpdate={false} />
            ))}
          </instancedMesh>
        </group>
      ))}

      <XRayMesh xray={cutaway} geometry={nozzleGeo} position={[0, 0, -3.35]} rotation={[Math.PI / 2, 0, 0]} />
      <XRayMesh xray={cutaway} geometry={new THREE.ConeGeometry(0.72, 1.15, 48)} position={[0, 0, -4.15]} rotation={[Math.PI / 2, 0, 0]} />

      <XRayMesh xray={cutaway} geometry={pipeGeo} edges={false} />

      <XRayMesh xray={cutaway} geometry={new THREE.BoxGeometry(0.35, 0.22, 0.9)} position={[0, -1.05, 0.2]} />

      {Array.from({ length: 4 }).map((_, i) => (
        <XRayMesh
          key={`bracket-${i}`}
          geometry={new THREE.TorusGeometry(0.78, 0.025, 8, 48)}
          position={[0, 0, -0.4 + i * 0.9]}
          rotation={[Math.PI / 2, 0, 0]}
          edges={false}
        />
      ))}

      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={particleCount} array={particles} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial size={0.028} color="#7dd3fc" transparent opacity={0.72} blending={THREE.AdditiveBlending} sizeAttenuation depthWrite={false} />
      </points>

      {showLabels && (
        <>
          <Html position={[0, 1.85, 2.45]} center>
            <div className="engine-label">Fan</div>
          </Html>
          <Html position={[1.95, 0.55, 0.15]} center>
            <div className="engine-label">Compressor</div>
          </Html>
          <Html position={[-1.95, 0.55, -1.55]} center>
            <div className="engine-label">Combustor</div>
          </Html>
          <Html position={[0, 1.55, -2.65]} center>
            <div className="engine-label">Turbine</div>
          </Html>
          <Html position={[0, 1.25, -4.0]} center>
            <div className="engine-label">Nozzle</div>
          </Html>
        </>
      )}
    </group>
  );
}


function EngineCanvasInner({ reducedMotion, lowPerf, onLowPerf, mode, playing, onProgress, onPhase, ...props }: EngineProps & { reducedMotion: boolean; mode: string; playing: boolean; onProgress: (value: number) => void; onPhase: (value: string) => void; }) {
  return (
    <Canvas
      camera={{ position: [-4, 2, 7], fov: 35 }}
      gl={{ antialias: true, alpha: true, toneMapping: THREE.NoToneMapping }}
      dpr={[1, 2]}
      frameloop="always"
    >
      <color attach="background" args={["#050a14"]} />
      <fog attach="fog" args={["#050a14", 8, 18]} />
      <ambientLight intensity={0.35} />
      <directionalLight position={[6, 6, 6]} intensity={1.0} />
      <directionalLight position={[-5, -2, -5]} intensity={0.5} color="#3de0ff" />
      <AirplaneJourneyScene {...props} reducedMotion={reducedMotion} lowPerf={lowPerf} onLowPerf={onLowPerf} mode={mode ?? "journey"} playing={playing ?? false} onProgress={onProgress ?? (() => {})} onPhase={onPhase ?? (() => {})} />
      <OrbitControls enablePan={false} enableZoom={false} enableDamping dampingFactor={0.08} autoRotate={!reducedMotion && !props.paused} autoRotateSpeed={0.45} target={[0, 0, 0]} />
      {!lowPerf && props.bloomEnabled !== false && (
        <EffectComposer>
          <Bloom intensity={props.bloomIntensity ?? 0.5} luminanceThreshold={props.bloomThreshold ?? 0.7} luminanceSmoothing={0.2} mipmapBlur radius={0.4} />
          <Vignette eskil={false} offset={0.28} darkness={0.72} />
        </EffectComposer>
      )}
    </Canvas>
  );
}

export function EngineCanvas({ height = 420, autoRotate = true }: { height?: number; autoRotate?: boolean }) {
  const reducedMotion = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [paused, setPaused] = useState(false);
  const [cutaway, setCutaway] = useState(true);
  const [exploded, setExploded] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [lowPerf, setLowPerf] = useState(false);
  const [visible, setVisible] = useState(true);
  const [mode, setMode] = useState(reducedMotion ? "engine" : "journey");
  const [playing, setPlaying] = useState(!reducedMotion);
  const [progress, setProgress] = useState(reducedMotion ? 1 : 0);
  const [phase, setPhase] = useState(reducedMotion ? "C" : "A");
  const debug = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("debug") === "1";
  const [bodyOpacity, setBodyOpacity] = useState(0.055);
  const [fresnelPower, setFresnelPower] = useState(3.0);
  const [edgeOpacity, setEdgeOpacity] = useState(0.32);
  const [bloomIntensity, setBloomIntensity] = useState(0.5);
  const [bloomThreshold, setBloomThreshold] = useState(0.7);
  const [bloomEnabled, setBloomEnabled] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { threshold: 0.1 });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const hasWebGL = typeof window !== "undefined" && !!document.createElement("canvas").getContext("webgl");

  return (
    <div ref={containerRef} className="canvas-wrap" style={{ height }} aria-label="Interactive holographic turbofan engine">
      {!hasWebGL ? (
        <div className="holographic-fallback" />
      ) : visible ? (
        <ErrorBoundary fallback={<div style={{ height: "100%", display: "grid", placeItems: "center", color: "var(--text-muted)" }}>3D engine view unavailable.</div>}>
          <EngineCanvasInner reducedMotion={reducedMotion} autoRotate={autoRotate} paused={paused} cutaway={cutaway} exploded={exploded} showLabels={showLabels} lowPerf={lowPerf} onLowPerf={setLowPerf} bodyOpacity={bodyOpacity} fresnelPower={fresnelPower} edgeOpacity={edgeOpacity} bloomIntensity={bloomIntensity} bloomThreshold={bloomThreshold} bloomEnabled={bloomEnabled} mode={mode} playing={playing} onProgress={setProgress} onPhase={setPhase} />
        </ErrorBoundary>
      ) : (
        <div style={{ height: "100%", display: "grid", placeItems: "center", color: "var(--text-dim)" }}>Engine paused off-screen</div>
      )}
      {debug && (
        <div style={{ position: "absolute", top: 12, right: 12, background: "rgba(5,10,20,0.85)", border: "1px solid var(--border)", borderRadius: 10, padding: 10, fontSize: 12, display: "grid", gap: 6, minWidth: 220, zIndex: 5 }}>
          <label>body opacity <input type="range" min="0.02" max="0.25" step="0.005" value={bodyOpacity} onChange={(e) => setBodyOpacity(Number(e.target.value))} /> {bodyOpacity.toFixed(3)}</label>
          <label>fresnel power <input type="range" min="1.5" max="5" step="0.1" value={fresnelPower} onChange={(e) => setFresnelPower(Number(e.target.value))} /> {fresnelPower.toFixed(1)}</label>
          <label>edge opacity <input type="range" min="0" max="0.8" step="0.01" value={edgeOpacity} onChange={(e) => setEdgeOpacity(Number(e.target.value))} /> {edgeOpacity.toFixed(2)}</label>
          <label>bloom intensity <input type="range" min="0" max="1.5" step="0.05" value={bloomIntensity} onChange={(e) => setBloomIntensity(Number(e.target.value))} /> {bloomIntensity.toFixed(2)}</label>
          <label>bloom threshold <input type="range" min="0.1" max="0.9" step="0.05" value={bloomThreshold} onChange={(e) => setBloomThreshold(Number(e.target.value))} /> {bloomThreshold.toFixed(2)}</label>
          <label><input type="checkbox" checked={bloomEnabled} onChange={(e) => setBloomEnabled(e.target.checked)} /> bloom enabled</label>
        </div>
      )}
      {phase === "A" && (
        <div className="story-overlay">
          <h2>AeroPulse — See Beyond the Surface</h2>
          <p>From aircraft to engine intelligence</p>
        </div>
      )}
      {phase === "C" && progress > 0.92 && (
        <div className="story-overlay story-overlay-bottom">
          <p>Engine intelligence. Predicted before failure.</p>
        </div>
      )}
      <div className="engine-controls">
        <button type="button" onClick={() => { setMode("journey"); setProgress(0); setPlaying(true); setPaused(false); }}>Replay Journey</button>
        <button type="button" onClick={() => setPlaying((v) => !v)}>{playing ? "Pause" : "Resume"}</button>
        <button type="button" onClick={() => { setMode("engine"); setProgress(1); setPlaying(false); setPaused(false); }}>Explore Engine</button>
        <button type="button" onClick={() => { setMode("aircraft"); setPlaying(true); setPaused(false); }}>View Aircraft</button>
        <button type="button" onClick={() => setCutaway((v) => !v)}>{cutaway ? "X-Ray" : "Solid"}</button>
        <button type="button" onClick={() => setExploded((v) => !v)}>{exploded ? "Assemble" : "Explode"}</button>
        <button type="button" onClick={() => setShowLabels((v) => !v)}>{showLabels ? "Hide Labels" : "Show Labels"}</button>
      </div>
    </div>
  );
}


const AIRPLANE_MATERIAL = new THREE.MeshStandardMaterial({ color: "#9aa4b2", metalness: 0.72, roughness: 0.34 });

function Airplane({ scale = 1 }: { scale?: number }) {
  const fuselageGeo = useMemo(() => new THREE.CylinderGeometry(1.0, 1.0, 14, 48), []);
  const noseGeo = useMemo(() => new THREE.ConeGeometry(1.0, 2.2, 32), []);
  const tailGeo = useMemo(() => new THREE.ConeGeometry(0.7, 1.8, 24), []);
  const wingGeo = useMemo(() => new THREE.BoxGeometry(5.6, 0.12, 2.3), []);
  const hStabGeo = useMemo(() => new THREE.BoxGeometry(2.4, 0.08, 1.2), []);
  const vStabGeo = useMemo(() => new THREE.BoxGeometry(0.1, 2.3, 1.35), []);
  const gearGeo = useMemo(() => new THREE.CylinderGeometry(0.08, 0.08, 1.0, 12), []);
  const windowGeo = useMemo(() => new THREE.BoxGeometry(0.08, 0.08, 0.04), []);

  const windows = useMemo(() => Array.from({ length: 28 }), []);

  return (
    <group scale={scale}>
      <mesh geometry={fuselageGeo} material={AIRPLANE_MATERIAL} rotation={[Math.PI / 2, 0, 0]} />
      <mesh geometry={noseGeo} material={AIRPLANE_MATERIAL} position={[0, 0, 7.9]} rotation={[Math.PI / 2, 0, 0]} />
      <mesh geometry={tailGeo} material={AIRPLANE_MATERIAL} position={[0, 0, -7.6]} rotation={[-Math.PI / 2, 0, 0]} />
      <mesh geometry={wingGeo} material={AIRPLANE_MATERIAL} position={[2.8, -0.12, 0]} />
      <mesh geometry={wingGeo} material={AIRPLANE_MATERIAL} position={[-2.8, -0.12, 0]} />
      <mesh geometry={hStabGeo} material={AIRPLANE_MATERIAL} position={[1.2, 0.05, -6.4]} />
      <mesh geometry={hStabGeo} material={AIRPLANE_MATERIAL} position={[-1.2, 0.05, -6.4]} />
      <mesh geometry={vStabGeo} material={AIRPLANE_MATERIAL} position={[0, 1.15, -6.4]} />
      {windows.map((_, i) => {
        const z = 5.5 - i * 0.38;
        return (
          <group key={`w-${i}`}>
            <mesh geometry={windowGeo} material={new THREE.MeshStandardMaterial({ color: "#1e3a8a", metalness: 0.2, roughness: 0.4, emissive: "#0b1f4d", emissiveIntensity: 0.15 })} position={[1.02, 0, z]} />
            <mesh geometry={windowGeo} material={new THREE.MeshStandardMaterial({ color: "#1e3a8a", metalness: 0.2, roughness: 0.4, emissive: "#0b1f4d", emissiveIntensity: 0.15 })} position={[-1.02, 0, z]} />
          </group>
        );
      })}
      <mesh geometry={gearGeo} material={AIRPLANE_MATERIAL} position={[0, -1.25, 3.2]} rotation={[Math.PI / 2, 0, 0]} />
      <mesh geometry={gearGeo} material={AIRPLANE_MATERIAL} position={[2.4, -1.25, -1.2]} rotation={[Math.PI / 2, 0, 0]} />
      <mesh geometry={gearGeo} material={AIRPLANE_MATERIAL} position={[-2.4, -1.25, -1.2]} rotation={[Math.PI / 2, 0, 0]} />
    </group>
  );
}



const SELECTED_ENGINE_POS = new THREE.Vector3(3.2, -1.35, 0);

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function cameraFor(progress: number) {
  const engine = SELECTED_ENGINE_POS;
  if (progress < 0.3) {
    const u = progress / 0.3;
    const angle = u * 0.55;
    return {
      pos: new THREE.Vector3(Math.sin(angle) * 10.5, 3.4, Math.cos(angle) * 10.5),
      look: new THREE.Vector3(0, 0, 0),
    };
  }
  if (progress < 0.6) {
    const u = smoothstep(0, 1, (progress - 0.3) / 0.3);
    const start = new THREE.Vector3(Math.sin(0.55) * 10.5, 3.4, Math.cos(0.55) * 10.5);
    const end = new THREE.Vector3(engine.x + 1.0, engine.y + 0.75, engine.z + 6.2);
    const pos = start.clone().lerp(end, u);
    const look = new THREE.Vector3(0, 0, 0).lerp(engine.clone(), u);
    return { pos, look };
  }
  const u = smoothstep(0, 1, (progress - 0.6) / 0.4);
  const start = new THREE.Vector3(engine.x + 1.0, engine.y + 0.75, engine.z + 6.2);
  const mid = new THREE.Vector3(engine.x + 0.85, engine.y + 0.2, engine.z + 2.0);
  const end = new THREE.Vector3(engine.x + 2.0, engine.y + 1.0, engine.z + 1.15);
  const pos = u < 0.5 ? start.clone().lerp(mid, u * 2) : mid.clone().lerp(end, (u - 0.5) * 2);
  const look = engine.clone();
  return { pos, look };
}

function JourneyController({ mode, playing, reducedMotion, onProgress, onPhase }: { mode: string; playing: boolean; reducedMotion: boolean; onProgress: (value: number) => void; onPhase: (value: string) => void; }) {
  const { camera } = useThree();
  const progressRef = useRef(0);
  const modeRef = useRef(mode);
  useEffect(() => {
    modeRef.current = mode;
    if (mode === "engine") progressRef.current = 1;
    if (mode === "aircraft") progressRef.current = 0;
  }, [mode]);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.05);
    if (!reducedMotion) {
      if (modeRef.current === "journey" && playing) {
        progressRef.current = Math.min(1, progressRef.current + dt / 12);
      } else if (modeRef.current === "aircraft" && playing) {
        progressRef.current = Math.max(0, progressRef.current - dt / 1.5);
      } else if (modeRef.current === "engine") {
        progressRef.current = 1;
      }
    }
    const p = progressRef.current;
    const { pos, look } = cameraFor(p);
    camera.position.copy(pos);
    camera.lookAt(look);
    onProgress(p);
    const phase = p < 0.3 ? "A" : p < 0.6 ? "B" : "C";
    onPhase(phase);
  });
  return null;
}

function AirplaneJourneyScene(props: EngineProps & { mode: string; playing: boolean; reducedMotion: boolean; onProgress: (value: number) => void; onPhase: (value: string) => void; }) {
  const { mode, playing, reducedMotion, onProgress, onPhase, ...engineProps } = props;
  return (
    <>
      <Airplane scale={1} />
      <group position={[3.2, -1.35, 0]} scale={0.42}>
        <HolographicEngine {...engineProps} showLabels={false} reducedMotion={reducedMotion} autoRotate={false} />
      </group>
      <group position={[-3.2, -1.35, 0]} scale={0.42}>
        <HolographicEngine {...engineProps} showLabels={false} reducedMotion={reducedMotion} autoRotate={false} />
      </group>
      <group position={[3.2, -1.35, 0]}>
        <HolographicEngine {...engineProps} reducedMotion={reducedMotion} />
      </group>
      <JourneyController mode={mode} playing={playing} reducedMotion={reducedMotion} onProgress={onProgress} onPhase={onPhase} />
    </>
  );
}

