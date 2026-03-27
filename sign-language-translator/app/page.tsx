"use client";

import { DrawingUtils, FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

const API_URL = "http://localhost:8000/predict";
const FRAME_INTERVAL_MS = 1000 / 30;
const STABILITY_TARGET = 15;
const NOTHING = "NOTHING";

type PredictResponse = {
  prediction?: string;
  letter?: string;
};

const normalizePrediction = (value: string | undefined) =>
  (value ?? NOTHING).trim().toUpperCase();

export default function Home() {
  const webcamRef = useRef<Webcam | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const inFlightRef = useRef(false);
  const mountedRef = useRef(false);
  const lastPredictionRef = useRef<string>(NOTHING);
  const stabilityCountRef = useRef(0);
  const committedLetterRef = useRef<string | null>(null);

  const [sentence, setSentence] = useState("");
  const [candidate, setCandidate] = useState(NOTHING);
  const [stabilityCount, setStabilityCount] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clearSentence = () => {
    setSentence("");
  };

  const backspaceSentence = () => {
    setSentence((prev) => prev.slice(0, -1));
  };

  const appendPrediction = useCallback((prediction: string) => {
    if (prediction === "SPACE") {
      setSentence((prev) => `${prev} `);
      return;
    }

    if (prediction === "DEL") {
      setSentence((prev) => prev.slice(0, -1));
      return;
    }

    setSentence((prev) => `${prev}${prediction}`);
  }, []);

  const handlePrediction = useCallback(
    (prediction: string) => {
      if (prediction === NOTHING) {
        lastPredictionRef.current = NOTHING;
        stabilityCountRef.current = 0;
        committedLetterRef.current = null;
        setCandidate(NOTHING);
        setStabilityCount(0);
        return;
      }

      if (prediction !== lastPredictionRef.current) {
        lastPredictionRef.current = prediction;
        stabilityCountRef.current = 1;
      } else {
        stabilityCountRef.current += 1;
      }

      setCandidate(prediction);
      setStabilityCount(stabilityCountRef.current);

      if (
        stabilityCountRef.current >= STABILITY_TARGET &&
        committedLetterRef.current !== prediction
      ) {
        appendPrediction(prediction);
        committedLetterRef.current = prediction;
      }
    },
    [appendPrediction]
  );

  const drawLandmarkOverlay = useCallback(
    (
      landmarks: any,
      videoWidth: number,
      videoHeight: number
    ) => {
      const canvas = overlayRef.current;
      if (!canvas) {
        return;
      }

      if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!landmarks || landmarks.length === 0) {
        return;
      }

      const drawingUtils = new DrawingUtils(ctx);
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: "#22d3ee",
        lineWidth: 3,
      });
      drawingUtils.drawLandmarks(landmarks, {
        color: "#f97316",
        radius: 4,
      });

      const wrist = landmarks[0];
      if (wrist) {
        ctx.fillStyle = "rgba(5, 13, 24, 0.78)";
        ctx.fillRect(12, 12, 260, 72);
        ctx.strokeStyle = "rgba(34, 211, 238, 0.7)";
        ctx.strokeRect(12, 12, 260, 72);
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "bold 14px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(`Wrist x: ${wrist.x.toFixed(3)}`, 20, 36);
        ctx.fillText(`Wrist y: ${wrist.y.toFixed(3)}`, 20, 56);
        ctx.fillText(`Wrist z: ${wrist.z.toFixed(3)}`, 20, 76);
      }

      ctx.fillStyle = "#e2e8f0";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      landmarks.forEach((point: any, index: number) => {
        const px = point.x * canvas.width + 8;
        const py = point.y * canvas.height - 8;
        ctx.fillText(`${index}: ${point.x.toFixed(2)}, ${point.y.toFixed(2)}`, px, py);
      });
    },
    []
  );

  useEffect(() => {
    mountedRef.current = true;

    const loadHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        detectorRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          numHands: 1,
          runningMode: "VIDEO",
        });

        if (mountedRef.current) {
          setIsReady(true);
          setError(null);
        }
      } catch {
        if (mountedRef.current) {
          setError("Unable to initialize MediaPipe hand tracker.");
        }
      }
    };

    loadHandLandmarker();

    return () => {
      mountedRef.current = false;
      detectorRef.current?.close();
      detectorRef.current = null;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const tick = async (timestamp: number) => {
      if (!mountedRef.current) {
        return;
      }

      rafRef.current = requestAnimationFrame(tick);

      if (timestamp - lastFrameTimeRef.current < FRAME_INTERVAL_MS) {
        return;
      }
      lastFrameTimeRef.current = timestamp;

      const detector = detectorRef.current;
      const video = webcamRef.current?.video;

      if (!detector || !video || video.readyState < 2) {
        return;
      }

      const result = detector.detectForVideo(video, performance.now());
      const hand = result.landmarks?.[0];

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        drawLandmarkOverlay(hand, video.videoWidth, video.videoHeight);
      }

      if (!hand || hand.length !== 21) {
        handlePrediction(NOTHING);
        return;
      }

      if (inFlightRef.current) {
        return;
      }

      const landmarks = hand.flatMap((point) => [point.x, point.y, point.z]);
      if (landmarks.length !== 63) {
        return;
      }

      inFlightRef.current = true;

      try {
        const response = await fetch(API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(landmarks),
        });

        if (!response.ok) {
          throw new Error("Prediction request failed");
        }

        const data = (await response.json()) as PredictResponse;
        const prediction = normalizePrediction(data.prediction ?? data.letter);

        if (mountedRef.current) {
          handlePrediction(prediction);
          setError(null);
        }
      } catch {
        if (mountedRef.current) {
          setError("FastAPI endpoint is unreachable at http://localhost:8000/predict");
        }
      } finally {
        inFlightRef.current = false;
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [drawLandmarkOverlay, handlePrediction]);

  const progress = Math.min((stabilityCount / STABILITY_TARGET) * 100, 100);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0a0a0f] text-slate-50 font-sans">
      {/* Dynamic Background Gradients (Optimized for GPU Performance) */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_10%,rgba(79,70,229,0.15),transparent_40%),radial-gradient(circle_at_80%_20%,rgba(217,70,239,0.15),transparent_40%),radial-gradient(circle_at_40%_80%,rgba(8,145,178,0.12),transparent_50%),linear-gradient(160deg,#0a0a0f_0%,#0a0a0f_100%)]" />

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-8 md:px-8">
        {/* Header */}
        <header className="flex flex-col gap-4 rounded-[2rem] border border-white/10 bg-white/5 px-8 py-6 shadow-2xl shadow-black/50 backdrop-blur-lg lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-5">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-lg shadow-indigo-500/25">
              <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-indigo-300/80">Premium Translation Engine</p>
              <h1 className="mt-1 text-3xl font-bold tracking-tight text-white md:text-4xl bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                ASL Vision
              </h1>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <div className={`flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm font-medium backdrop-blur-md ${isReady ? 'bg-emerald-500/10 text-emerald-300' : 'bg-amber-500/10 text-amber-300'}`}>
              <span className={`h-2 w-2 rounded-full ${isReady ? 'animate-pulse bg-emerald-400' : 'bg-amber-400'}`} />
              {isReady ? "System Active" : "Initializing..."}
            </div>
            <div className="flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-4 py-2 text-sm font-medium text-indigo-300 backdrop-blur-md">
              <span className="h-2 w-2 rounded-full bg-indigo-400" />
              60 FPS Neural Link
            </div>
          </div>
        </header>

        {/* Main Interface Grid */}
        <section className="grid flex-1 gap-6 lg:grid-cols-[1fr_360px]">
          {/* Camera Feed */}
          <div className="group relative flex min-h-[480px] flex-col overflow-hidden rounded-[2.5rem] border border-white/10 bg-black/40 shadow-2xl backdrop-blur-xl transition-all duration-500 hover:border-white/20 hover:shadow-indigo-500/10">
            <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent p-6">
              <div className="flex items-center gap-3 rounded-2xl bg-black/40 px-4 py-2 backdrop-blur-md border border-white/5">
                <span className="relative flex h-3 w-3">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex h-3 w-3 rounded-full bg-rose-500"></span>
                </span>
                <span className="text-sm font-medium tracking-wide text-slate-200">Live Feed</span>
              </div>
            </div>

            <div className="relative flex-1 bg-black">
              <Webcam
                ref={webcamRef}
                mirrored={false}
                className="h-full w-full object-cover"
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: 1280,
                  height: 720,
                  facingMode: "user",
                }}
              />
              <canvas
                ref={overlayRef}
                className="pointer-events-none absolute inset-0 h-full w-full object-cover z-20"
                aria-label="MediaPipe hand landmarks overlay"
              />
              
              {/* Floating Progress Bar */}
              <div className="absolute inset-x-6 bottom-6 z-30 overflow-hidden rounded-2xl border border-white/10 bg-black/60 p-5 backdrop-blur-2xl shadow-xl transition-all duration-300">
                <div className="mb-3 flex items-center justify-between text-sm font-medium">
                  <span className="text-slate-300">Gesture Confidence</span>
                  <span className="text-indigo-300">
                    {Math.min(stabilityCount, STABILITY_TARGET)} / {STABILITY_TARGET} Frames
                  </span>
                </div>
                <div className="relative h-3 w-full overflow-hidden rounded-full bg-white/5">
                  <div className="absolute inset-0 bg-white/5" />
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-fuchsia-500 to-pink-500 transition-all duration-200 ease-out shadow-[0_0_15px_rgba(217,70,239,0.5)]"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            </div>
            {/* Glowing borders around camera feed using absolute inset */}
            <div className="pointer-events-none absolute inset-0 rounded-[2.5rem] ring-1 ring-inset ring-white/10 transition-all duration-500 group-hover:ring-white/20" />
          </div>

          {/* Side Panel */}
          <aside className="flex flex-col gap-6">
            {/* Prediction Card */}
            <div className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur-lg group">
              <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-fuchsia-500/20 blur-3xl transition-transform duration-700 group-hover:scale-150" />
              <div className="relative z-10">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Current Sign</p>
                <div className="mt-4 flex items-baseline gap-2">
                  <p className="text-7xl font-bold text-white tracking-tighter drop-shadow-lg">{candidate}</p>
                  {candidate !== "NOTHING" && <span className="animate-pulse text-2xl text-fuchsia-400">●</span>}
                </div>
                <p className="mt-6 text-sm leading-relaxed text-slate-400">
                  Hold the gesture steady for <strong className="text-indigo-300">15 frames</strong> to commit it to the sentence.
                </p>
              </div>
            </div>

            {/* Controls Card */}
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur-lg">
              <p className="mb-5 text-xs font-semibold uppercase tracking-widest text-slate-400">Controls</p>
              <div className="grid gap-3">
                <button
                  onClick={backspaceSentence}
                  className="group relative flex items-center justify-center gap-2 overflow-hidden rounded-2xl border border-white/10 bg-white/5 px-6 py-4 text-sm font-semibold transition-all hover:bg-white/10 hover:shadow-lg active:scale-95"
                >
                  <svg className="h-5 w-5 text-slate-300 group-hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M3 12l6.414 6.414a2 2 0 001.414.586H19a2 2 0 002-2V7a2 2 0 00-2-2h-8.172a2 2 0 00-1.414.586L3 12z" />
                  </svg>
                  <span className="text-slate-200 group-hover:text-white transition-colors">Backspace</span>
                </button>
                <button
                  onClick={clearSentence}
                  className="group relative flex items-center justify-center gap-2 overflow-hidden rounded-2xl border border-rose-500/30 bg-rose-500/10 px-6 py-4 text-sm font-semibold text-rose-200 transition-all hover:bg-rose-500/20 hover:shadow-[0_0_20px_rgba(244,63,94,0.3)] active:scale-95"
                >
                  <svg className="h-5 w-5 opacity-70 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <span>Clear All</span>
                </button>
              </div>
            </div>

            {/* Status Card */}
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur-3xl mt-auto">
              {error ? (
                <div className="flex gap-3 text-rose-300">
                  <svg className="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <p className="text-sm font-medium">{error}</p>
                </div>
              ) : (
                <div className="flex gap-3 text-emerald-300">
                  <svg className="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm font-medium">Neural engine connected. Awaiting gestures.</p>
                </div>
              )}
            </div>
          </aside>
        </section>

        {/* Sentence Output Strip */}
        <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur-3xl group">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 via-fuchsia-500/5 to-cyan-500/5 opacity-0 transition-opacity duration-1000 group-hover:opacity-100" />
          <div className="relative z-10">
            <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16m-7 6h7" />
              </svg>
              Translation Output
            </p>
            <div className="mt-5 min-h-[5rem] rounded-2xl border border-white/5 bg-black/40 p-5 shadow-inner backdrop-blur-sm">
              <p className="text-3xl font-medium leading-relaxed tracking-wide text-white md:text-4xl drop-shadow-md">
                {sentence ? (
                  <>
                    {sentence}
                    <span className="ml-1 inline-block h-8 w-1 animate-pulse bg-indigo-400 align-middle"></span>
                  </>
                ) : (
                  <span className="text-slate-500">Awaiting input...</span>
                )}
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
