"use client";

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
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
  }, [handlePrediction]);

  const progress = Math.min((stabilityCount / STABILITY_TARGET) * 100, 100);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#070a0f] text-slate-100">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_14%,rgba(249,115,22,0.16),transparent_42%),radial-gradient(circle_at_85%_16%,rgba(20,184,166,0.22),transparent_36%),linear-gradient(160deg,#070a0f_0%,#0f1723_42%,#09111a_100%)]" />

      <main className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-5 px-4 py-5 md:px-6">
        <header className="rounded-3xl border border-white/15 bg-white/6 px-5 py-4 shadow-[0_14px_40px_-18px_rgba(0,0,0,0.85)] backdrop-blur-xl">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.32em] text-teal-300">Realtime Sign Interpreter</p>
              <h1 className="mt-1 text-3xl font-semibold tracking-tight text-white md:text-4xl">
                ASL Stream Console
              </h1>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full border border-white/15 bg-black/35 px-4 py-1.5 text-xs uppercase tracking-[0.16em] text-slate-200">
                {isReady ? "Tracker Online" : "Booting Tracker"}
              </span>
              <span className="rounded-full border border-teal-200/35 bg-teal-300/12 px-4 py-1.5 text-xs uppercase tracking-[0.16em] text-teal-200">
                30 FPS Target
              </span>
            </div>
          </div>
        </header>

        <section className="grid flex-1 gap-5 lg:grid-cols-[1.8fr_1fr]">
          <div className="flex min-h-[430px] flex-col overflow-hidden rounded-3xl border border-white/15 bg-black/35 shadow-[0_20px_55px_-18px_rgba(0,0,0,0.8)]">
            <div className="flex items-center justify-between border-b border-white/10 bg-black/25 px-4 py-3">
              <p className="text-sm font-medium tracking-wide text-slate-200">Camera Feed</p>
              <span className="rounded-xl border border-white/20 bg-white/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-slate-100">
                Not Mirrored
              </span>
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
              <div className="absolute inset-x-4 bottom-4 rounded-2xl border border-white/12 bg-[#0b1320]/85 p-3 backdrop-blur-lg">
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="text-slate-300">Stability Progress</span>
                  <span className="font-semibold text-teal-300">
                    {Math.min(stabilityCount, STABILITY_TARGET)}/{STABILITY_TARGET}
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700/70">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-orange-400 via-teal-300 to-emerald-300 transition-all duration-100"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <aside className="grid gap-4">
            <section className="rounded-3xl border border-white/15 bg-white/7 p-4 backdrop-blur-xl">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-300">Current Prediction</p>
              <p className="mt-3 text-5xl font-semibold leading-none text-teal-300">{candidate}</p>
              <p className="mt-3 text-sm text-slate-300">A letter is committed after 15 consecutive frames.</p>
            </section>

            <section className="rounded-3xl border border-white/15 bg-white/7 p-4 backdrop-blur-xl">
              <p className="mb-3 text-xs uppercase tracking-[0.2em] text-slate-300">Controls</p>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={backspaceSentence}
                  className="rounded-xl border border-white/15 bg-white/12 px-4 py-2.5 text-sm font-medium transition hover:bg-white/20"
                >
                  Backspace
                </button>
                <button
                  onClick={clearSentence}
                  className="rounded-xl border border-orange-300/35 bg-orange-400/18 px-4 py-2.5 text-sm font-medium text-orange-50 transition hover:bg-orange-400/28"
                >
                  Clear
                </button>
              </div>
            </section>

            <section className="rounded-3xl border border-white/15 bg-white/7 p-4 backdrop-blur-xl">
              {error ? (
                <p className="rounded-xl border border-amber-300/35 bg-amber-300/10 px-3 py-2 text-sm text-amber-100">
                  {error}
                </p>
              ) : (
                <p className="rounded-xl border border-teal-300/35 bg-teal-300/10 px-3 py-2 text-sm text-teal-100">
                  Inference active. New API calls wait until the current request is complete.
                </p>
              )}
            </section>
          </aside>
        </section>

        <section className="rounded-3xl border border-white/15 bg-black/30 p-4 shadow-[0_20px_50px_-24px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Sentence Strip</p>
          <div className="mt-3 min-h-20 rounded-2xl border border-white/12 bg-[#0a1422]/80 px-4 py-3 text-2xl leading-relaxed tracking-[0.04em] text-white md:text-3xl">
            {sentence || "..."}
          </div>
        </section>
      </main>
    </div>
  );
}
