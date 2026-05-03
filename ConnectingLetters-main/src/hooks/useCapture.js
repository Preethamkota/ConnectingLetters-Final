import React, { useRef, useEffect } from "react";

let video = null;
let canvas = null;
let uploadUrl = null;
let analyzeUrl = "http://localhost:8000/analyze";
let intervalTime = 3000;
let intervalId = null;
let metricsProvider = () => ({});
let analysisCallback = null;
let analysisSamples = [];

// Assigns the video and canvas elements once they're ready
const assignElements = (videoEl, canvasEl) => {
  video = videoEl;
  canvas = canvasEl;
};

// Allows setting a custom upload URL
const assignUploadUrl = (url) => {
  if (url) uploadUrl = url;
};

const assignAnalyzeUrl = (url) => {
  if (url) analyzeUrl = url;
};

// Allows customizing the time interval between automatic captures
const assignInterval = (interval) => {
  if (interval) intervalTime = interval;
};

const assignMetricsProvider = (provider) => {
  if (typeof provider === "function") metricsProvider = provider;
};

const assignAnalysisCallback = (callback) => {
  analysisCallback = typeof callback === "function" ? callback : null;
};

const resetAnalysis = () => {
  analysisSamples = [];
};

const buildAnalysisSummary = () => {
  const validSamples = analysisSamples.filter((sample) => !sample.error);
  const totalSamples = validSamples.length;
  const emotionLabels = ["happy", "neutral", "frustrated", "confused"];
  const emotionCounts = emotionLabels.reduce((counts, label) => {
    counts[label] = 0;
    return counts;
  }, {});

  let focusedSamples = 0;
  let distractions = 0;
  let focusRun = 0;
  let focusRunCount = 0;
  let focusRunTotalSeconds = 0;
  let previousFocused = null;

  validSamples.forEach((sample) => {
    const label = sample.emotion_label;
    if (label && emotionCounts[label] !== undefined) {
      emotionCounts[label] += 1;
    }

    const focused = sample.focused === 1;
    if (focused) {
      focusedSamples += 1;
      focusRun += 1;
    } else {
      if (previousFocused === true) distractions += 1;
      if (focusRun > 0) {
        focusRunCount += 1;
        focusRunTotalSeconds += focusRun * (intervalTime / 1000);
        focusRun = 0;
      }
    }
    previousFocused = focused;
  });

  if (focusRun > 0) {
    focusRunCount += 1;
    focusRunTotalSeconds += focusRun * (intervalTime / 1000);
  }

  const emotionPercentages = emotionLabels.reduce((percentages, label) => {
    percentages[label] =
      totalSamples === 0 ? 0 : Number(((emotionCounts[label] / totalSamples) * 100).toFixed(2));
    return percentages;
  }, {});

  return {
    sampleCount: totalSamples,
    failedSampleCount: analysisSamples.length - totalSamples,
    emotion: {
      percentages: emotionPercentages,
      counts: emotionCounts,
      frustrationSpikes: emotionCounts.frustrated,
    },
    gaze: {
      percentTimeFocused: totalSamples === 0 ? 0 : Number(((focusedSamples / totalSamples) * 100).toFixed(2)),
      distractions,
      averageFocusDuration:
        focusRunCount === 0 ? 0 : Number((focusRunTotalSeconds / focusRunCount).toFixed(2)),
    },
    samples: analysisSamples,
  };
};

const getAnalysisSummary = () => buildAnalysisSummary();

const postFormData = (url, formData) =>
  fetch(url, {
    method: "POST",
    body: formData,
  }).then((res) => res.json());

// Captures a snapshot from the video and uploads it to the server
const capture = () => {
  if (!video || !canvas) return;

  // Wait until video has valid dimensions
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    console.warn("Video dimensions are not ready yet.");
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw the current video frame onto the canvas
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert canvas content to an image blob, analyze it locally, and optionally upload it.
  canvas.toBlob((blob) => {
    if (!blob) return;

    const metrics = metricsProvider();

    if (analyzeUrl) {
      const analysisFormData = new FormData();
      analysisFormData.append("image", blob, "capture.png");
      analysisFormData.append("metrics", JSON.stringify(metrics));

      postFormData(analyzeUrl, analysisFormData)
        .then((data) => {
          const sample = {
            ...data,
            capturedAt: new Date().toISOString(),
          };
          analysisSamples.push(sample);
          if (analysisCallback) analysisCallback(sample, buildAnalysisSummary());
          console.log("Analyzed:", sample);
        })
        .catch((err) => console.error("Analysis failed:", err));
    }

    if (uploadUrl) {
      const uploadFormData = new FormData();
      uploadFormData.append("image", blob, "capture.png");

      postFormData(uploadUrl, uploadFormData)
        .then((data) => console.log("Uploaded:", data))
        .catch((err) => console.error("Upload failed:", err));
    }
  });
};

// Start auto capture at interval
const startCapture = () => {
  if (intervalId) return;
  intervalId = setInterval(capture, intervalTime);
};

// Stop automaticimage capturing
const endCapture = () => {
  clearInterval(intervalId);
  intervalId = null;

  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }
};

// This component sets up the hidden video and canvas elements and starts the webcam
const CaptureElements = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;

    if (!videoEl || !canvasEl) return;

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoEl.srcObject = stream;

        videoEl.onloadedmetadata = () => {
          videoEl.play();
          assignElements(videoEl, canvasEl);
        };
      })
      .catch((err) => {
        console.error("Failed to access webcam:", err);
      });

    return () => {
      if (videoEl.srcObject) {
        const tracks = videoEl.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);
  return (
    <>
      <video ref={videoRef} autoPlay playsInline style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{ display: "none" }}
      />
    </>
  );
};

export {
  CaptureElements, // Component to initialize video/canvas
  startCapture, // Call to begin auto-capturing
  endCapture, // Call to stop auto-capturing
  assignUploadUrl, // Set your custom upload path
  assignAnalyzeUrl, // Set the FastAPI /analyze endpoint
  assignInterval, // Set capture interval
  assignMetricsProvider, // Attach current game metrics to each analysis request
  assignAnalysisCallback, // Receive each gaze/emotion result
  getAnalysisSummary, // Build final emotion/gaze summaries
  resetAnalysis, // Clear samples when a new game starts
};
