import React, { useRef, useEffect } from "react";

let video = null;
let canvas = null;
let uploadUrl = null;
let intervalTime = 3000;
let intervalId = null;

// Assigns the video and canvas elements once they're ready
const assignElements = (videoEl, canvasEl) => {
  video = videoEl;
  canvas = canvasEl;
};

// Allows setting a custom upload URL
const assignUploadUrl = (url) => {
  if (url) uploadUrl = url;
};

// Allows customizing the time interval between automatic captures
const assignInterval = (interval) => {
  if (interval) intervalTime = interval;
};

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

  // Convert canvas content to an image blob and upload it
  canvas.toBlob((blob) => {
    if (!blob) return;

    const formData = new FormData();
    formData.append("image", blob, "capture.png");

    fetch(uploadUrl, {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => console.log("Uploaded:", data))
      .catch((err) => console.error("Upload failed:", err));
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
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
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
  assignInterval, // Set capture interval
};