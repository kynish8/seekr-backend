import { useEffect, useRef, useState } from "react";
import { useWebRTC } from "../hooks/useWebRTC";

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [label, setLabel] = useState("…");

  useWebRTC({
    videoRef,
    onResult: (result) => {
      console.log(result);
      setLabel(result.label);
    },
  });

  return (
    <div style={{ position: "relative", width: 400 }}>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{ width: "100%" }}
      />

      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: 12,
          padding: "6px 10px",
          background: "rgba(0,0,0,0.7)",
          color: "white",
          fontSize: 18,
          borderRadius: 6,
        }}
      >
        {label}
      </div>
    </div>
  );
}
