import { useEffect, useRef } from "react";

export function useWebRTC({
  videoRef,
  onResult,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  onResult: (result: any) => void;
}) {
  const pcRef = useRef<RTCPeerConnection | null>(null);


  useEffect(() => {
    async function setup() {
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      pcRef.current = pc;

      // ✅ MUST be set before offer/answer
      pc.ondatachannel = (event) => {
        const channel = event.channel;
        console.log("DataChannel received:", channel.label);

        channel.onopen = () => {
          console.log("✅ DataChannel open on frontend");
        };

        channel.onmessage = (e) => {
          const data = JSON.parse(e.data);
          console.log("Received data:", data);
          onResult(data);
        };
      };

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }

      stream.getTracks().forEach((track) => pc.addTrack(track, stream));

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // 🔥 WAIT FOR ICE GATHERING TO COMPLETE
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          };
          pc.addEventListener("icegatheringstatechange", checkState);
        }
      });

      const res = await fetch("http://localhost:8000/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // ✅ SEND THE FINAL SDP WITH ICE CANDIDATES
        body: JSON.stringify(pc.localDescription),
      });

      const answer = await res.json();
      await pc.setRemoteDescription(answer);
    }

    setup();

    return () => {
      pcRef.current?.close();
    };
  }, [onResult, videoRef]);

}
