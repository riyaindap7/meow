"use client";

import { useState, useRef } from "react";
import { Mic, Square, Languages } from "lucide-react";

interface VoiceInputProps {
  onTranscript: (text: string) => void;
  authToken: string;
}

const SUPPORTED_LANGUAGES = [
  { code: "en", name: "English", flag: "🇬🇧" },
  { code: "hi", name: "Hindi", flag: "🇮🇳" },
  { code: "ta", name: "Tamil", flag: "🇮🇳" },
  { code: "te", name: "Telugu", flag: "🇮🇳" },
  { code: "bn", name: "Bengali", flag: "🇮🇳" },
  { code: "mr", name: "Marathi", flag: "🇮🇳" },
  { code: "gu", name: "Gujarati", flag: "🇮🇳" },
  { code: "kn", name: "Kannada", flag: "🇮🇳" },
  { code: "ml", name: "Malayalam", flag: "🇮🇳" },
  { code: "pa", name: "Punjabi", flag: "🇮🇳" },
];

export default function VoiceInput({ onTranscript, authToken }: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  const [showLanguageMenu, setShowLanguageMenu] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus"
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        await transcribeAudio(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      console.log("🎤 Recording started...");
    } catch (err) {
      console.error("❌ Microphone access error:", err);
      setError("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      console.log("⏹️ Recording stopped");
    }
  };

  const transcribeAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");
      formData.append("language", selectedLanguage);

      console.log(`🔄 Transcribing audio (language: ${selectedLanguage})...`);

      const response = await fetch("http://localhost:8000/voice/transcribe", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${authToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Transcription failed");
      }

      const data = await response.json();
      console.log("✅ Transcription result:", data.transcript);

      onTranscript(data.transcript);
    } catch (err) {
      console.error("❌ Transcription error:", err);
      setError(err instanceof Error ? err.message : "Transcription failed");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="relative flex items-center gap-2">
      {/* Language Selector */}
      <div className="relative">
        <button
          onClick={() => setShowLanguageMenu(!showLanguageMenu)}
          className="p-2 rounded-lg bg-neutral-800/60 hover:bg-neutral-700/60 border border-neutral-700/50 transition-colors"
          title="Select Language"
        >
          <Languages className="w-5 h-5 text-neutral-300" />
        </button>

        {showLanguageMenu && (
          <div className="absolute bottom-12 right-0 bg-neutral-900/95 backdrop-blur-xl rounded-lg shadow-2xl border border-neutral-700/50 max-h-64 overflow-y-auto z-50">
            <div className="p-2 space-y-1">
              {SUPPORTED_LANGUAGES.map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => {
                    setSelectedLanguage(lang.code);
                    setShowLanguageMenu(false);
                  }}
                  className={`w-full text-left px-3 py-2 rounded-md flex items-center gap-2 transition-colors ${
                    selectedLanguage === lang.code
                      ? "bg-neutral-700/60 text-neutral-100"
                      : "hover:bg-neutral-800/60 text-neutral-300"
                  }`}
                >
                  <span className="text-xl">{lang.flag}</span>
                  <span className="text-sm font-medium">{lang.name}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Voice Recording Button */}
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isProcessing}
        className={`p-3 rounded-lg transition-all ${
          isRecording
            ? "bg-red-500/80 hover:bg-red-600/80 animate-pulse border border-red-400/50"
            : isProcessing
            ? "bg-neutral-800/60 cursor-not-allowed border border-neutral-700/50"
            : "bg-neutral-800/60 hover:bg-neutral-700/60 border border-neutral-700/50"
        } text-white shadow-lg`}
        title={isRecording ? "Stop Recording" : "Start Voice Input"}
      >
        {isProcessing ? (
          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
        ) : isRecording ? (
          <Square className="w-5 h-5" fill="white" />
        ) : (
          <Mic className="w-5 h-5" />
        )}
      </button>

      {/* Status/Error Display */}
      {(isRecording || isProcessing || error) && (
        <div className="absolute bottom-16 right-0 bg-neutral-900/95 backdrop-blur-xl rounded-lg shadow-2xl border border-neutral-700/50 px-4 py-2 min-w-48">
          {isRecording && (
            <p className="text-sm text-red-400 flex items-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
              Recording...
            </p>
          )}
          {isProcessing && (
            <p className="text-sm text-blue-400">Transcribing...</p>
          )}
          {error && (
            <p className="text-sm text-red-400">{error}</p>
          )}
        </div>
      )}

      {/* Current Language Badge */}
      <div className="absolute -top-8 right-0 text-xs text-neutral-400">
        {SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage)?.flag}{" "}
        {SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage)?.name}
      </div>
    </div>
  );
}