"use client";

import { useState, useEffect } from "react";
import { useAppStore, detectionModes } from "@/store/useAppStore";
import { apiClient } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { MediaUpload } from "@/components/MediaUpload";
import { ModeSelector } from "@/components/ModeSelector";
import { DetectionProgress } from "@/components/DetectionProgress";
import { Timeline } from "@/components/Timeline";
import { EventInspector } from "@/components/EventInspector";
import { Header } from "@/components/Header";
import { WebSocketMessage } from "@/types";

export default function Home() {
  const {
    currentSession,
    mediaFile,
    selectedMode,
    events,
    isDetecting,
    selectedEventId,
    setCurrentSession,
    setMediaFile,
    setSelectedMode,
    setEvents,
    setIsDetecting,
    setSelectedEventId,
    setIsConnected,
  } = useAppStore();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection for real-time updates
  const { isConnected, sendMessage } = useWebSocket({
    url: process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws",
    onMessage: (message: WebSocketMessage) => {
      switch (message.type) {
        case "progress":
          // Update detection progress
          break;
        case "event":
          // Add new event
          setEvents([...events, message.data]);
          break;
        case "complete":
          setIsDetecting(false);
          break;
        case "error":
          setError(message.data.message);
          setIsDetecting(false);
          break;
      }
    },
    onOpen: () => setIsConnected(true),
    onClose: () => setIsConnected(false),
  });

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const uploadedFile = await apiClient.uploadMedia(file);
      setMediaFile(uploadedFile);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleModeSelect = (mode: (typeof detectionModes)[0]) => {
    setSelectedMode(mode);
  };

  const handleStartDetection = async () => {
    if (!mediaFile || !selectedMode || !currentSession) return;

    setIsDetecting(true);
    setError(null);

    try {
      await apiClient.startDetection(currentSession.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
      setIsDetecting(false);
    }
  };

  const handleCreateSession = async () => {
    if (!mediaFile || !selectedMode) return;

    setIsLoading(true);
    setError(null);

    try {
      const session = await apiClient.createSession(mediaFile, selectedMode.id);
      setCurrentSession(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Session creation failed");
    } finally {
      setIsLoading(false);
    }
  };

  const selectedEvent = events.find((e) => e.id === selectedEventId);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {!mediaFile ? (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-4">
                Highlight Detector
              </h1>
              <p className="text-lg text-gray-600">
                Upload an MP4 file to detect exciting moments in sports or
                podcast content
              </p>
            </div>
            <MediaUpload onUpload={handleFileUpload} isLoading={isLoading} />
          </div>
        ) : !selectedMode ? (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Choose Detection Mode
              </h2>
              <p className="text-gray-600">
                Select the type of content to optimize detection for
              </p>
            </div>
            <ModeSelector modes={detectionModes} onSelect={handleModeSelect} />
          </div>
        ) : !currentSession ? (
          <div className="max-w-2xl mx-auto text-center">
            <div className="card mb-6">
              <h3 className="text-xl font-semibold mb-4">Ready to Detect</h3>
              <div className="space-y-4">
                <div className="text-left">
                  <p className="text-sm text-gray-600">File:</p>
                  <p className="font-medium">{mediaFile.name}</p>
                </div>
                <div className="text-left">
                  <p className="text-sm text-gray-600">Mode:</p>
                  <p className="font-medium">{selectedMode.name}</p>
                </div>
              </div>
            </div>
            <button
              onClick={handleCreateSession}
              disabled={isLoading}
              className="btn-primary"
            >
              {isLoading ? "Creating Session..." : "Start Detection"}
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Timeline Area */}
            <div className="lg:col-span-2 space-y-6">
              {isDetecting ? (
                <DetectionProgress />
              ) : (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h2 className="text-xl font-semibold">Timeline</h2>
                    <button
                      onClick={handleStartDetection}
                      className="btn-primary"
                    >
                      Detect Highlights
                    </button>
                  </div>
                  <Timeline
                    events={events}
                    selectedEventId={selectedEventId}
                    onEventSelect={setSelectedEventId}
                    duration={mediaFile.duration}
                  />
                </div>
              )}
            </div>

            {/* Event Inspector */}
            <div className="lg:col-span-1">
              <EventInspector event={selectedEvent} mediaFile={mediaFile} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
