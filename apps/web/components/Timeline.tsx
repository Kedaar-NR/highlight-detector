"use client";

import { HighlightEvent, MediaFile } from "@/types";
import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";
import { useState, useRef, useEffect } from "react";

interface TimelineProps {
  events: HighlightEvent[];
  selectedEventId: string | null;
  onEventSelect: (eventId: string | null) => void;
  duration: number;
}

export function Timeline({
  events,
  selectedEventId,
  onEventSelect,
  duration,
}: TimelineProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getEventPosition = (event: HighlightEvent) => {
    return (event.startTime / duration) * 100;
  };

  const getEventWidth = (event: HighlightEvent) => {
    return ((event.endTime - event.startTime) / duration) * 100;
  };

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;

    setCurrentTime(newTime);
  };

  const handleEventClick = (event: HighlightEvent) => {
    onEventSelect(event.id);
    setCurrentTime(event.startTime);
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Timeline</h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5" />
            )}
          </button>
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            {isMuted ? (
              <VolumeX className="w-5 h-5" />
            ) : (
              <Volume2 className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Time Display */}
      <div className="flex justify-between items-center mb-4">
        <span className="text-sm text-gray-600">{formatTime(currentTime)}</span>
        <span className="text-sm text-gray-600">{formatTime(duration)}</span>
      </div>

      {/* Timeline Track */}
      <div
        ref={timelineRef}
        className="relative h-16 bg-gray-200 rounded-lg cursor-pointer overflow-hidden"
        onClick={handleTimelineClick}
      >
        {/* Current Time Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-primary-600 z-10"
          style={{ left: `${(currentTime / duration) * 100}%` }}
        />

        {/* Events */}
        {events.map((event, index) => (
          <motion.div
            key={event.id}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className={`absolute top-2 bottom-2 rounded cursor-pointer transition-all duration-200 ${
              selectedEventId === event.id
                ? "bg-primary-600 shadow-lg"
                : "bg-primary-400 hover:bg-primary-500"
            }`}
            style={{
              left: `${getEventPosition(event)}%`,
              width: `${getEventWidth(event)}%`,
            }}
            onClick={(e) => {
              e.stopPropagation();
              handleEventClick(event);
            }}
          >
            <div className="h-full flex items-center justify-center">
              <div className="text-center text-white">
                <div className="text-xs font-medium truncate px-2">
                  {event.label}
                </div>
                <div className="text-xs opacity-75">
                  {Math.round(event.confidence * 100)}%
                </div>
              </div>
            </div>
          </motion.div>
        ))}

        {/* Time Markers */}
        <div className="absolute inset-0 pointer-events-none">
          {Array.from({ length: Math.ceil(duration / 60) }, (_, i) => (
            <div
              key={i}
              className="absolute top-0 bottom-0 w-px bg-gray-300"
              style={{ left: `${((i * 60) / duration) * 100}%` }}
            />
          ))}
        </div>
      </div>

      {/* Event List */}
      {events.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-medium text-gray-700 mb-3">
            Detected Events
          </h4>
          <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-hide">
            {events.map((event) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`p-3 rounded-lg border cursor-pointer transition-colors duration-200 ${
                  selectedEventId === event.id
                    ? "border-primary-300 bg-primary-50"
                    : "border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => onEventSelect(event.id)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">
                      {event.label}
                    </div>
                    <div className="text-sm text-gray-600">
                      {formatTime(event.startTime)} -{" "}
                      {formatTime(event.endTime)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      {Math.round(event.confidence * 100)}%
                    </div>
                    <div className="text-xs text-gray-500">
                      {event.category}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {events.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <p>
            No events detected yet. Click "Detect Highlights" to start analysis.
          </p>
        </div>
      )}
    </div>
  );
}
