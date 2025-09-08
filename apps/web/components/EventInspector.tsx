"use client";

import { HighlightEvent, MediaFile, OutputPreset } from "@/types";
import { motion } from "framer-motion";
import {
  Play,
  Download,
  Settings,
  BarChart3,
  Image,
  Clock,
} from "lucide-react";
import { useState } from "react";

interface EventInspectorProps {
  event: HighlightEvent | undefined;
  mediaFile: MediaFile;
}

const outputPresets: OutputPreset[] = [
  {
    id: "vertical",
    name: "Vertical (9:16)",
    width: 1080,
    height: 1920,
    aspectRatio: "9:16",
    cropStrategy: "motion_centroid",
  },
  {
    id: "square",
    name: "Square (1:1)",
    width: 1080,
    height: 1080,
    aspectRatio: "1:1",
    cropStrategy: "center",
  },
  {
    id: "wide",
    name: "Wide (16:9)",
    width: 1920,
    height: 1080,
    aspectRatio: "16:9",
    cropStrategy: "center",
  },
];

export function EventInspector({ event, mediaFile }: EventInspectorProps) {
  const [selectedPreset, setSelectedPreset] = useState<OutputPreset>(
    outputPresets[0]
  );
  const [isRendering, setIsRendering] = useState(false);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const handleRender = async () => {
    if (!event) return;

    setIsRendering(true);
    // TODO: Implement render API call
    setTimeout(() => setIsRendering(false), 2000);
  };

  if (!event) {
    return (
      <div className="card">
        <div className="text-center py-8 text-gray-500">
          <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>Select an event to view details</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-6"
    >
      {/* Event Details */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-primary-100 rounded-lg">
            <BarChart3 className="w-5 h-5 text-primary-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {event.label}
            </h3>
            <p className="text-sm text-gray-600">{event.category}</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Confidence</span>
            <span className="text-sm font-medium">
              {Math.round(event.confidence * 100)}%
            </span>
          </div>

          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Duration</span>
            <span className="text-sm font-medium">
              {formatTime(event.endTime - event.startTime)}
            </span>
          </div>

          <div className="flex justify-between">
            <span className="text-sm text-gray-600">Start Time</span>
            <span className="text-sm font-medium">
              {formatTime(event.startTime)}
            </span>
          </div>

          <div className="flex justify-between">
            <span className="text-sm text-gray-600">End Time</span>
            <span className="text-sm font-medium">
              {formatTime(event.endTime)}
            </span>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mt-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-600">Confidence</span>
            <span className="text-xs text-gray-600">
              {Math.round(event.confidence * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-primary-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${event.confidence * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Feature Analysis */}
      <div className="card">
        <h4 className="text-sm font-medium text-gray-700 mb-4">
          Feature Analysis
        </h4>
        <div className="space-y-3">
          {Object.entries(event.features).map(([key, value]) => (
            <div key={key} className="flex justify-between items-center">
              <span className="text-sm text-gray-600 capitalize">
                {key.replace(/([A-Z])/g, " $1").trim()}
              </span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-gray-200 rounded-full h-1.5">
                  <div
                    className="bg-primary-500 h-1.5 rounded-full"
                    style={{ width: `${value * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 w-8 text-right">
                  {Math.round(value * 100)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Evidence */}
      {event.evidence && (
        <div className="card">
          <h4 className="text-sm font-medium text-gray-700 mb-4">Evidence</h4>
          <div className="grid grid-cols-2 gap-3">
            {event.evidence.audioChart && (
              <div className="text-center">
                <Image className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-xs text-gray-600">Audio Chart</p>
              </div>
            )}
            {event.evidence.motionChart && (
              <div className="text-center">
                <Image className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-xs text-gray-600">Motion Chart</p>
              </div>
            )}
            {event.evidence.scoreboardBefore && (
              <div className="text-center">
                <Image className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-xs text-gray-600">Scoreboard Before</p>
              </div>
            )}
            {event.evidence.scoreboardAfter && (
              <div className="text-center">
                <Image className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-xs text-gray-600">Scoreboard After</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Export Options */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="w-4 h-4 text-gray-600" />
          <h4 className="text-sm font-medium text-gray-700">Export Options</h4>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Output Format
            </label>
            <div className="grid grid-cols-1 gap-2">
              {outputPresets.map((preset) => (
                <label
                  key={preset.id}
                  className={`flex items-center space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedPreset.id === preset.id
                      ? "border-primary-300 bg-primary-50"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <input
                    type="radio"
                    name="preset"
                    value={preset.id}
                    checked={selectedPreset.id === preset.id}
                    onChange={() => setSelectedPreset(preset)}
                    className="text-primary-600 focus:ring-primary-500"
                  />
                  <div>
                    <div className="text-sm font-medium text-gray-900">
                      {preset.name}
                    </div>
                    <div className="text-xs text-gray-600">
                      {preset.width} × {preset.height}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={handleRender}
            disabled={isRendering}
            className="w-full btn-primary flex items-center justify-center space-x-2"
          >
            {isRendering ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                <span>Rendering...</span>
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                <span>Export Clip</span>
              </>
            )}
          </button>
        </div>
      </div>
    </motion.div>
  );
}
