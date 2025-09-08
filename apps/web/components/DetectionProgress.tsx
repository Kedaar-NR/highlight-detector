"use client";

import { motion } from "framer-motion";
import { Activity, Clock, CheckCircle } from "lucide-react";

interface DetectionProgressProps {
  stage?: string;
  progress?: number;
  message?: string;
  estimatedTimeRemaining?: number;
}

const stages = [
  {
    id: "ingest",
    name: "Ingesting Media",
    description: "Analyzing video and audio streams",
  },
  {
    id: "audio_features",
    name: "Audio Analysis",
    description: "Extracting audio features and patterns",
  },
  {
    id: "vision_features",
    name: "Vision Analysis",
    description: "Detecting motion and visual cues",
  },
  {
    id: "fusion",
    name: "Feature Fusion",
    description: "Combining audio and visual signals",
  },
  {
    id: "classification",
    name: "Classification",
    description: "Identifying highlight moments",
  },
  { id: "complete", name: "Complete", description: "Detection finished" },
];

export function DetectionProgress({
  stage = "ingest",
  progress = 0,
  message = "Starting detection...",
  estimatedTimeRemaining = 60,
}: DetectionProgressProps) {
  const currentStageIndex = stages.findIndex((s) => s.id === stage);
  const currentStage = stages[currentStageIndex];

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="card">
      <div className="flex items-center space-x-3 mb-6">
        <div className="p-2 bg-primary-100 rounded-lg">
          <Activity className="w-6 h-6 text-primary-600" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-gray-900">
            Detecting Highlights
          </h2>
          <p className="text-gray-600">{message}</p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">
            Overall Progress
          </span>
          <span className="text-sm text-gray-600">
            {Math.round(progress * 100)}%
          </span>
        </div>
        <div className="timeline-track">
          <motion.div
            className="timeline-progress"
            initial={{ width: 0 }}
            animate={{ width: `${progress * 100}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Stage Progress */}
      <div className="space-y-4">
        {stages.map((stageItem, index) => {
          const isCompleted = index < currentStageIndex;
          const isCurrent = index === currentStageIndex;
          const isPending = index > currentStageIndex;

          return (
            <motion.div
              key={stageItem.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`flex items-center space-x-3 p-3 rounded-lg transition-colors duration-200 ${
                isCurrent
                  ? "bg-primary-50 border border-primary-200"
                  : isCompleted
                  ? "bg-green-50 border border-green-200"
                  : "bg-gray-50 border border-gray-200"
              }`}
            >
              <div
                className={`p-1 rounded-full ${
                  isCompleted
                    ? "bg-green-100"
                    : isCurrent
                    ? "bg-primary-100"
                    : "bg-gray-100"
                }`}
              >
                {isCompleted ? (
                  <CheckCircle className="w-5 h-5 text-green-600" />
                ) : isCurrent ? (
                  <div className="w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <div className="w-5 h-5 border-2 border-gray-300 rounded-full" />
                )}
              </div>

              <div className="flex-1">
                <h3
                  className={`font-medium ${
                    isCurrent
                      ? "text-primary-900"
                      : isCompleted
                      ? "text-green-900"
                      : "text-gray-700"
                  }`}
                >
                  {stageItem.name}
                </h3>
                <p
                  className={`text-sm ${
                    isCurrent
                      ? "text-primary-700"
                      : isCompleted
                      ? "text-green-700"
                      : "text-gray-500"
                  }`}
                >
                  {stageItem.description}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Time Estimate */}
      {estimatedTimeRemaining > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-6 p-4 bg-gray-50 rounded-lg"
        >
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-gray-600" />
            <span className="text-sm text-gray-600">
              Estimated time remaining: {formatTime(estimatedTimeRemaining)}
            </span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
