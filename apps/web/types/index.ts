export interface MediaFile {
  id: string;
  name: string;
  path: string;
  size: number;
  duration: number;
  resolution: {
    width: number;
    height: number;
  };
  fps: number;
  audioFormat: string;
  videoFormat: string;
}

export interface DetectionMode {
  id: 'sports' | 'podcast';
  name: string;
  description: string;
  icon: string;
}

export interface HighlightEvent {
  id: string;
  startTime: number;
  endTime: number;
  confidence: number;
  label: string;
  category: string;
  evidence: EventEvidence;
  features: EventFeatures;
}

export interface EventEvidence {
  audioChart?: string;
  motionChart?: string;
  scoreboardBefore?: string;
  scoreboardAfter?: string;
  classifierLogits: number[];
  topFeatures: string[];
}

export interface EventFeatures {
  audioPeak: number;
  motionMagnitude: number;
  voiceActivity: number;
  prosody: number;
  shotBoundary: number;
  scoreboardChange?: number;
  replayCue?: number;
  laughter?: number;
  applause?: number;
  excitement?: number;
  topicShift?: number;
}

export interface Session {
  id: string;
  mediaFile: MediaFile;
  mode: DetectionMode;
  status: 'idle' | 'detecting' | 'completed' | 'error';
  progress: number;
  events: HighlightEvent[];
  createdAt: Date;
  updatedAt: Date;
}

export interface RenderJob {
  id: string;
  sessionId: string;
  eventIds: string[];
  preset: OutputPreset;
  status: 'pending' | 'rendering' | 'completed' | 'error';
  progress: number;
  outputPath?: string;
  createdAt: Date;
}

export interface OutputPreset {
  id: string;
  name: string;
  width: number;
  height: number;
  aspectRatio: string;
  cropStrategy: 'center' | 'motion_centroid' | 'face_tracking';
}

export interface WebSocketMessage {
  type: 'progress' | 'event' | 'error' | 'complete';
  data: any;
  timestamp: Date;
}

export interface DetectionProgress {
  stage: 'ingest' | 'audio_features' | 'vision_features' | 'fusion' | 'classification' | 'complete';
  progress: number;
  message: string;
  estimatedTimeRemaining?: number;
}
