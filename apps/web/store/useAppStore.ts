import { create } from 'zustand';
import { MediaFile, Session, HighlightEvent, DetectionMode, RenderJob } from '@/types';

interface AppState {
    // Current session
    currentSession: Session | null;
    setCurrentSession: (session: Session | null) => void;

    // Media file
    mediaFile: MediaFile | null;
    setMediaFile: (file: MediaFile | null) => void;

    // Detection mode
    selectedMode: DetectionMode | null;
    setSelectedMode: (mode: DetectionMode | null) => void;

    // Events
    events: HighlightEvent[];
    setEvents: (events: HighlightEvent[]) => void;
    addEvent: (event: HighlightEvent) => void;
    updateEvent: (id: string, updates: Partial<HighlightEvent>) => void;
    removeEvent: (id: string) => void;

    // UI state
    isDetecting: boolean;
    setIsDetecting: (detecting: boolean) => void;

    selectedEventId: string | null;
    setSelectedEventId: (id: string | null) => void;

    // Render jobs
    renderJobs: RenderJob[];
    addRenderJob: (job: RenderJob) => void;
    updateRenderJob: (id: string, updates: Partial<RenderJob>) => void;

    // WebSocket connection
    isConnected: boolean;
    setIsConnected: (connected: boolean) => void;

    // Reset
    reset: () => void;
}

const detectionModes: DetectionMode[] = [
    {
        id: 'sports',
        name: 'Sports',
        description: 'Detect exciting moments in sports broadcasts',
        icon: 'sports'
    },
    {
        id: 'podcast',
        name: 'Podcast',
        description: 'Find highlights in podcast conversations',
        icon: 'podcast'
    }
];

export const useAppStore = create<AppState>((set, get) => ({
    // Current session
    currentSession: null,
    setCurrentSession: (session) => set({ currentSession: session }),

    // Media file
    mediaFile: null,
    setMediaFile: (file) => set({ mediaFile: file }),

    // Detection mode
    selectedMode: null,
    setSelectedMode: (mode) => set({ selectedMode: mode }),

    // Events
    events: [],
    setEvents: (events) => set({ events }),
    addEvent: (event) => set((state) => ({ events: [...state.events, event] })),
    updateEvent: (id, updates) => set((state) => ({
        events: state.events.map(event =>
            event.id === id ? { ...event, ...updates } : event
        )
    })),
    removeEvent: (id) => set((state) => ({
        events: state.events.filter(event => event.id !== id)
    })),

    // UI state
    isDetecting: false,
    setIsDetecting: (detecting) => set({ isDetecting: detecting }),

    selectedEventId: null,
    setSelectedEventId: (id) => set({ selectedEventId: id }),

    // Render jobs
    renderJobs: [],
    addRenderJob: (job) => set((state) => ({ renderJobs: [...state.renderJobs, job] })),
    updateRenderJob: (id, updates) => set((state) => ({
        renderJobs: state.renderJobs.map(job =>
            job.id === id ? { ...job, ...updates } : job
        )
    })),

    // WebSocket connection
    isConnected: false,
    setIsConnected: (connected) => set({ isConnected: connected }),

    // Reset
    reset: () => set({
        currentSession: null,
        mediaFile: null,
        selectedMode: null,
        events: [],
        isDetecting: false,
        selectedEventId: null,
        renderJobs: [],
        isConnected: false
    })
}));

export { detectionModes };
