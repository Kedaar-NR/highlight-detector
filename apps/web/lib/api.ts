import { MediaFile, Session, HighlightEvent, RenderJob, OutputPreset } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
    private baseUrl: string;

    constructor(baseUrl: string = API_BASE) {
        this.baseUrl = baseUrl;
    }

    private async request<T>(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`;
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }

        return response.json();
    }

    // Session management
    async createSession(mediaFile: MediaFile, mode: string): Promise<Session> {
        return this.request<Session>('/api/sessions', {
            method: 'POST',
            body: JSON.stringify({ mediaFile, mode }),
        });
    }

    async getSession(sessionId: string): Promise<Session> {
        return this.request<Session>(`/api/sessions/${sessionId}`);
    }

    async deleteSession(sessionId: string): Promise<void> {
        return this.request<void>(`/api/sessions/${sessionId}`, {
            method: 'DELETE',
        });
    }

    // Detection
    async startDetection(sessionId: string): Promise<void> {
        return this.request<void>(`/api/sessions/${sessionId}/detect`, {
            method: 'POST',
        });
    }

    async getEvents(sessionId: string): Promise<HighlightEvent[]> {
        return this.request<HighlightEvent[]>(`/api/sessions/${sessionId}/events`);
    }

    // Rendering
    async startRender(
        sessionId: string,
        eventIds: string[],
        preset: OutputPreset
    ): Promise<RenderJob> {
        return this.request<RenderJob>('/api/render', {
            method: 'POST',
            body: JSON.stringify({ sessionId, eventIds, preset }),
        });
    }

    async getRenderJob(jobId: string): Promise<RenderJob> {
        return this.request<RenderJob>(`/api/render/${jobId}`);
    }

    // Media upload
    async uploadMedia(file: File): Promise<MediaFile> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/api/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        return response.json();
    }

    // Configuration
    async getOutputPresets(): Promise<OutputPreset[]> {
        return this.request<OutputPreset[]>('/api/presets');
    }

    // Health check
    async healthCheck(): Promise<{ status: string; version: string }> {
        return this.request<{ status: string; version: string }>('/api/health');
    }
}

export const apiClient = new ApiClient();
export default apiClient;
