const API_URL = 'http://localhost:8000';

export interface Video {
  id: number;
  youtube_id: string;
  title: string;
  url: string;
  thumbnail: string | null;
  processed: boolean;
  status: string;
  created_at: string;
}

export interface ChatMessage {
  user: string;
  assistant: string;
}

export const fetchVideos = async (): Promise<Video[]> => {
  const res = await fetch(`${API_URL}/videos`);
  if (!res.ok) throw new Error('Failed to fetch videos');
  const data = await res.json();
  return data.videos;
};

export const ingestVideo = async (url: string): Promise<Video> => {
  const res = await fetch(`${API_URL}/videos/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ youtube_url: url })
  });
  if (!res.ok) throw new Error('Failed to ingest video');
  return res.json();
};

export const transcribeVideo = async (youtubeId: string): Promise<any> => {
  const res = await fetch(`${API_URL}/videos/${youtubeId}/transcribe`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error('Failed to start transcription');
  return res.json();
};

export const chatWithVideo = async (youtubeId: string, query: string, searchMode: string = "both"): Promise<any> => {
  const res = await fetch(`${API_URL}/videos/${youtubeId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 5, search_mode: searchMode })
  });
  if (!res.ok) throw new Error('Chat request failed');
  return res.json();
};

export const deleteVideo = async (youtubeId: string): Promise<any> => {
  const res = await fetch(`${API_URL}/videos/${youtubeId}`, {
    method: 'DELETE'
  });
  if (!res.ok) throw new Error('Failed to delete video');
  return res.json();
};
