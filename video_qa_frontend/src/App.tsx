import { useState, useEffect } from 'react';
import { PlaySquare, MessageSquare, Search, Plus, Video, Image as ImageIcon, Trash2, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { fetchVideos, ingestVideo, transcribeVideo, chatWithVideo, deleteVideo } from './api';
import type { Video as VideoType } from './api';

function App() {
  const [videos, setVideos] = useState<VideoType[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoType | null>(null);
  const [urlInput, setUrlInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const [chatQuery, setChatQuery] = useState('');
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [isChatting, setIsChatting] = useState(false);
  const [searchMode, setSearchMode] = useState('both');

  useEffect(() => {
    loadVideos();
    const interval = setInterval(() => {
      loadVideos();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadVideos = async () => {
    try {
      const data = await fetchVideos();
      setVideos(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleIngest = async () => {
    if (!urlInput) return;
    setIsLoading(true);
    try {
      await ingestVideo(urlInput);
      setUrlInput('');
      await loadVideos();
    } catch (err) {
      console.error(err);
    }
    setIsLoading(false);
  };

  const handleTranscribe = async (v: VideoType) => {
    try {
      await transcribeVideo(v.youtube_id);
      alert('Transcription and Visual Extraction started in the background!');
      loadVideos();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (e: React.MouseEvent, v: VideoType) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete ${v.title} and all its extracted data?`)) return;
    
    try {
      await deleteVideo(v.youtube_id);
      if (selectedVideo?.id === v.id) {
        setSelectedVideo(null);
        setChatHistory([]);
      }
      loadVideos();
    } catch (err) {
      console.error(err);
      alert("Failed to delete video");
    }
  };

  const handleChat = async () => {
    if (!chatQuery || !selectedVideo) return;
    
    const userMsg = chatQuery;
    setChatQuery('');
    setChatHistory(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsChatting(true);

    try {
      const res = await chatWithVideo(selectedVideo.youtube_id, userMsg, searchMode);
      setChatHistory(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: res.answer, 
          chunks: res.relevant_chunks 
        }
      ]);
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, { role: 'assistant', content: 'Error communicating with server.' }]);
    }
    setIsChatting(false);
  };

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem' }}>
          <Video color="#ef4444" />
          Video Library
        </h2>
        
        <div style={{ display: 'flex', gap: '8px' }}>
          <input 
            type="text" 
            placeholder="Paste YouTube URL..." 
            className="input-glass"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
          />
          <button className="btn-primary" onClick={handleIngest} disabled={isLoading} style={{ padding: '10px' }}>
            <Plus size={20} />
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {videos.map(v => (
            <div 
              key={v.id} 
              style={{ 
                padding: '12px', 
                background: selectedVideo?.id === v.id ? 'rgba(59, 130, 246, 0.2)' : 'rgba(0,0,0,0.2)',
                border: selectedVideo?.id === v.id ? '1px solid var(--accent)' : '1px solid transparent',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
              onClick={() => { setSelectedVideo(v); setChatHistory([]); }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '4px', flex: 1, paddingRight: '8px' }}>{v.title}</h4>
                <button 
                  onClick={(e) => handleDelete(e, v)}
                  style={{ background: 'transparent', border: 'none', color: 'var(--danger)', cursor: 'pointer', padding: '4px' }}
                  title="Delete Video"
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{v.youtube_id}</p>
              {!v.processed && (
                <div style={{ marginTop: '10px' }}>
                  <p style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--accent)', fontStyle: 'italic', marginBottom: '6px' }}>
                    <Loader2 size={12} className="animate-spin" />
                    Status: {v.status}
                  </p>
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleTranscribe(v); }}
                    className="btn-primary" 
                    style={{ fontSize: '0.8rem', padding: '6px 10px', width: '100%' }}
                  >
                    Process Video
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* MAIN VIDEO PANEL */}
      <div className="glass-panel animate-fade-in" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem' }}>
          <PlaySquare color="#3b82f6" />
          Player
        </h2>
        
        <div style={{ 
          width: '100%', 
          aspectRatio: '16/9', 
          background: 'rgba(0,0,0,0.5)', 
          borderRadius: '12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          overflow: 'hidden'
        }}>
          {selectedVideo ? (
            <iframe 
              width="100%" 
              height="100%" 
              src={`https://www.youtube.com/embed/${selectedVideo.youtube_id}`} 
              title="YouTube video player" 
              frameBorder="0" 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowFullScreen
            ></iframe>
          ) : (
            "Select a video from the library to start."
          )}
        </div>

        <div>
          <h3 style={{ fontSize: '1.5rem', marginBottom: '8px' }}>
            {selectedVideo ? selectedVideo.title : "No Video Selected"}
          </h3>
        </div>
      </div>

      {/* CHAT PANEL */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.2rem' }}>
          <MessageSquare color="#10b981" />
          Multimodal Chat
        </h2>

        <div style={{ flex: 1, overflowY: 'auto', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
           {chatHistory.length === 0 ? (
             <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center', marginTop: '20px' }}>
              Ask questions about the video!
            </p>
           ) : (
             chatHistory.map((msg, i) => (
               <div key={i} style={{ alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%' }}>
                 <div style={{ 
                   background: msg.role === 'user' ? 'var(--accent)' : 'rgba(255,255,255,0.1)',
                   padding: '12px',
                   borderRadius: '12px',
                   borderBottomRightRadius: msg.role === 'user' ? '0' : '12px',
                   borderBottomLeftRadius: msg.role === 'assistant' ? '0' : '12px',
                 }}>
                   {msg.role === 'assistant' ? (
                     <div className="markdown-body">
                       <ReactMarkdown>{msg.content}</ReactMarkdown>
                     </div>
                   ) : (
                     msg.content
                   )}
                 </div>
                 
                 {/* Render Visual Chunk Sources */}
                 {msg.chunks && msg.chunks.filter((c: any) => c.text.includes('[VISUAL]')).length > 0 && (
                   <div style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                     {msg.chunks.filter((c: any) => c.text.includes('[VISUAL]')).map((c: any, idx: number) => (
                       <span key={idx} style={{ fontSize: '0.75rem', background: 'rgba(16, 185, 129, 0.2)', color: '#10b981', padding: '4px 8px', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                         <ImageIcon size={12} /> Frame {c.start}s
                       </span>
                     ))}
                   </div>
                 )}
               </div>
             ))
           )}
           {isChatting && <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>AI is thinking...</p>}
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <select 
            className="input-glass" 
            style={{ width: 'auto', padding: '10px' }}
            value={searchMode}
            onChange={e => setSearchMode(e.target.value)}
          >
            <option value="both">All Context</option>
            <option value="audio">Audio Only</option>
            <option value="visual">Visual Only</option>
          </select>
          <input 
            type="text" 
            placeholder="Ask a question..." 
            className="input-glass"
            style={{ flex: 1 }}
            value={chatQuery}
            onChange={e => setChatQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleChat()}
          />
          <button className="btn-primary" onClick={handleChat} disabled={isChatting} style={{ padding: '10px', background: 'var(--success)' }}>
            <Search size={20} />
          </button>
        </div>
      </div>

    </div>
  );
}

export default App;
