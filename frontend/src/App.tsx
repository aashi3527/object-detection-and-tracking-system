import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, 
  Upload, 
  Play, 
  Square, 
  Settings, 
  Cpu, 
  TrendingUp, 
  Eye
} from 'lucide-react';

// Default list of COCO classes used by YOLOv8, in case API query fails initially
const DEFAULT_COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

interface TrackMetadata {
  track_id: number;
  class_name: string;
  bbox: number[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'webcam' | 'video'>('webcam');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Real-time tracking data & statistics
  const [fps, setFps] = useState<number>(0);
  const [activeTracks, setActiveTracks] = useState<TrackMetadata[]>([]);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [cumulativeIds, setCumulativeIds] = useState<Set<number>>(new Set());
  
  // Tracking settings
  const [confidence, setConfidence] = useState<number>(0.35);
  const [availableClasses, setAvailableClasses] = useState<string[]>(DEFAULT_COCO_CLASSES);
  const [selectedClasses, setSelectedClasses] = useState<string[]>(['person', 'car', 'bicycle', 'motorcycle']);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoSrc, setVideoSrc] = useState<string>('');
  const [showTrails, setShowTrails] = useState<boolean>(true);

  // Refs for element access & streaming state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopRef = useRef<any>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const totalCountRef = useRef<Set<number>>(new Set());
  const isPlayingRef = useRef<boolean>(false);

  // Fetch available classes from FastAPI backend on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/classes')
      .then(res => res.json())
      .then(data => {
        if (data.classes) {
          // The backend returns model.names dictionary: { 0: 'person', 1: 'bicycle', ... }
          const classList = Object.values(data.classes) as string[];
          setAvailableClasses(classList);
        }
      })
      .catch(err => {
        console.warn("Backend API not reachable. Using fallback COCO classes.", err);
      });
  }, []);

  // Handle cleanup of webcam streams and intervals when tab changes
  useEffect(() => {
    stopTracking();
  }, [activeTab]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopTracking();
    };
  }, []);

  // Toggle class selection for filtering
  const handleClassToggle = (className: string) => {
    setSelectedClasses(prev => {
      if (prev.includes(className)) {
        return prev.filter(c => c !== className);
      } else {
        return [...prev, className];
      }
    });
  };

  const handleSelectAllClasses = () => {
    setSelectedClasses([]); // Empty means detect all classes
  };

  const handleSelectNoneClasses = () => {
    setSelectedClasses(['person']); // Default to person only
  };

  // Video File Selection
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      stopTracking();
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setCumulativeIds(new Set());
      totalCountRef.current = new Set();
      setActiveTracks([]);
      setClassCounts({});
    }
  };

  // Initialize tracking
  const startTracking = () => {
    if (isPlaying) return;

    setWsStatus('connecting');
    
    // Connect to WebSocket server
    const ws = new WebSocket('ws://localhost:8000/ws/track');
    wsRef.current = ws;

    ws.onopen = async () => {
      setWsStatus('connected');
      setIsPlaying(true);
      isPlayingRef.current = true;
      totalCountRef.current = new Set();
      setCumulativeIds(new Set());

      if (activeTab === 'webcam') {
        try {
          // Request webcam access
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 360, frameRate: 20 }
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
          // Start the paced loop by sending the first frame
          sendNextFrame();
        } catch (err) {
          console.error("Error accessing webcam:", err);
          alert("Could not access webcam. Please verify permissions.");
          stopTracking();
        }
      } else {
        // Tab is video file tracking
        if (videoRef.current && videoSrc) {
          // Keep the video paused so it doesn't auto-advance in the background
          videoRef.current.pause();
          videoRef.current.currentTime = 0;
          // Start the paced loop by sending the first frame
          sendNextFrame();
        } else {
          alert("Please upload a video file first!");
          stopTracking();
        }
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error("Backend track error:", data.error);
        return;
      }

      // Calculate real FPS (rolling average)
      const now = performance.now();
      if (lastFrameTimeRef.current > 0) {
        const frameFps = 1000 / (now - lastFrameTimeRef.current);
        setFps(prev => Math.round(prev * 0.85 + frameFps * 0.15));
      }
      lastFrameTimeRef.current = now;

      // Update track lists & counters
      if (data.tracks) {
        setActiveTracks(data.tracks);
        
        // Add new tracked IDs to cumulative set
        const ids = new Set(totalCountRef.current);
        data.tracks.forEach((t: TrackMetadata) => ids.add(t.track_id));
        totalCountRef.current = ids;
        setCumulativeIds(ids);
      }
      if (data.counts) {
        setClassCounts(data.counts);
      }

      // Render the annotated frame received from the server
      if (data.image) {
        const img = new Image();
        img.src = data.image;
        img.onload = () => {
          const canvas = outputCanvasRef.current;
          const ctx = canvas?.getContext('2d');
          if (canvas && ctx) {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          }
        };
      }

      // Paced frame request: capture and send the next frame only after receiving the previous result
      if (isPlayingRef.current) {
        if (activeTab === 'video' && videoRef.current) {
          const video = videoRef.current;
          // Advance the video by exactly 1 frame (assuming 25 FPS video, so 40ms step)
          const nextTime = video.currentTime + (1 / 25);
          if (nextTime < video.duration) {
            // Listen for the seeked event *before* starting the seek operation
            // to prevent race conditions where the seek completes synchronously.
            const handleSeeked = () => {
              video.removeEventListener('seeked', handleSeeked);
              if (isPlayingRef.current) {
                sendNextFrame();
              }
            };
            video.addEventListener('seeked', handleSeeked);
            video.currentTime = nextTime;
          } else {
            // Reached the end of the video file
            stopTracking();
          }
        } else {
          // Webcam: capture next live frame after a tiny timeout to keep the UI responsive
          loopRef.current = setTimeout(sendNextFrame, 10);
        }
      }
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
      stopTracking();
    };

    ws.onerror = (err) => {
      console.error("WebSocket connection error:", err);
      stopTracking();
    };
  };

  // Stop tracking and clean up resources
  const stopTracking = () => {
    setIsPlaying(false);
    isPlayingRef.current = false;
    setWsStatus('disconnected');
    setFps(0);

    // Stop frame loop timer
    if (loopRef.current) {
      clearTimeout(loopRef.current);
      loopRef.current = null;
    }

    // Stop WebSocket connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Stop webcam camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Pause video element
    if (videoRef.current) {
      videoRef.current.pause();
      if (activeTab === 'webcam') {
        videoRef.current.srcObject = null;
      }
    }

    lastFrameTimeRef.current = 0;
  };

  // Frame Capture & WebSocket Send loop (Paced Request-Response Flow)
  const sendNextFrame = () => {
    if (!isPlayingRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const video = videoRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    const outputCanvas = outputCanvasRef.current;

    if (!video || !hiddenCanvas || !outputCanvas) {
      return;
    }

    // Wait if video element has not initialized metadata or dimensions yet
    if (video.videoWidth === 0 || video.readyState < 2) {
      loopRef.current = setTimeout(sendNextFrame, 100);
      return;
    }

    // 1. Downscale the frame in the browser to keep payload size small (~35KB instead of ~1MB)
    // This maintains the original aspect ratio but scales width to a maximum of 640px.
    const targetWidth = 640;
    const aspectRatio = video.videoHeight / video.videoWidth;
    const targetHeight = Math.round(targetWidth * aspectRatio);

    if (hiddenCanvas.width !== targetWidth) {
      hiddenCanvas.width = targetWidth;
      hiddenCanvas.height = targetHeight;
    }

    if (outputCanvas.width !== targetWidth) {
      outputCanvas.width = targetWidth;
      outputCanvas.height = targetHeight;
    }

    const ctx = hiddenCanvas.getContext('2d');
    if (ctx) {
      // Draw frame onto the hidden downscaled canvas
      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
      
      // Export frame to JPEG string (0.6 quality is optimal for detection accuracy and speed)
      const jpegBase64 = hiddenCanvas.toDataURL('image/jpeg', 0.60);

      // Pack payload with settings
      const payload = {
        image: jpegBase64,
        settings: {
          confidence: confidence,
          classes: selectedClasses.length > 0 ? selectedClasses : null,
          show_trails: showTrails
        }
      };

      // Send to backend
      try {
        wsRef.current.send(JSON.stringify(payload));
      } catch (e) {
        console.error("Error sending frame over WebSocket:", e);
      }
    }
  };

  return (
    <div className="app-container">
      {/* Header Bar */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">
            <Cpu size={24} color="#ffffff" />
          </div>
          <div>
            <h1 className="brand-title">SentinelTrack AI</h1>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              YOLOv8 + DeepSORT Real-time Tracking System
            </span>
          </div>
        </div>

        {/* Server Status Indicators */}
        <div className="status-badge">
          <span className={`status-dot ${wsStatus}`}></span>
          <span style={{ textTransform: 'capitalize' }}>
            {wsStatus === 'connected' ? 'Server Connected' : wsStatus === 'connecting' ? 'Connecting...' : 'Offline'}
          </span>
        </div>
      </header>

      {/* Main Dashboard Grid */}
      <main className="dashboard-grid">
        
        {/* Sidebar Controls */}
        <aside className="sidebar">
          
          {/* Source Tabs */}
          <div className="glass-panel" style={{ padding: '1rem' }}>
            <h2 className="section-title">
              <Camera size={16} /> Input Source
            </h2>
            <div className="tabs-container">
              <button 
                className={`tab-btn ${activeTab === 'webcam' ? 'active' : ''}`}
                onClick={() => setActiveTab('webcam')}
                disabled={isPlaying}
              >
                <Camera size={14} /> Webcam Feed
              </button>
              <button 
                className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
                onClick={() => setActiveTab('video')}
                disabled={isPlaying}
              >
                <Upload size={14} /> Video File
              </button>
            </div>

            {/* Video File Upload Area */}
            {activeTab === 'video' && (
              <div className="control-group">
                <div className="drag-drop-area">
                  <Upload size={28} style={{ color: 'var(--text-muted)' }} />
                  <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>
                    {videoFile ? videoFile.name : 'Upload .mp4 / .avi video'}
                  </span>
                  <input 
                    type="file" 
                    accept="video/*" 
                    onChange={handleVideoUpload}
                    style={{ display: 'none' }}
                    id="file-upload"
                    disabled={isPlaying}
                  />
                  <button 
                    type="button" 
                    className="btn btn-secondary" 
                    style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem', marginTop: '0.5rem' }}
                    onClick={() => document.getElementById('file-upload')?.click()}
                    disabled={isPlaying}
                  >
                    Select File
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Model Config Panel */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Settings size={16} /> Model Settings
            </h2>
            
            {/* Confidence Slider */}
            <div className="control-group">
              <div className="control-label">
                <span>Confidence Threshold</span>
                <span style={{ color: 'var(--secondary)', fontWeight: 600 }}>{Math.round(confidence * 100)}%</span>
              </div>
              <input 
                type="range" 
                min="0.1" 
                max="0.9" 
                step="0.05" 
                value={confidence} 
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="slider-input"
              />
            </div>

            {/* Show Trails Toggle */}
            <div className="control-group" style={{ marginTop: '1rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                <input 
                  type="checkbox" 
                  checked={showTrails} 
                  onChange={(e) => setShowTrails(e.target.checked)}
                  style={{ 
                    cursor: 'pointer',
                    accentColor: 'var(--primary)',
                    width: '15px',
                    height: '15px'
                  }}
                />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Show Motion Trails
                </span>
              </label>
            </div>

            {/* Detection Classes Selection */}
            <div className="control-group" style={{ marginTop: '1.5rem' }}>
              <div className="control-label" style={{ marginBottom: '0.25rem' }}>
                <span>Tracking Filters</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--primary)' }}>
                  {selectedClasses.length === 0 ? 'All Classes' : `${selectedClasses.length} Active`}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <button 
                  onClick={handleSelectAllClasses} 
                  className="btn btn-secondary"
                  style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem', flex: 1 }}
                >
                  Track All
                </button>
                <button 
                  onClick={handleSelectNoneClasses} 
                  className="btn btn-secondary"
                  style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem', flex: 1 }}
                >
                  Person Only
                </button>
              </div>

              <div className="classes-grid">
                {availableClasses.map((cls) => {
                  const isChecked = selectedClasses.includes(cls);
                  return (
                    <label 
                      key={cls} 
                      className={`class-checkbox-label ${isChecked ? 'checked' : ''}`}
                    >
                      <input 
                        type="checkbox" 
                        checked={isChecked}
                        onChange={() => handleClassToggle(cls)}
                      />
                      <span style={{ textTransform: 'capitalize' }}>{cls}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </aside>

        {/* Workspace Display Area */}
        <section className="main-workspace">
          
          {/* Tracking Player Canvas */}
          <div className="video-container">
            {/* Hidden video element that plays feed */}
            <video 
              ref={videoRef}
              src={activeTab === 'video' ? videoSrc : undefined}
              loop={activeTab === 'video'}
              muted
              playsInline
              style={{ display: 'none' }}
            />
            
            {/* Hidden canvas to grab individual raw frames */}
            <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />

            {/* Display Canvas showing processed visual overlays */}
            <canvas 
              ref={outputCanvasRef} 
              className="video-canvas"
              style={{ display: isPlaying ? 'block' : 'none' }}
            />

            {!isPlaying && (
              <div className="video-placeholder">
                <Camera className="video-placeholder-icon" />
                <div>
                  <h3 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>System Ready</h3>
                  <p style={{ fontSize: '0.85rem', maxWidth: '300px' }}>
                    {activeTab === 'webcam' 
                      ? 'Click Start to run real-time object detection and tracking on your camera.' 
                      : 'Upload a video file, then click Start to process tracks.'
                    }
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons & Statistics Bar */}
          <div className="controls-bar">
            <div>
              {isPlaying ? (
                <button className="btn btn-danger" onClick={stopTracking}>
                  <Square size={16} /> Stop Tracking
                </button>
              ) : (
                <button 
                  className="btn btn-primary" 
                  onClick={startTracking}
                  disabled={activeTab === 'video' && !videoFile}
                >
                  <Play size={16} /> Start Tracking
                </button>
              )}
            </div>
            
            {/* Speed & Tracker Latency Stats */}
            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              <div>Processing Rate: <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{fps} FPS</span></div>
            </div>
          </div>

          {/* Tracking Metrics Grid */}
          <div className="stats-grid">
            <div className="glass-panel stat-card">
              <span className="stat-label">Frame Detections</span>
              <span className="stat-value fps">{activeTracks.length}</span>
            </div>
            <div className="glass-panel stat-card">
              <span className="stat-label">Active Targets</span>
              <span className="stat-value active">
                {Object.values(classCounts).reduce((a, b) => a + b, 0)}
              </span>
            </div>
            <div className="glass-panel stat-card">
              <span className="stat-label">Cumulative Tracks</span>
              <span className="stat-value total">{cumulativeIds.size}</span>
            </div>
          </div>

          {/* Live Data Tables split */}
          <div className="data-section-grid">
            
            {/* Active Class Counts table */}
            <div className="glass-panel">
              <h3 className="section-title" style={{ borderLeftColor: 'var(--secondary)' }}>
                <TrendingUp size={16} /> Tracked Counts
              </h3>
              <div className="tracking-table-container">
                <table className="tracking-table">
                  <thead>
                    <tr>
                      <th>Class Label</th>
                      <th>Current Active Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(classCounts).map(([label, count]) => (
                      <tr key={label}>
                        <td style={{ textTransform: 'capitalize', fontWeight: 500 }}>{label}</td>
                        <td>
                          <span className="badge badge-primary">{count}</span>
                        </td>
                      </tr>
                    ))}
                    {Object.keys(classCounts).length === 0 && (
                      <tr>
                        <td colSpan={2} style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '1rem' }}>
                          No targets currently tracked
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Active Targets list */}
            <div className="glass-panel">
              <h3 className="section-title" style={{ borderLeftColor: 'var(--accent)' }}>
                <Eye size={16} /> Active Targets Details
              </h3>
              <div className="tracking-table-container">
                <table className="tracking-table">
                  <thead>
                    <tr>
                      <th>Track ID</th>
                      <th>Category</th>
                      <th>BBox Coordinates</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeTracks.map((t) => (
                      <tr key={t.track_id}>
                        <td style={{ fontWeight: 600, color: 'var(--primary)' }}>#{t.track_id}</td>
                        <td style={{ textTransform: 'capitalize' }}>{t.class_name}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {`[${t.bbox.join(', ')}]`}
                        </td>
                      </tr>
                    ))}
                    {activeTracks.length === 0 && (
                      <tr>
                        <td colSpan={3} style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '1rem' }}>
                          Waiting for tracking inputs...
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        </section>

      </main>
    </div>
  );
}
