document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const videoUpload = document.getElementById('video-upload');
    const fileNameDisplay = document.getElementById('file-name-display');
    const promptInput = document.getElementById('prompt-input');
    const submitBtn = document.getElementById('submit-btn');
    
    const initialView = document.getElementById('initial-view');
    const splitView = document.getElementById('split-view');
    
    const originalVideo = document.getElementById('original-video');
    const generatedVideo = document.getElementById('generated-video');
    const generatedVideoContainer = document.getElementById('generated-video-container');
    
    const terminal = document.getElementById('terminal');
    const chatSection = document.getElementById('chat-section');
    const chatContainer = document.getElementById('chat-container');
    const reportSection = document.getElementById('report-section');
    const reportContent = document.getElementById('report-content');

    let selectedVideoFile = null;
    let videoUrl = null;

    if (new URLSearchParams(window.location.search).get('test') === 'true') {
        setTimeout(async () => {
            const res = await fetch('/test_video.mp4');
            const blob = await res.blob();
            const file = new File([blob], 'sample.mp4', { type: 'video/mp4' });
            handleFile(file);
        }, 500);
    }

    // --- File Upload Logic ---
    uploadArea.addEventListener('click', () => videoUpload.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    videoUpload.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    const uploadIdleState = document.getElementById('upload-idle-state');
    const uploadPreviewContainer = document.getElementById('upload-preview-container');
    const uploadVideoPreview = document.getElementById('upload-video-preview');
    const previewFilename = document.getElementById('preview-filename');
    const previewDuration = document.getElementById('preview-duration');
    const previewFilesize = document.getElementById('preview-filesize');
    const changeVideoBtn = document.getElementById('change-video-btn');

    // Prevent uploadArea click from firing when clicking Change Video button
    if (changeVideoBtn) {
        changeVideoBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            videoUpload.click();
        });
    }

    function handleFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please upload a valid video file.');
            return;
        }
        selectedVideoFile = file;
        
        // Create object URL for video
        if (videoUrl) URL.revokeObjectURL(videoUrl);
        videoUrl = URL.createObjectURL(file);
        
        // Update Preview Display
        previewFilename.textContent = file.name;
        previewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
        
        uploadVideoPreview.src = videoUrl;
        uploadVideoPreview.onloadedmetadata = () => {
            const dur = uploadVideoPreview.duration || 0;
            const mins = Math.floor(dur / 60);
            const secs = Math.floor(dur % 60);
            previewDuration.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };
        
        // Toggle view from Idle to Preview state
        uploadIdleState.classList.add('hidden');
        uploadPreviewContainer.classList.remove('hidden');
        
        checkReady();
    }

    // --- Input & Presets Logic ---
    promptInput.addEventListener('input', checkReady);
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !submitBtn.disabled) {
            startProcessing();
        }
    });

    document.querySelectorAll('.preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            promptInput.value = chip.dataset.prompt;
            checkReady();
            promptInput.focus();
        });
    });

    function checkReady() {
        if (selectedVideoFile && promptInput.value.trim().length > 0) {
            submitBtn.disabled = false;
        } else {
            submitBtn.disabled = true;
        }
    }

    submitBtn.addEventListener('click', startProcessing);

    // --- Processing Pipeline ---
    function startProcessing() {
        const promptText = promptInput.value.trim();
        
        // Transition views
        initialView.classList.add('fade-out');
        setTimeout(() => {
            initialView.style.display = 'none';
            splitView.classList.remove('hidden');
            splitView.classList.add('visible');
            
            // Set video source
            originalVideo.src = videoUrl;
            
            // Start pipeline
            runIngestion(promptText);
        }, 300);
    }

    // Helper: Add log line
    function addLog(text, type = '') {
        let isOverwrite = false;
        if (text.startsWith('\r') || text.includes('\r')) {
            isOverwrite = true;
            text = text.replace(/\r/g, '');
        }
        
        const progressRegex = /[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⠋] Ingesting|Progress|%|━━━━━━━━/;
        if (progressRegex.test(text)) {
            isOverwrite = true;
        }

        const cleanText = text.trim();
        if (!cleanText) return;

        const timestamp = new Date().toISOString().split('T')[1].substring(0, 12);
        const logContent = `[${timestamp}] ${cleanText}`;

        if (isOverwrite && terminal.lastElementChild) {
            terminal.lastElementChild.className = `log-line ${type}`;
            terminal.lastElementChild.textContent = logContent;
        } else {
            const line = document.createElement('div');
            line.className = `log-line ${type}`;
            line.textContent = logContent;
            terminal.appendChild(line);
        }
        
        terminal.scrollTop = terminal.scrollHeight;
    }

    // Helper: Add chat message
    function addChat(text, isUser = false) {
        const msg = document.createElement('div');
        msg.className = `chat-msg ${isUser ? 'user' : 'agent'}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        avatar.innerHTML = isUser ? 
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>' : 
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"></path><path d="M12 18a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-2a2 2 0 0 1 2-2z"></path><path d="M4.93 4.93a2 2 0 0 1 2.83 0l1.41 1.41a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0L4.93 7.76a2 2 0 0 1 0-2.83z"></path><path d="M16.24 16.24a2 2 0 0 1 2.83 0l1.41 1.41a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-1.41-1.41a2 2 0 0 1 0-2.83z"></path><path d="M2 12a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2 2 2 0 0 1-2 2H4a2 2 0 0 1-2-2z"></path><path d="M18 12a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2z"></path><path d="M4.93 19.07a2 2 0 0 1 0-2.83l1.41-1.41a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-1.41 1.41a2 2 0 0 1-2.83 0z"></path><path d="M16.24 7.76a2 2 0 0 1 0-2.83l1.41-1.41a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-1.41 1.41a2 2 0 0 1-2.83 0z"></path></svg>';
            
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        
        let escaped = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
        
        let formatted = escaped
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/- (.*?)(<br>|$)/g, '• $1$2')
            .replace(/\n/g, '<br>');
        
        bubble.innerHTML = formatted;
        
        msg.appendChild(avatar);
        msg.appendChild(bubble);
        chatContainer.appendChild(msg);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        const rightPanel = document.querySelector('.right-panel');
        if (rightPanel) rightPanel.scrollTo({ top: rightPanel.scrollHeight, behavior: 'smooth' });
    }

    // Step 1: Ingestion
    function runIngestion(promptText) {
        document.querySelector('#logs-section .spinner').style.display = 'block';
        document.querySelector('#logs-section .section-header span').textContent = 'Ingesting Media...';

        const formData = new FormData();
        formData.append('file', selectedVideoFile);

        fetch('http://localhost:8000/ingest/stream', {
            method: 'POST',
            body: formData
        }).then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            function readChunk() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    const events = buffer.split('\n\n');
                    buffer = events.pop(); // keep partial chunk in buffer
                    
                    for (const ev of events) {
                        if (!ev.startsWith('data: ')) continue;
                        try {
                            const data = JSON.parse(ev.substring(6));
                            console.log("Ingest event:", data);
                            if (data.event === 'hash') {
                                window.currentVideoId = data.video_id;
                                addLog(`Computed video hash: ${data.video_id}`, 'system');
                            } else if (data.event === 'log') {
                                addLog(data.text);
                            } else if (data.event === 'stage_done') {
                                addLog(`Completed stage: ${data.stage}`, 'success');
                            } else if (data.event === 'error') {
                                addLog(`Error: ${data.error}`, 'error');
                                document.querySelector('#logs-section .spinner').style.display = 'none';
                                document.querySelector('#logs-section .section-header span').textContent = 'Ingestion Failed';
                            } else if (data.event === 'done') {
                                document.querySelector('#logs-section .spinner').style.display = 'none';
                                document.querySelector('#logs-section .section-header span').textContent = 'Ingestion Complete';
                                window.currentVideoId = data.video_id;
                                runChat(promptText);
                            }
                        } catch(e) {
                            console.error("Parse error in ingest:", e, ev);
                        }
                    }
                    readChunk();
                }).catch(e => {
                    console.error("Stream read error", e);
                });
            }
            readChunk();
        }).catch(err => {
            addLog(`Failed to start ingestion: ${err.message}`, 'error');
        });
    }

    // Step 2: Chat / Query execution
    function runChat(promptText) {
        chatSection.classList.remove('hidden');
        addChat(promptText, true);
        addLog('--- STARTING EDIT QUERY PIPELINE ---', 'system');

        const formData = new FormData();
        formData.append('prompt', promptText);

        fetch(`http://localhost:8000/edit/${window.currentVideoId}/stream`, {
            method: 'POST',
            body: formData
        }).then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            function readChunk() {
                reader.read().then(({ done, value }) => {
                    if (done) return;
                    buffer += decoder.decode(value, { stream: true });
                    const events = buffer.split('\n\n');
                    buffer = events.pop();
                    
                    for (const ev of events) {
                        if (!ev.startsWith('data: ')) continue;
                        try {
                            const data = JSON.parse(ev.substring(6));
                            console.log("Edit event:", data);
                            if (data.event === 'state_change') {
                                const stateMsg = `Agent state transition: ${data.state}`;
                                addChat(stateMsg, false);
                                addLog(`[Session] State: ${data.state}`, 'system');
                            } else if (data.event === 'log') {
                                addLog(data.text);
                            } else if (data.event === 'report') {
                                renderFinalVideo(data.report, data.version);
                            } else if (data.event === 'error') {
                                addChat(`Error: ${data.error}`, false);
                                addLog(`[Error] ${data.error}`, 'error');
                            }
                        } catch(e) {
                            console.error("Parse error in edit:", e, ev);
                        }
                    }
                    readChunk();
                }).catch(e => {
                    console.error("Stream read error", e);
                });
            }
            readChunk();
        }).catch(err => {
            addChat(`Failed to start edit: ${err.message}`, false);
        });
    }

    // Step 3: Render and Report
    function renderFinalVideo(report, version) {
        let msg = `Video rendering complete! ✨\n\n`;
        msg += `**Verdict:** ${report.critic_verdict_summary}\n`;
        msg += `**Reduction:** ${report.reduction_pct.toFixed(1)}% (Trimmed down to ${report.duration_after_s.toFixed(1)}s from ${report.duration_before_s.toFixed(1)}s)\n`;
        
        if (report.continuity_warnings && report.continuity_warnings.length > 0) {
            msg += `\n**Continuity & Coherence Notes:**\n` + report.continuity_warnings.map(w => `- ${w}`).join('\n');
        }
        
        if (report.unsatisfied_ops && report.unsatisfied_ops.length > 0) {
            msg += `\n**Unsatisfied Constraints:**\n` + report.unsatisfied_ops.map(o => `- ${o}`).join('\n');
        }
        
        addChat(msg, false);
        
        reportSection.classList.remove('hidden');
        
        let removalsHtml = report.removals.map(r => `
            <div class="removal-item" style="margin-bottom: 8px; font-size: 0.9em; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 6px; border-left: 3px solid #60a5fa;">
                <strong style="color: #60a5fa;">${r.start.toFixed(1)}s - ${r.end.toFixed(1)}s</strong>: ${r.reason}
                ${r.evidence_quote ? `<br><span style="color: #9ca3af; font-style: italic; display: inline-block; margin-top: 4px;">"${r.evidence_quote}"</span>` : ''}
            </div>
        `).join('');

        reportContent.innerHTML = `
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-label">Original Duration</div>
                    <div class="stat-value">${report.duration_before_s.toFixed(1)}s</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Trimmed Duration</div>
                    <div class="stat-value">${report.duration_after_s.toFixed(1)}s</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Reduction</div>
                    <div class="stat-value">${report.reduction_pct.toFixed(1)}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Segments Kept</div>
                    <div class="stat-value">${report.clip_count || report.removals.length}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Agent Cost</div>
                    <div class="stat-value">$${report.cost_usd.toFixed(4)}</div>
                </div>
            </div>
            
            <div style="margin-top: 24px;">
                <h4 style="margin-bottom: 12px; color: #e5e7eb; font-size: 1.1em;">✂️ Removals (${report.removals.length})</h4>
                ${removalsHtml || '<div style="color: #9ca3af;">No removals made.</div>'}
            </div>
        `;
        
        generatedVideo.src = `http://localhost:8000/edits/${window.currentVideoId}/${version}/output`;
        generatedVideoContainer.classList.remove('hidden');
        
        // Wait for next frame to trigger CSS transition
        requestAnimationFrame(() => {
            generatedVideoContainer.classList.add('visible');
        });
    }
});
