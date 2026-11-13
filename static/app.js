document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const recordBtn = document.getElementById("record-btn");
  const recordLabel = document.getElementById("record-label");
  const transcriptBox = document.getElementById("transcript-box");
  const clearTranscriptBtn = document.getElementById("clear-transcript-btn");
  const reminderForm = document.getElementById("reminder-form");
  const reminderTitle = document.getElementById("reminder-title");
  const reminderCategory = document.getElementById("reminder-category");
  const reminderDate = document.getElementById("reminder-date");
  const reminderTime = document.getElementById("reminder-time");
  const reminderRepeat = document.getElementById("reminder-repeat");
  const reminderAlert = document.getElementById("reminder-alert");
  const remindersList = document.getElementById("reminders-list");
  const remindersCount = document.getElementById("reminders-count");

  // State
  let isRecording = false;
  let socket = null;
  let audioContext = null;
  let mediaStream = null;
  let scriptProcessor = null;
  let sessionTranscript = "";

  // Render Transcript View
  function updateTranscriptView() {
    transcriptBox.textContent = sessionTranscript.trim();
  }

  // Handle direct manual typing into contenteditable transcript box
  transcriptBox.addEventListener("input", () => {
    sessionTranscript = transcriptBox.innerText;
  });

  // Set default form values (today's date, 09:00)
  function resetFormDefaults() {
    const today = new Date().toISOString().split("T")[0];
    reminderDate.value = today;
    reminderTime.value = "09:00";
    reminderCategory.value = "To-Do";
    reminderRepeat.value = "None";
    reminderAlert.value = "Morning of event";
    reminderTitle.value = "";
  }
  resetFormDefaults();

  // Load existing reminders
  loadReminders();

  // Record / Stop Button Toggle
  recordBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // Clear transcript
  clearTranscriptBtn.addEventListener("click", () => {
    sessionTranscript = "";
    updateTranscriptView();
  });

  // Start Audio Recording & WebSocket
  async function startRecording() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/ws/voice`;
      socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        setupAudioPipeline(mediaStream);
        isRecording = true;
        recordBtn.classList.add("recording");
        recordLabel.textContent = "Stop Recording";
      };

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "transcript") {
            sessionTranscript = msg.accumulated || (sessionTranscript ? sessionTranscript + " " + msg.text : msg.text).trim();
            updateTranscriptView();
          } else if (msg.type === "reminder" && msg.data) {
            populateReminderCard(msg.data);
            if (msg.transcript) {
              sessionTranscript = msg.transcript.trim();
              updateTranscriptView();
            }
            if (!isRecording && socket && socket.readyState === WebSocket.OPEN) {
              socket.close();
            }
          } else if ((msg.type === "empty" || msg.type === "error") && !isRecording) {
            if (socket && socket.readyState === WebSocket.OPEN) {
              socket.close();
            }
          }
        } catch (e) {
          console.error("Error handling message:", e);
        }
      };

      socket.onerror = (err) => {
        console.error("WebSocket error:", err);
        stopRecording();
      };

      socket.onclose = () => {
        if (isRecording) {
          stopRecording();
        }
      };
    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Unable to access microphone. Please grant audio permissions.");
    }
  }

  // Set up Audio Context and PCM streaming
  function setupAudioPipeline(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const sourceNode = audioContext.createMediaStreamSource(stream);
    const bufferSize = 4096;
    scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
      if (!isRecording || !socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }
      const inputBuffer = audioProcessingEvent.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(
        inputBuffer,
        audioContext.sampleRate,
        16000
      );
      const pcm16 = floatTo16BitPCM(downsampled);
      socket.send(pcm16);
    };

    sourceNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
  }

  // Stop Recording
  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordLabel.textContent = "Record Reminder";

    // Flush remaining audio buffer to server
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "flush" }));
    }

    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }

    if (audioContext && audioContext.state !== "closed") {
      audioContext.close();
      audioContext = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
  }

  // Linear Interpolation Downsampling to 16000Hz
  function downsampleBuffer(buffer, inputSampleRate, targetSampleRate = 16000) {
    if (inputSampleRate === targetSampleRate) {
      return buffer;
    }
    const sampleRateRatio = inputSampleRate / targetSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const pos = i * sampleRateRatio;
      const idx = Math.floor(pos);
      const fraction = pos - idx;
      const nextIdx = idx + 1 < buffer.length ? idx + 1 : idx;
      result[i] = buffer[idx] + fraction * (buffer[nextIdx] - buffer[idx]);
    }
    return result;
  }

  // Float32 to 16-bit PCM Converter
  function floatTo16BitPCM(float32Array) {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    let offset = 0;
    for (let i = 0; i < float32Array.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }

  // Populate form with parsed data
  function populateReminderCard(data) {
    if (data.title) reminderTitle.value = data.title;
    if (data.category) reminderCategory.value = data.category;
    if (data.date) reminderDate.value = data.date;
    if (data.time) reminderTime.value = data.time;
    if (data.repeat) reminderRepeat.value = data.repeat;
    if (data.advance_alert) reminderAlert.value = data.advance_alert;
  }

  // Save Reminder Handler
  reminderForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      title: reminderTitle.value.trim() || "Untitled Reminder",
      category: reminderCategory.value,
      date: reminderDate.value,
      time: reminderTime.value,
      repeat: reminderRepeat.value,
      advance_alert: reminderAlert.value,
    };

    try {
      const res = await fetch("/api/reminders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Failed to save reminder");
      }

      resetFormDefaults();
      confirmedTranscript = "";
      interimTranscript = "";
      updateTranscriptView();
      await loadReminders();
    } catch (err) {
      console.error("Save error:", err);
      alert("Could not save reminder. Please try again.");
    }
  });

  // Load Reminders from API
  async function loadReminders() {
    try {
      const res = await fetch("/api/reminders");
      if (!res.ok) return;
      const reminders = await res.json();
      renderReminders(reminders);
    } catch (err) {
      console.error("Load error:", err);
    }
  }

  // Render Saved Reminders List
  function renderReminders(reminders) {
    remindersList.innerHTML = "";
    const count = reminders.length;
    remindersCount.textContent = `${count} ${count === 1 ? "reminder" : "reminders"}`;

    if (count === 0) {
      remindersList.innerHTML = `<div class="empty-state">No saved reminders yet.</div>`;
      return;
    }

    reminders.forEach((r) => {
      const item = document.createElement("div");
      item.className = "reminder-item";

      const repeatText = r.repeat && r.repeat !== "None" ? `Repeat: ${r.repeat}` : "";
      const alertText = r.advance_alert && r.advance_alert !== "None" ? `Alert: ${r.advance_alert}` : "";

      const metaItems = [
        r.category,
        formatDisplayDate(r.date),
        formatDisplayTime(r.time),
        repeatText,
        alertText,
      ].filter(Boolean);

      item.innerHTML = `
        <div class="reminder-content">
          <div class="reminder-item-title">${escapeHtml(r.title)}</div>
          <div class="reminder-meta-row">
            ${metaItems.join('<span class="meta-dot">&bull;</span>')}
          </div>
        </div>
        <button class="delete-btn" data-id="${r.id}" type="button">Delete</button>
      `;

      item.querySelector(".delete-btn").addEventListener("click", async () => {
        await deleteReminder(r.id);
      });

      remindersList.appendChild(item);
    });
  }

  // Delete Reminder
  async function deleteReminder(id) {
    try {
      const res = await fetch(`/api/reminders/${id}`, { method: "DELETE" });
      if (res.ok) {
        await loadReminders();
      }
    } catch (err) {
      console.error("Delete error:", err);
    }
  }

  // Format Helpers
  function formatDisplayDate(dateStr) {
    if (!dateStr) return "";
    try {
      const parts = dateStr.split("-");
      if (parts.length === 3) {
        const d = new Date(parts[0], parts[1] - 1, parts[2]);
        return d.toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
        });
      }
    } catch (e) {}
    return dateStr;
  }

  function formatDisplayTime(timeStr) {
    if (!timeStr) return "";
    try {
      const [h, m] = timeStr.split(":");
      const hour = parseInt(h, 10);
      const ampm = hour >= 12 ? "PM" : "AM";
      const displayHour = hour % 12 || 12;
      return `${displayHour}:${m} ${ampm}`;
    } catch (e) {}
    return timeStr;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});

