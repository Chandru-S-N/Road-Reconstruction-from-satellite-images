/* Bilingual voice guidance (English / Tamil) with the browser's Web Speech API.
   Texts come from the server and are built from the actual analysis results;
   this file only plays them. The caption bar always shows the current text. */
"use strict";

class VoiceGuide {
  constructor(onStatus) {
    this.onStatus = onStatus || (() => {});
    this.supported = typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
    this.enabled = this._load("voice.enabled", "true") === "true";
    this.lang = this._load("voice.lang", "en") === "ta" ? "ta" : "en";
    this.auto = this._load("voice.auto", "true") === "true";
    this.last = null;            // {en, ta}
    this.speaking = false;
    this.voices = [];
    this.session = 0;            // increments on every new utterance so stale callbacks are ignored
    if (this.supported) {
      this._refresh();
      window.speechSynthesis.onvoiceschanged = () => { this._refresh(); this._status(); };
    }
    this._status();
  }

  _load(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } }
  _save(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable: settings last for this visit */ } }
  _refresh() { try { this.voices = window.speechSynthesis.getVoices() || []; } catch { this.voices = []; } }

  voiceFor(lang) {
    const want = lang === "ta" ? ["ta-IN", "ta"] : ["en-IN", "en-GB", "en-US", "en"];
    for (const w of want) {
      const v = this.voices.find(x => x.lang && x.lang.toLowerCase().replace("_", "-").startsWith(w.toLowerCase()));
      if (v) return v;
    }
    return null;
  }

  hasVoice(lang) { return !!this.voiceFor(lang); }

  setEnabled(on) { this.enabled = on; this._save("voice.enabled", String(on)); if (!on) this.stop(); this._status(); }
  setLang(lang) { this.lang = lang === "ta" ? "ta" : "en"; this._save("voice.lang", this.lang); this.stop(); this._status(); }
  setAuto(on) { this.auto = on; this._save("voice.auto", String(on)); this._status(); }

  text(obj) { return obj ? (obj[this.lang] || obj.en || "") : ""; }

  /** Called when a stage or action actually happens. Speaks only if voice and automatic voice are on.
      With {queue: true} the text waits until the current sentence group has finished. */
  announce(obj, opts = {}) {
    if (!obj) return;
    if (opts.queue && this.speaking && this.enabled && this.auto) { this.pending = obj; return; }
    this.pending = null;
    this.last = obj;
    this.onStatus({ caption: this.text(obj) });
    if (this.enabled && this.auto) this.speak(obj);
    else this.stop();
  }

  replay() { if (this.last) { this.onStatus({ caption: this.text(this.last) }); if (this.enabled) this.speak(this.last); } }

  stop() {
    this.session += 1;
    this.pending = null;
    if (this.supported) { try { window.speechSynthesis.cancel(); } catch { /* ignore */ } }
    this.speaking = false;
    this._status();
  }

  speak(obj) {
    const keep = this.pending;
    this.stop();
    this.pending = keep;
    const text = this.text(obj);
    if (!text) return;
    if (!this.supported) { this._status("This browser has no speech support; captions only."); return; }
    const voice = this.voiceFor(this.lang);
    if (this.lang === "ta" && !voice) {
      this._status("No Tamil voice is installed: Tamil captions only. Microsoft Edge on Windows 11 includes Tamil voices.");
      return;
    }
    const session = this.session;
    // short sentences: avoids Chrome cutting off long utterances
    const parts = text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
    this.speaking = true;
    this._status();
    parts.forEach((part, i) => {
      const u = new SpeechSynthesisUtterance(part);
      u.lang = this.lang === "ta" ? "ta-IN" : (voice ? voice.lang : "en-US");
      if (voice) u.voice = voice;
      u.rate = this.lang === "ta" ? 0.92 : 1.0;
      u.onend = () => {
        if (session !== this.session || i !== parts.length - 1) return;
        this.speaking = false;
        this._status();
        if (this.pending) { const p = this.pending; this.pending = null; this.announce(p); }
      };
      u.onerror = (e) => {
        if (session !== this.session || e.error === "interrupted" || e.error === "canceled") return;
        this.speaking = false;
        this._status(`Voice playback failed (${e.error || "unknown error"}); captions are still shown.`);
      };
      try { window.speechSynthesis.speak(u); } catch { this._status("Voice playback is not available; captions only."); }
    });
  }

  _status(warning) {
    const langName = this.lang === "ta" ? "தமிழ்" : "English";
    let state = !this.enabled ? "muted" : this.speaking ? "speaking" : this.auto ? "on" : "manual";
    let warn = warning || "";
    if (!warn && this.enabled && this.supported && this.lang === "ta" && this.voices.length && !this.hasVoice("ta")) {
      warn = "No Tamil voice installed: captions only";
    }
    if (!this.supported) warn = warn || "No speech support in this browser: captions only";
    this.onStatus({ status: `Voice: ${state} · ${langName}`, speaking: this.speaking, warning: warn });
  }
}

window.VoiceGuide = VoiceGuide;
