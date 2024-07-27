import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import re
import time
import threading
from google.cloud import texttospeech
import io
from pydub import AudioSegment
from pydub.silence import detect_nonsilent  # Correct import
import numpy as np
from scipy.optimize import curve_fit, root_scalar
import random
import pyaudio
import wave

class SubtitleToSpeechApp:
    def __init__(self, master):
        self.master = master
        master.title("Subtitle-to-Speech Converter")
        master.geometry("900x800")

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.main_frame = ttk.Frame(self.notebook)
        self.debug_frame = ttk.Frame(self.notebook)
        self.graph_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.main_frame, text="Main")
        self.notebook.add(self.debug_frame, text="Debug")
        self.notebook.add(self.graph_frame, text="Graphs")

        self.setup_main_frame()
        self.setup_debug_frame()
        self.setup_graph_frame()

        self.client = texttospeech.TextToSpeechClient()
        self.voices = self.get_available_voices()
        self.voice_combo['values'] = [voice.name for voice in self.voices]

        self.conversion_thread = None
        self.paused = False
        self.stopped = False

    def setup_main_frame(self):
        ttk.Label(self.main_frame, text="Select Voice:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.voice_combo = ttk.Combobox(self.main_frame, values=[])
        self.voice_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.voice_combo.bind("<<ComboboxSelected>>", self.update_voice_preview)

        ttk.Label(self.main_frame, text="Preview Text:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.preview_text = tk.Entry(self.main_frame)
        self.preview_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.preview_text.insert(0, "Hello, this is a voice preview.")

        ttk.Button(self.main_frame, text="Play Preview", command=self.play_preview).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(self.main_frame, text="Speaking Rate:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.rate_var = tk.DoubleVar(value=1.0)
        self.rate_entry = ttk.Entry(self.main_frame, textvariable=self.rate_var)
        self.rate_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        self.rate_slider = tk.Scale(self.main_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.01, variable=self.rate_var)
        self.rate_slider.grid(row=2, column=2, sticky="ew", padx=5, pady=5)

        self.rate_var.trace("w", lambda *args: self.rate_slider.set(self.rate_var.get()))
        self.rate_slider.bind("<Motion>", lambda event: self.rate_var.set(self.rate_slider.get()))

        ttk.Label(self.main_frame, text="Pitch:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.pitch_var = tk.IntVar(value=0)
        self.pitch_entry = ttk.Entry(self.main_frame, textvariable=self.pitch_var)
        self.pitch_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(self.main_frame, text="Select SRT Files", command=self.select_srt_files).grid(row=4, column=0, padx=5, pady=5)
        self.srt_label = ttk.Label(self.main_frame, text="No files selected")
        self.srt_label.grid(row=4, column=1, columnspan=2, sticky="w", padx=5, pady=5)

        ttk.Button(self.main_frame, text="Select Output Directory and Filename", command=self.select_output_file).grid(row=5, column=0, padx=5, pady=5)
        self.output_label = ttk.Label(self.main_frame, text="No directory selected")
        self.output_label.grid(row=5, column=1, columnspan=2, sticky="w", padx=5, pady=5)

        ttk.Label(self.main_frame, text="Min Speaking Rate:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.min_rate_var = tk.DoubleVar(value=0.85)
        self.min_rate_entry = ttk.Entry(self.main_frame, textvariable=self.min_rate_var)
        self.min_rate_entry.grid(row=6, column=1, sticky="ew", padx=5, pady=5)

        self.min_rate_slider = tk.Scale(self.main_frame, from_=0.5, to=1.5, orient=tk.HORIZONTAL, resolution=0.01, variable=self.min_rate_var)
        self.min_rate_slider.grid(row=6, column=2, sticky="ew", padx=5, pady=5)

        self.min_rate_var.trace("w", lambda *args: self.min_rate_slider.set(self.min_rate_var.get()))
        self.min_rate_slider.bind("<Motion>", lambda event: self.min_rate_var.set(self.min_rate_slider.get()))

        ttk.Label(self.main_frame, text="Max Speaking Rate:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.max_rate_var = tk.DoubleVar(value=1.15)
        self.max_rate_entry = ttk.Entry(self.main_frame, textvariable=self.max_rate_var)
        self.max_rate_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=5)

        self.max_rate_slider = tk.Scale(self.main_frame, from_=1.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.01, variable=self.max_rate_var)
        self.max_rate_slider.grid(row=7, column=2, sticky="ew", padx=5, pady=5)

        self.max_rate_var.trace("w", lambda *args: self.max_rate_slider.set(self.max_rate_var.get()))
        self.max_rate_slider.bind("<Motion>", lambda event: self.max_rate_var.set(self.max_rate_slider.get()))

        self.start_button = ttk.Button(self.main_frame, text="Start Conversion", command=self.start_conversion)
        self.start_button.grid(row=8, column=0, padx=5, pady=5)
        self.pause_button = ttk.Button(self.main_frame, text="Pause", command=self.pause_conversion, state=tk.DISABLED)
        self.pause_button.grid(row=8, column=1, padx=5, pady=5)
        self.stop_button = ttk.Button(self.main_frame, text="Stop", command=self.stop_conversion, state=tk.DISABLED)
        self.stop_button.grid(row=8, column=2, padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.grid(row=9, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    def setup_debug_frame(self):
        self.debug_text = tk.Text(self.debug_frame, wrap=tk.WORD, height=20)
        self.debug_text.pack(fill=tk.BOTH, expand=True)

        self.advanced_debug_var = tk.BooleanVar()
        self.advanced_debug_check = ttk.Checkbutton(self.debug_frame, text="Advanced Debugging", variable=self.advanced_debug_var)
        self.advanced_debug_check.pack()

    def setup_graph_frame(self):
        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def get_available_voices(self):
        response = self.client.list_voices()
        return response.voices

    def update_voice_preview(self, event):
        pass

    def play_preview(self):
        voice_name = self.voice_combo.get()
        text = self.preview_text.get()
        rate = self.rate_var.get()
        pitch = self.pitch_var.get()

        voice = next((v for v in self.voices if v.name == voice_name), None)
        if voice:
            language_code = voice.language_codes[0]
            ssml_gender = voice.ssml_gender

            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=ssml_gender
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=rate,
                pitch=pitch
            )

            try:
                response = self.client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )

                audio_data = response.audio_content

                # Play the audio directly in the GUI
                self.play_audio(audio_data)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to play preview: {e}")

    def play_audio(self, audio_data):
        with wave.open(io.BytesIO(audio_data), 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()

    def select_srt_files(self):
        self.srt_files = filedialog.askopenfilenames(filetypes=[("SRT files", "*.srt")])
        self.srt_label.config(text=f"{len(self.srt_files)} file(s) selected")

    def select_output_file(self):
        self.output_file = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 files", "*.mp3")])
        self.output_label.config(text=self.output_file)

    def start_conversion(self):
        if not self.srt_files or not self.output_file:
            messagebox.showerror("Error", "Please select SRT files and output file")
            return

        self.conversion_thread = threading.Thread(target=self.conversion_process)
        self.conversion_thread.start()

        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

    def pause_conversion(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
        else:
            self.pause_button.config(text="Pause")

    def stop_conversion(self):
        self.stopped = True

    def conversion_process(self):
        voice_name = self.voice_combo.get()
        voice = next((v for v in self.voices if v.name == voice_name), None)
        if not voice:
            messagebox.showerror("Error", "Invalid voice selected")
            return

        language_code = voice.language_codes[0]
        ssml_gender = voice.ssml_gender

        min_rate = self.min_rate_var.get()
        max_rate = self.max_rate_var.get()

        # Determine voice speaking characteristics
        cpm_model = self.determine_voice_characteristics(voice_name, language_code, ssml_gender)

        self.debug_text.delete("1.0", tk.END)  # Clear debug text before starting new conversion

        final_audio = AudioSegment.silent(duration=0)

        for srt_file in self.srt_files:
            if self.stopped:
                break

            subtitles = self.parse_srt(srt_file)
            total_subtitles = len(subtitles)

            for i, subtitle in enumerate(subtitles):
                while self.paused:
                    time.sleep(0.1)
                if self.stopped:
                    break

                start_time, end_time, text = subtitle
                subtitle_start_time = self.time_to_seconds(start_time)
                subtitle_end_time = self.time_to_seconds(end_time)
                original_subtitle_duration = subtitle_end_time - subtitle_start_time

                final_audio_duration = len(final_audio) / 1000
                amount_lagging_behind = final_audio_duration - subtitle_start_time
                final_subtitle_duration = max(original_subtitle_duration - amount_lagging_behind, 1)

                cpm_needed = len(text) / (final_subtitle_duration / 60)
                final_cpm = min(max(cpm_needed, cpm_model(min_rate)), cpm_model(max_rate))
                final_rate = self.find_rate_for_cpm(cpm_model, final_cpm)

                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name,
                    ssml_gender=ssml_gender
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=final_rate,
                    pitch=self.pitch_var.get()  # Apply pitch setting to conversion
                )

                try:
                    response = self.client.synthesize_speech(
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )

                    audio = AudioSegment.from_mp3(io.BytesIO(response.audio_content))

                    # Trim silence from the beginning and end of the audio
                    non_silence_ranges = detect_nonsilent(audio, silence_thresh=-40, min_silence_len=100)
                    if non_silence_ranges:
                        start_trim = non_silence_ranges[0][0]
                        end_trim = non_silence_ranges[-1][1]
                        audio = audio[start_trim:end_trim]

                    # Add silence based on punctuation
                    if text.strip().endswith(('.', '。', '！', '!', '?', '？')):
                        audio += AudioSegment.silent(duration=500)
                    elif text.strip().endswith((',', '，', '、', ';', '；')):
                        audio += AudioSegment.silent(duration=250)

                    audio = AudioSegment.silent(duration=70) + audio + AudioSegment.silent(duration=70)

                    pre_processed_duration = len(audio) / 1000
                    if pre_processed_duration < final_subtitle_duration:
                        audio = AudioSegment.silent(duration=(final_subtitle_duration - pre_processed_duration) * 1000) + audio

                    post_processed_duration = len(audio) / 1000

                    final_audio += audio

                    if self.advanced_debug_var.get():
                        debug_info = (
                            f"Subtitle {i+1}/{total_subtitles}:\n"
                            f"Start Time: {subtitle_start_time:.2f}s\n"
                            f"End Time: {subtitle_end_time:.2f}s\n"
                            f"Original Duration: {original_subtitle_duration:.2f}s\n"
                            f"Final Audio Duration: {final_audio_duration:.2f}s\n"
                            f"Amount Lagging Behind: {amount_lagging_behind:.2f}s\n"
                            f"Final Subtitle Duration: {final_subtitle_duration:.2f}s\n"
                            f"CPM Needed: {cpm_needed:.2f}\n"
                            f"Final CPM: {final_cpm:.2f} {'(max)' if final_cpm == cpm_model(max_rate) else '(min)' if final_cpm == cpm_model(min_rate) else ''}\n"
                            f"Pre-Processed Duration: {pre_processed_duration:.2f}s\n"
                            f"Post-Processed Duration: {post_processed_duration:.2f}s\n\n"
                        )
                        self.debug_text.insert(tk.END, debug_info)
                        self.debug_text.see(tk.END)
                        print(debug_info)  # Print debug info to console as well

                    self.progress_bar['value'] = (i + 1) / total_subtitles * 100
                    self.master.update_idletasks()
                except Exception as e:
                    self.debug_text.insert(tk.END, f"Error processing subtitle {i+1}: {e}\n")
                    self.debug_text.see(tk.END)
                    print(f"Error processing subtitle {i+1}: {e}\n")

        if not self.stopped:
            final_audio.export(self.output_file, format="mp3")

        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        messagebox.showinfo("Conversion Complete", "Audio files have been generated successfully.")

    def determine_voice_characteristics(self, voice_name, language_code, ssml_gender):
        test_texts = self.get_random_subtitle_lines(2)
        rates = [0.5, 0.75, 1.0, 1.5, 2.0]
        cpm_data = []

        for rate in rates:
            for text in test_texts:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name,
                    ssml_gender=ssml_gender
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=rate
                )

                try:
                    response = self.client.synthesize_speech(
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )

                    audio = AudioSegment.from_mp3(io.BytesIO(response.audio_content))
                    duration = len(audio) / 1000  # Convert to seconds
                    cpm = len(text) / (duration / 60)
                    cpm_data.append((rate, cpm))
                except Exception as e:
                    self.debug_text.insert(tk.END, f"Error during voice characteristics determination: {e}\n")
                    self.debug_text.see(tk.END)
                    print(f"Error during voice characteristics determination: {e}\n")
                    continue

        # Calculate average CPM for each rate
        avg_cpm_data = {}
        for rate, cpm in cpm_data:
            if rate not in avg_cpm_data:
                avg_cpm_data[rate] = []
            avg_cpm_data[rate].append(cpm)

        avg_cpm_data = [(rate, sum(cpms) / len(cpms)) for rate, cpms in avg_cpm_data.items()]

        # Fit polynomial regression model
        rates, cpms = zip(*avg_cpm_data)
        coeffs = np.polyfit(rates, cpms, 2)
        cpm_model = np.poly1d(coeffs)

        # Plot the graph
        self.ax.clear()
        self.ax.plot(rates, cpms, 'ro', label='Data points')
        x_range = np.linspace(0.5, 2.0, 100)
        self.ax.plot(x_range, cpm_model(x_range), 'b-', label='Fitted curve')
        self.ax.set_xlabel('Speaking Rate')
        self.ax.set_ylabel('Characters per Minute (CPM)')
        self.ax.set_title('Voice Characteristics')
        self.ax.legend()
        self.canvas.draw()

        return cpm_model

    def get_random_subtitle_lines(self, num_lines):
        all_lines = []
        for srt_file in self.srt_files:
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = re.findall(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.+?)(?:\n\n|\Z)', content, re.DOTALL)
                all_lines.extend([line.strip() for line in lines if len(line.strip()) >= 5])

        if len(all_lines) < num_lines:
            return all_lines

        return random.sample(all_lines, num_lines)

    def find_rate_for_cpm(self, cpm_model, target_cpm):
        def objective(x):
            return cpm_model(x) - target_cpm

        result = root_scalar(objective, bracket=[0.5, 2.0], method='brentq')
        return result.root

    def parse_srt(self, srt_file):
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        subtitle_pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?:\n\n|\Z)', re.DOTALL)
        return [(m.group(2), m.group(3), m.group(4).strip()) for m in subtitle_pattern.finditer(content)]

    def time_to_seconds(self, time_str):
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleToSpeechApp(root)
    root.mainloop()
