import sys
from audio_recorder import *
from pathlib import Path
import time
import threading
from tkinter import PhotoImage

sys.path.insert(0, '/Users/reginaceballos/Documents/MIT/2024-02 - Spring/6.8510 Intelligent Multimodal Interfaces/Final Project/emma/')

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/reginaceballos/Documents/MIT/2024-02 - Spring/6.8510 Intelligent Multimodal Interfaces/Final Project/emma/build/assets/frames")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def start(file_suffix = ''):
    global running
    if file_suffix != '':
        file_suffix = '_' + file_suffix

    running = None
    running = rec.open('answer_questions_speech/recording' + file_suffix + '.wav', 'wb')
    running.start_recording()

def stop():
    running.stop_recording()
    running.close()


def play_audio(file_path):

    chunk = 1024

    with wave.open(file_path, 'rb') as wf:
        # Instantiate PyAudio and initialize PortAudio system resources (1)
        p = pyaudio.PyAudio()

        # Open stream (2)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Play samples from the wave file (3)
        while len(data := wf.readframes(chunk)):  # Requires Python 3.8+ for :=
            stream.write(data)

        # Close stream (4)
        time.sleep(0.5)
        stream.stop_stream()
        stream.close()

        # Release PortAudio system resources (5)
        p.terminate()


rec = Recorder(channels=1)


def move_to_next_question_frame(next_frame, next_first_ask_function = None):
    next_frame.tkraise()

    if next_first_ask_function:
        thread_speech(next_first_ask_function)


def thread_speech(first_ask):
    thread = threading.Thread(target=first_ask)
    thread.start()




