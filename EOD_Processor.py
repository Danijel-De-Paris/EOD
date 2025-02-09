import asyncio
import functools
import queue
import numpy as np

from typing import NamedTuple
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

from utils.audio_utils import create_audio_stream
from vad import Vad
from utils.file_utils import write_audio
from utils import print_data
from openai_api import OpenAIAPI

import torch
from TTS.api import TTS
from LLM import llm

import asyncio, sounddevice as sd
# import logging

# Configure the logging module
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class AppOptions(NamedTuple):
    audio_device: int
    silence_limit: int = 8
    noise_threshold: int = 5
    non_speech_threshold: float = 0.1
    include_non_speech: bool = False
    create_audio_file: bool = True
    use_openai_api: bool = False


class EODProcessor:
    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        whisper_model: WhisperModel,
        transcribe_settings: dict,
        app_options: AppOptions,
        openai_api: OpenAIAPI,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Init TTS
        self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        self.event_loop = event_loop
        self.whisper_model: WhisperModel = whisper_model
        self.transcribe_settings = transcribe_settings
        self.app_options = app_options
        self.openai_api = openai_api
        self.vad = Vad(app_options.non_speech_threshold)
        self.silence_counter: int = 0
        self.audio_data_list = []
        self.all_audio_data_list = []
        self.audio_queue = queue.Queue()
        self.transcribing = False
        self.stream = None
        self._running = asyncio.Event()
        self._transcribe_task = None

    
    async def play_sound(self, wav):
        sd.play(wav, samplerate=22050)
        await asyncio.sleep(len(wav)/22050+1) 
    
    async def process_TTS(self, input_text):
        """
        General Text to Speech Function using coqui.ai TTS module.

        INPUT
            input_text: string input text for transcribe.
        """
        wav = self.tts.tts(text=input_text)
        
        await self.play_sound(wav)
    
    async def ask_LLM(self, question):
        # llm.gen(model_path, passages_path, index_path, question, online, gpu)

        # Test mode
        await self.process_TTS("Hi I am joseph dingess. Welcome to use EOD Project. I am full stack developer")
        await self.process_TTS("Ask another Question")

    async def process_EOD(self):
        # Ignore parameters that affect performance
        # logging.info("starting process_EOD function")
        transcribe_settings = self.transcribe_settings.copy()
        transcribe_settings["without_timestamps"] = True
        transcribe_settings["word_timestamps"] = False
        # logging.info("starting process_EOD function")

        await self.process_TTS("Hi, how can I help you today? Ask Questions, please.")

        with ThreadPoolExecutor() as executor:
            while self.transcribing:
                try:
                    # Get audio data from queue with a timeout
                    audio_data = await self.event_loop.run_in_executor(
                        executor, functools.partial(self.audio_queue.get, timeout=3.0)
                    )

                    # logging.debug(f"In process_EOD {audio_data}")
                    
                    # Create a partial function for the model's transcribe method
                    func = functools.partial(
                        self.whisper_model.transcribe,
                        audio=audio_data,
                        **transcribe_settings,
                    )

                    # Run the transcribe method in a thread
                    segments, _ = await self.event_loop.run_in_executor(executor, func)

                    temp_text = ""

                    for segment in segments:
                        print_data.display_transcription(segment.text)
                        temp_text += segment.text
                    
                    await self.ask_LLM(temp_text)
                    

                except queue.Empty:
                    # Skip to the next iteration if a timeout occurs
                    continue
                except Exception as e:
                    print_data.print_err_message(str(e))

    def process_audio(self, audio_data: np.ndarray, frames: int, time, status):
        is_speech = self.vad.is_speech(audio_data)
        if is_speech:
            self.silence_counter = 0
            self.audio_data_list.append(audio_data.flatten())
        else:
            self.silence_counter += 1
            if self.app_options.include_non_speech:
                self.audio_data_list.append(audio_data.flatten())

        if not is_speech and self.silence_counter > self.app_options.silence_limit:
            self.silence_counter = 0

            if self.app_options.create_audio_file:
                self.all_audio_data_list.extend(self.audio_data_list)

            if len(self.audio_data_list) > self.app_options.noise_threshold:
                concatenate_audio_data = np.concatenate(self.audio_data_list)
                self.audio_data_list.clear()
                self.audio_queue.put(concatenate_audio_data)
            else:
                # noise clear
                self.audio_data_list.clear()

    def batch_process_EOD(self, audio_data: np.ndarray):
        segment_list = []
        segments, _ = self.whisper_model.transcribe(
            audio=audio_data, **self.transcribe_settings
        )

        for segment in segments:
            word_list = []
            if self.transcribe_settings["word_timestamps"] == True:
                for word in segment.words:
                    word_list.append(
                        {
                            "start": word.start,
                            "end": word.end,
                            "text": word.word,
                        }
                    )
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": word_list,
                }
            )

        print("Cleared")

        if self.openai_api is not None:
            self.text_proofreading(segment_list)
        else:
            print("eel.on_recive_segments(segment_list)")

    def text_proofreading(self, segment_list: list):
        # Use [#] as a separator
        combined_text = "[#]" + "[#]".join(segment["text"] for segment in segment_list)
        result = self.openai_api.text_proofreading(combined_text)
        split_text = result.split("[#]")

        del split_text[0]

        print_data.display_transcription("Before text proofreading.")
        print_data.on_recive_segments(segment_list)
        

        if len(split_text) == len(segment_list):
            for i, segment in enumerate(segment_list):
                segment["text"] = split_text[i]
                segment["words"] = []
            print_data.on_recive_message("proofread success.")
            print_data.display_transcription("After text proofreading.")
            print_data.on_recive_segments(segment_list)
        else:
            print_data.on_recive_message("proofread failure.")
            print_data.on_recive_message(result)

    async def start_process(self):
        # logging.info("Starting start_transcription")
        try:
            self.transcribing = True
            self.stream = create_audio_stream(
                self.app_options.audio_device, self.process_audio
            )
            # logging.info("In start_transcription, created audio stream successfully.")
            self.stream.start()
            self._running.set()
            self._transcribe_task = asyncio.run_coroutine_threadsafe(
                self.process_EOD(), self.event_loop
            )

            print_data.on_recive_message("Transcription started.")
            
            while self._running.is_set():
                await asyncio.sleep(1)
        except Exception as e:
            print_data.on_recive_message(str(e))

    async def stop_transcription(self):
        try:
            self.transcribing = False
            if self._transcribe_task is not None:
                self.event_loop.call_soon_threadsafe(self._transcribe_task.cancel)
                self._transcribe_task = None

            if self.app_options.create_audio_file and len(self.all_audio_data_list) > 0:
                audio_data = np.concatenate(self.all_audio_data_list)
                self.all_audio_data_list.clear()
                write_audio("web", "voice", audio_data)
                self.batch_process_EOD(audio_data)

            if self.stream is not None:
                self._running.clear()
                self.stream.stop()
                self.stream.close()
                self.stream = None
                print_data.on_recive_message("Transcription stopped.")
            else:
                print_data.on_recive_message("No active stream to stop.")
        except Exception as e:
            print_data.on_recive_message(str(e))
