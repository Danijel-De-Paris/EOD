import asyncio
import sys, time
import threading

from faster_whisper import WhisperModel
from EOD_Processor import AppOptions
from EOD_Processor import EODProcessor
from utils.audio_utils import get_valid_input_devices, base64_to_audio
from utils.file_utils import read_json, write_json, write_audio
from utils import print_data
from openai_api import OpenAIAPI

# import logging

# Configure the logging module
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

processor: EODProcessor = None
event_loop: asyncio.AbstractEventLoop = None
thread: threading.Thread = None
openai_api: OpenAIAPI = None

def get_valid_devices():
    devices = get_valid_input_devices()
    return [
        {
            "index": d["index"],
            "name": f"{d['name']} {d['host_api_name']} ({d['max_input_channels']} in)",
        }
        for d in devices
    ]

def get_dropdown_options():
    data_types = ["model_sizes", "compute_types", "languages"]

    dropdown_options = {}
    for data_type in data_types:
        data = read_json("assets", data_type)
        dropdown_options[data_type] = data[data_type]

    return dropdown_options


def get_user_settings():
    data_types = ["app_settings", "model_settings", "transcribe_settings"]
    user_settings = {}

    try:
        data = read_json("settings", "user_settings")
        for data_type in data_types:
            user_settings[data_type] = data[data_type]
    except Exception as e:
        print_data.on_recive_message(str(e))

    return user_settings


def start_transcription(user_settings):
    # logging.info("Starting start_transcription function")
    global processor, event_loop, thread, openai_api
    try:
        (
            filtered_app_settings,
            filtered_model_settings,
            filtered_transcribe_settings,
        ) = extracting_each_setting(user_settings)

        whisper_model = WhisperModel(**filtered_model_settings)
        app_settings = AppOptions(**filtered_app_settings)
        event_loop = asyncio.new_event_loop()
        # logging.info("In Start_Transcription, Load models successfully.")

        if app_settings.use_openai_api:
            openai_api = OpenAIAPI()

        processor = EODProcessor(
            event_loop,
            whisper_model,
            filtered_transcribe_settings,
            app_settings,
            openai_api,
        )
        asyncio.set_event_loop(event_loop)
        thread = threading.Thread(target=event_loop.run_forever, daemon=True)
        thread.start()

        # logging.info("In Start_Transcription, Thread started.")
        asyncio.run_coroutine_threadsafe(processor.start_process(), event_loop)
        # asyncio.run(processor.start_transcription())
    except Exception as e:
        print(str(e))

def stop_transcription():
    global processor, event_loop, thread, websocket_server, openai_api
    if processor is None:
        print("processor is None")
        return
    processor_future = asyncio.run_coroutine_threadsafe(
        processor.stop_transcription(), event_loop
    )
    processor_future.result()

    if thread.is_alive():
        event_loop.call_soon_threadsafe(event_loop.stop)
        thread.join()
    
    event_loop.close()
    processor = None
    event_loop = None
    thread = None
    openai_api = None

    print(print_data.transcription_stoppd())


def audio_transcription(user_settings, base64data):
    global processor, openai_api
    try:
        (
            filtered_app_settings,
            filtered_model_settings,
            filtered_transcribe_settings,
        ) = extracting_each_setting(user_settings)

        whisper_model = WhisperModel(**filtered_model_settings)
        app_settings = AppOptions(**filtered_app_settings)

        if app_settings.use_openai_api:
            openai_api = OpenAIAPI()

        processor = EODProcessor(
            event_loop,
            whisper_model,
            filtered_transcribe_settings,
            app_settings,
            None,
            openai_api,
        )

        audio_data = base64_to_audio(base64data)
        if len(audio_data) > 0:
            write_audio("web", "voice", audio_data)
            processor.batch_transcribe_audio(audio_data)

    except Exception as e:
        print(str(e))

    openai_api = None


def get_filtered_app_settings(settings):
    valid_keys = AppOptions.__annotations__.keys()
    return {k: v for k, v in settings.items() if k in valid_keys}


def get_filtered_model_settings(settings):
    valid_keys = WhisperModel.__init__.__annotations__.keys()
    return {k: v for k, v in settings.items() if k in valid_keys}


def get_filtered_transcribe_settings(settings):
    valid_keys = WhisperModel.transcribe.__annotations__.keys()
    return {k: v for k, v in settings.items() if k in valid_keys}


def extracting_each_setting(user_settings):
    filtered_app_settings = get_filtered_app_settings(user_settings["app_settings"])
    filtered_model_settings = get_filtered_model_settings(
        user_settings["model_settings"]
    )
    filtered_transcribe_settings = get_filtered_transcribe_settings(
        user_settings["transcribe_settings"]
    )

    write_json(
        "settings",
        "user_settings",
        {
            "app_settings": filtered_app_settings,
            "model_settings": filtered_model_settings,
            "transcribe_settings": filtered_transcribe_settings,
        },
    )

    return filtered_app_settings, filtered_model_settings, filtered_transcribe_settings


def on_close(page, sockets):
    print(page, "was closed")

    if processor and processor.transcribing:
        stop_transcription()
    sys.exit()


if __name__ == "__main__":
    user_settings = get_user_settings()
    start_transcription(user_settings)

    while True:
        time.sleep(0.2)