"""
app.py
Full turn-based voice agent using a micro-ADK pattern, Twilio webhooks, Google STT/TTS, and an LLM.
- /voice endpoint: initial TwiML to record the caller.
- /process_recording: Twilio posts the recording URL here; we STT -> LLM -> TTS -> play back, or escalate.
- Simple in-memory conversation memory (for learning). For production, swap to Redis/Firestore.

Usage:
  pip install -r requirements.txt
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
  set other env vars (see .env.example)
  ngrok http 5000        # or deploy to Cloud Run and set APP_BASE_URL
  python app.py
"""

import os
import io
import uuid
import time
import logging
import requests
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioRestClient

# Google clients
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from google.cloud import storage

# LLM
import openai

load_dotenv()
logging.basicConfig(level=logging.INFO)

# -------------------------
# Configuration (env)
# -------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
APP_BASE_URL = os.getenv("APP_BASE_URL")  # must be public for Twilio webhooks
HUMAN_AGENT_NUMBER = os.getenv("HUMAN_AGENT_NUMBER")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "en-US")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_PROVIDER == "openai":
    openai.api_key = OPENAI_API_KEY

# Twilio client
twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Google clients
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client(project=GCP_PROJECT_ID)

# In-memory conversation store: {caller_number: [{"role":"user"/"assistant","text":...}, ...] }
CONVERSATIONS = {}

# DNC set (local file)
DNC_FILE = "do_not_call.txt"
if not os.path.exists(DNC_FILE):
    with open(DNC_FILE, "w") as f:
        f.write("")  # create empty

def load_dnc():
    with open(DNC_FILE, "r") as f:
        return set([line.strip() for line in f if line.strip()])

DNC_SET = load_dnc()


# -------------------------
# Micro-ADK tiny wrapper (learning)
# -------------------------
# This micro-ADK implements minimal concepts: AgentState, tools (callable functions), and an Agent with step().
# If you have a real ADK package, replace this with the real imports and wire tools into it.
class AgentState:
    def __init__(self, caller, call_sid, metadata=None):
        self.caller = caller
        self.call_sid = call_sid
        self.metadata = metadata or {}
        self.last_transcript = ""
        self.response_text = ""
        self.escalate = False

# tools are just plain functions here; decorate if you want.
# Example tools are implemented below (speech_to_text, tts_to_gcs, escalate_to_human).


# -------------------------
# Utility helpers
# -------------------------
def download_twilio_recording(recording_url: str) -> bytes:
    """Download recording from Twilio RecordingUrl (append .wav) with Twilio auth."""
    # Twilio sometimes returns a URL without extension. Fetch both wav and mp3 if needed.
    possible_urls = []
    if recording_url.endswith(".wav") or recording_url.endswith(".mp3"):
        possible_urls.append(recording_url)
    else:
        possible_urls.append(recording_url + ".wav")
        possible_urls.append(recording_url + ".mp3")

    for url in possible_urls:
        try:
            resp = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
            if resp.status_code == 200 and resp.content:
                logging.info(f"Downloaded recording from {url}")
                return resp.content
            else:
                logging.debug(f"Recording download failed for {url} status {resp.status_code}")
        except Exception as e:
            logging.warning(f"Error downloading {url}: {e}")
    raise RuntimeError("Unable to download Twilio recording. Check RecordingUrl and credentials.")


def detect_audio_encoding_from_bytes(data: bytes) -> str:
    """Simple detection by checking header bytes (very basic)."""
    if data[:3] == b'ID3' or data[:2] == b'\xff\xfb' or data[:2] == b'\xff\xf3':
        return "MP3"
    if data[:4] == b'RIFF':
        # WAV file
        return "LINEAR16"
    # fallback assume LINEAR16
    return "LINEAR16"


def speech_to_text_from_bytes(audio_bytes: bytes) -> str:
    """Use Google Cloud Speech-to-Text (sync) to transcribe a short recording."""
    encoding = detect_audio_encoding_from_bytes(audio_bytes)
    if encoding == "MP3":
        enc = speech.RecognitionConfig.AudioEncoding.MP3
    else:
        enc = speech.RecognitionConfig.AudioEncoding.LINEAR16

    config = speech.RecognitionConfig(
        encoding=enc,
        sample_rate_hertz=8000,
        language_code=LANGUAGE_CODE,
        enable_automatic_punctuation=True,
    )

    audio = speech.RecognitionAudio(content=audio_bytes)
    response = speech_client.recognize(config=config, audio=audio)
    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)
    transcript_text = " ".join(transcripts).strip()
    logging.info(f"STT result: {transcript_text}")
    return transcript_text


def synthesize_tts_bytes(text: str) -> bytes:
    """Use Google Cloud Text-to-Speech to create MP3 bytes."""
    if not text:
        text = "I am sorry, I didn't get that."
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=LANGUAGE_CODE,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    logging.info("Synthesized TTS audio bytes (mp3).")
    return response.audio_content


def upload_bytes_to_gcs_and_signed_url(bucket_name: str, blob_name: str, data: bytes, expires_in_seconds: int = 3600) -> str:
    """Upload a bytes object to GCS and return a signed URL (temporary public access)."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="audio/mpeg")
    # create signed url
    url = blob.generate_signed_url(expiration=int(time.time()) + expires_in_seconds, method="GET")
    logging.info(f"Uploaded TTS to gs://{bucket_name}/{blob_name} with signed url.")
    return url


def append_to_conversation(caller: str, role: str, text: str):
    if caller not in CONVERSATIONS:
        CONVERSATIONS[caller] = []
    CONVERSATIONS[caller].append({"role": role, "text": text})
    # keep history short for demo
    CONVERSATIONS[caller] = CONVERSATIONS[caller][-10:]


# Simple escalation detection using keywords + light LLM-based fallback
ESCALATION_KEYWORDS = [
    "human", "real person", "agent", "representative", "talk to someone", "customer service", "someone please"
]

def should_escalate_by_text(text: str) -> bool:
    t = text.lower()
    for kw in ESCALATION_KEYWORDS:
        if kw in t:
            return True
    return False


# -------------------------
# LLM wrapper (OpenAI default; placeholder for Vertex)
# -------------------------
def generate_llm_reply(caller: str, user_transcript: str) -> str:
    """
    Generate a smart reply. This is a minimal system+history prompt. You can replace with LangChain or Vertex logic.
    """
    # Build simple conversational messages
    history = CONVERSATIONS.get(caller, [])
    messages = []
    # system role - instruct the assistant
    messages.append({
        "role": "system",
        "content": (
            "You are a concise, polite voice assistant for an outbound call. Keep replies short (1-2 sentences). "
            "If user asks something that requires escalation (legal complaint, wants to talk to a human), say 'ESCL' in special token."
        )
    })
    # append history
    for m in history:
        role = "assistant" if m["role"] == "assistant" else "user"
        messages.append({"role": role, "content": m["text"]})

    # add current user utterance
    messages.append({"role": "user", "content": user_transcript})

    if LLM_PROVIDER == "openai":
        # Use ChatCompletion (this works with gpt-3.5-turbo / gpt-4 family depending on access)
        try:
            model_name = "gpt-4o-mini"  # replace if not available; fallback handled by API
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )
            reply_text = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.warning(f"OpenAI call failed, trying gpt-3.5-turbo: {e}")
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )
            reply_text = resp["choices"][0]["message"]["content"].strip()
    else:
        # If you want Vertex AI, plug it here (pseudocode / placeholder)
        reply_text = f"(Vertex placeholder) I heard: {user_transcript[:200]}"
    logging.info(f"LLM reply: {reply_text}")
    return reply_text


# -------------------------
# Flask app endpoints / Twilio flow
# -------------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/voice", methods=["POST", "GET"])
def voice_entrypoint():
    """
    Twilio will call this when the outbound call is answered.
    We respond with TwiML to greet & record the caller audio, then Twilio posts recording to /process_recording.
    """
    from_number = request.values.get('From')
    call_sid = request.values.get('CallSid')
    logging.info(f"Incoming Twilio webhook /voice from {from_number} call_sid {call_sid}")

    # Simple TwiML to instruct call: greet and record.
    resp = VoiceResponse()
    resp.say("Hello. This is a test voice assistant. Please say your question after the beep. You may ask for a human anytime.", voice="alice", language=LANGUAGE_CODE)
    # record and when done Twilio will POST to /process_recording
    # limit length to 20 seconds (adjust for your use)
    resp.record(action=f"{APP_BASE_URL}/process_recording", max_length=20, play_beep=True, trim="trim-silence")
    # if no recording or after record Twilio will reach here (hang up)
    resp.say("Goodbye.")
    resp.hangup()
    return Response(str(resp), mimetype="application/xml")


@app.route("/process_recording", methods=["POST"])
def process_recording():
    """
    Handle Twilio's recording callback. Steps:
     - Download recording (Twilio synchonous recording URL)
     - STT -> transcript
     - Update conversation history
     - Decide escalation (keywords or LLM signal)
     - LLM reply -> TTS -> upload to GCS -> return TwiML <Play> to play audio
    """
    recording_url = request.values.get("RecordingUrl")
    from_number = request.values.get("From")
    call_sid = request.values.get("CallSid")
    logging.info(f"/process_recording called for {from_number} recording_url {recording_url}")

    if not recording_url:
        logging.error("No recording URL provided by Twilio.")
        resp = VoiceResponse()
        resp.say("Sorry, we couldn't capture your voice. Goodbye.")
        resp.hangup()
        return Response(str(resp), mimetype="application/xml")

    # Download recording bytes
    try:
        audio_bytes = download_twilio_recording(recording_url)
    except Exception as e:
        logging.error(f"Failed to download recording: {e}")
        resp = VoiceResponse()
        resp.say("Sorry, I couldn't access the recording. Goodbye.")
        resp.hangup()
        return Response(str(resp), mimetype="application/xml")

    # STT
    try:
        transcript_text = speech_to_text_from_bytes(audio_bytes)
    except Exception as e:
        logging.error(f"STT failed: {e}")
        transcript_text = ""

    # instantiate micro-ADK state
    state = AgentState(caller=from_number, call_sid=call_sid)
    state.last_transcript = transcript_text

    # store transcript into short conversation
    append_to_conversation(from_number, "user", transcript_text)

    # Escalation by keyword
    if should_escalate_by_text(transcript_text):
        logging.info("Escalation requested by keyword.")
        resp = VoiceResponse()
        resp.say("Okay, I will connect you to a human agent now.", voice="alice", language=LANGUAGE_CODE)
        # Dial the human number - Twilio will create a call to agent and bridge
        resp.dial(HUMAN_AGENT_NUMBER)
        return Response(str(resp), mimetype="application/xml")

    # Generate LLM reply
    reply_text = generate_llm_reply(from_number, transcript_text)
    append_to_conversation(from_number, "assistant", reply_text)

    # Extra check: if LLM decided to escalate (we used token ESCL in system prompt, check)
    if "ESCL" in reply_text.upper() or "ESCALATE" in reply_text.upper():
        logging.info("Escalation requested by LLM result.")
        resp = VoiceResponse()
        resp.say("I am transferring you to a live agent now.", voice="alice", language=LANGUAGE_CODE)
        resp.dial(HUMAN_AGENT_NUMBER)
        return Response(str(resp), mimetype="application/xml")

    # Synthesize TTS, upload to GCS, return TwiML to play it
    try:
        tts_bytes = synthesize_tts_bytes(reply_text)
        # file name: tts/{call_sid}-{uuid}.mp3
        blob_name = f"tts/{call_sid}-{uuid.uuid4().hex}.mp3"
        tts_url = upload_bytes_to_gcs_and_signed_url(GCS_BUCKET_NAME, blob_name, tts_bytes)
    except Exception as e:
        logging.error(f"TTS/upload failed: {e}")
        resp = VoiceResponse()
        resp.say("Sorry, I am having trouble responding now. I will connect you to an agent.", voice="alice")
        resp.dial(HUMAN_AGENT_NUMBER)
        return Response(str(resp), mimetype="application/xml")

    # Play the audio back to caller
    resp = VoiceResponse()
    resp.play(tts_url)
    # Optionally ask if they want more help and record again in a loop - keep simple here:
    resp.say("If you need anything else, please say it after the beep. Otherwise hang up.", voice="alice")
    resp.record(action=f"{APP_BASE_URL}/process_recording", max_length=20, play_beep=True, trim="trim-silence")
    return Response(str(resp), mimetype="application/xml")


# -------------------------
# CSV caller (simple) - you can run separately or import
# -------------------------
def place_call_via_twilio(to_number: str):
    """
    Place outbound call via Twilio for the given number, instruct Twilio to request the /voice webhook.
    """
    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{APP_BASE_URL}/voice"
        )
        logging.info(f"Placed call {call.sid} -> {to_number}")
        return call.sid
    except Exception as e:
        logging.error(f"Failed to place call to {to_number}: {e}")
        return None

if __name__ == "__main__":
    # Local debug only - use production server for deployment
    app.run(host="0.0.0.0", port=5000, debug=True)
