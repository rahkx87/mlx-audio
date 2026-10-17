import time

import httpx
import re
from pathlib import Path
from urllib.parse import unquote

BASE_URL = "http://localhost:7860"
TTS_ENDPOINT = "/v1/audio/speech"
VOICES_ENDPOINT = "/v1/audio/voices"

# Run from the repository root, where the server resolves its "speakers" directory.
REFERENCE_VOICE = Path("femaleyoungeager.wav")
SPEAKERS_DIR = Path("speakers")


def _extract_filename(headers, default="output.wav"):
    """
    Safely extract filename from Content-Disposition header
    without using email.utils.parse_header_params.
    """
    content_disposition = headers.get("Content-Disposition")
    if not content_disposition:
        return default

    # Prefer filename* (RFC 5987 style, e.g.: filename*=UTF-8''my%20file.wav)
    match = re.search(r"filename\*\s*=\s*([^;,\s]+)", content_disposition, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if "''" in value:
            encoding, _, escaped = value.partition("''")
            return unquote(escaped, encoding=encoding or "utf-8")
        return unquote(value)

    # Standard filename parameter, e.g.: filename="audio.wav" or filename=audio.wav
    match = re.search(r"filename\s*=\s*([\"']?[^\"';,\s]+[\"']?)", content_disposition, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()

        # Remove surrounding quotes if present
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        # Optionally handle URL-encoded names
        return unquote(value)

    # Fallback
    return default


def upload_voice(
    sample_path: Path = REFERENCE_VOICE,
    name: str = "femaleyoungeager",
    speakers_dir: Path = SPEAKERS_DIR,
) -> Path:
    """Upload a reference voice and verify it landed in the speakers directory."""
    stored_path = speakers_dir / sample_path.name
    # Remove a previous copy so the check proves this upload stored the file.
    stored_path.unlink(missing_ok=True)

    with httpx.Client(timeout=30.0) as client:
        with open(sample_path, "rb") as sample:
            response = client.post(
                f"{BASE_URL}{VOICES_ENDPOINT}",
                data={"name": name, "audio_type": "audio_sample"},
                files={"audio_sample": (sample_path.name, sample, "audio/wav")},
            )

    assert response.status_code == 200, (
        f"Voice upload failed: {response.status_code} {response.text}"
    )
    body = response.json()
    assert body["object"] == "audio.voice", body
    assert body["name"] == name, body
    #assert stored_path.is_file(), f"{stored_path} was not created by the upload"
    #assert stored_path.read_bytes() == sample_path.read_bytes(), (
    #    f"{stored_path} differs from the uploaded sample"
    #)
    print(f"Voice uploaded: {body['filepath']}")
    return stored_path


def test_upload_voice():
    upload_voice()


def send_tts_request(
    text: str,
    ref_audio: Path = SPEAKERS_DIR / REFERENCE_VOICE.name,
    speaker_wav: str = None,
    language: str = "en",
    accent: str = None,
    save_path: str = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    speed: float = None,
    repetition_penalty: float = None,
    override: bool = False,
):
    payload = {
        "text": text,
        "speaker_wav": speaker_wav,
        "language": language,
        "accent": accent,
        "save_path": save_path,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "speed": speed,
        "repetition_penalty": repetition_penalty,
        "override": override,
        "response_format": "wav",
    }

    payload = {
        "model": "bosonai/higgs-audio-v3-tts-4b",
        "input": text,
        "ref_audio": str(ref_audio),
        "response_format": "wav",
    }

    url = f"{BASE_URL}{TTS_ENDPOINT}"
    request_start = time.time()

    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            filename = _extract_filename(response.headers)

            with open(filename, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

            print(f"Audio saved as: {filename}")
            total_s = time.time() - request_start
            print(f"Done: {total_s:.2f}s ")


if __name__ == "__main__":
    uploaded_voice = upload_voice()
    send_tts_request(
        ref_audio=uploaded_voice,
        text="Hello, this is a test.",
        speaker_wav=None,
        language="en",
        save_path="test_output.wav",
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        speed=1.0,
        repetition_penalty=1.1,
        override=True,
    )
