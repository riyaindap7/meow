import os
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class SpeechService:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.base_url = "https://api.elevenlabs.io/v1"
    
    async def transcribe_audio(self, audio_data: bytes, filename: str = "audio.webm", language_code: str = "en") -> dict:
        """
        Transcribe audio using ElevenLabs Speech-to-Text API
        
        Args:
            audio_data: Raw audio bytes
            filename: Original filename with extension
            language_code: Language code (e.g., 'en', 'hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa')
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")
        
        url = f"{self.base_url}/speech-to-text"
        
        headers = {
            "xi-api-key": self.api_key,
        }
        
        extension = filename.split('.')[-1].lower()
        content_type_map = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'webm': 'audio/webm',
            'm4a': 'audio/mp4',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac',
        }
        content_type = content_type_map.get(extension, 'audio/webm')
        
        files = {
            "file": (filename, audio_data, content_type),
        }
        
        data = {
            "model_id": "scribe_v1",
            "language_code": language_code,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs STT error: {response.status_code} - {response.text}")
            
            result = response.json()
            return {"text": result.get("text", "")}


_speech_service: Optional[SpeechService] = None

def get_speech_service() -> SpeechService:
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service