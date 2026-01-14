

# ============================================================================
# FILE 6: inference.py
# ============================================================================
"""
Real-time inference for wake word detection
Usage: python inference.py --config config.json --checkpoint checkpoints/best_model.pt --audio test.wav
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
from queue import Queue
import threading
import time
from typing import Optional, Callable
import logging

from config import Config
from model import create_wake_word_model
from utils.manage_audio import AudioPreprocessor

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Real-time wake word detection"""
    
    def __init__(
        self,
        model: nn.Module,
        preprocessor: AudioPreprocessor,
        config: Config,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.config = config
        self.device = device
        self.threshold = threshold
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio file for inference"""
        # Compute MFCCs
        mfccs = self.preprocessor.compute_mfccs(audio_path)
        audio_tensor = torch.FloatTensor(mfccs)
        
        # Pad or truncate
        target_length = self.config.audio.target_length
        current_length = audio_tensor.shape[0]
        
        if current_length < target_length:
            padding = torch.zeros(
                target_length - current_length,
                audio_tensor.shape[1],
                audio_tensor.shape[2]
            )
            audio_tensor = torch.cat([audio_tensor, padding], dim=0)
        elif current_length > target_length:
            audio_tensor = audio_tensor[:target_length]
        
        # Permute to [channels, time, mels]
        audio_tensor = audio_tensor.permute(2, 0, 1)
        
        return audio_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, audio_input) -> Tuple[int, float]:
        """
        Predict wake word
        
        Args:
            audio_input: Either path to audio file or preprocessed tensor
            
        Returns:
            prediction (0 or 1), confidence score
        """
        # Preprocess if needed
        if isinstance(audio_input, str):
            audio_tensor = self.preprocess_audio(audio_input)
        else:
            audio_tensor = audio_input
        
        # Inference
        with torch.no_grad():
            audio_tensor = audio_tensor.to(self.device)
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = probabilities.max(1)
        
        return prediction.item(), confidence.item()
    
    def detect_wake_word(self, audio_path: str) -> bool:
        """Detect if wake word is present"""
        prediction, confidence = self.predict(audio_path)
        is_wake_word = (prediction == 1) and (confidence >= self.threshold)
        
        logger.info(f"Prediction: {'WAKE WORD' if is_wake_word else 'NOT WAKE WORD'}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        return is_wake_word


class StreamingDetector:
    """Real-time streaming audio detection"""
    
    def __init__(
        self,
        detector: WakeWordDetector,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        callback: Optional[Callable] = None
    ):
        self.detector = detector
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.callback = callback
        self.audio_queue = Queue()
        self.is_running = False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_stream(self):
        """Process audio stream"""
        while self.is_running:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                
                # Save chunk temporarily
                temp_path = 'temp_chunk.wav'
                import soundfile as sf
                sf.write(temp_path, audio_chunk, self.sample_rate)
                
                # Detect wake word
                is_wake_word = self.detector.detect_wake_word(temp_path)
                
                # Trigger callback if wake word detected
                if is_wake_word and self.callback:
                    self.callback()
                
                # Clean up
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    def start(self):
        """Start streaming detection"""
        logger.info("Starting real-time wake word detection...")
        logger.info("Press Ctrl+C to stop")
        
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_stream)
        process_thread.start()
        
        # Start audio stream
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        ):
            try:
                while self.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.is_running = False
        
        process_thread.join()
        logger.info("✓ Streaming stopped")
    
    def stop(self):
        """Stop streaming detection"""
        self.is_running = False


def main():
    """Main inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wake word detection inference')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--stream', action='store_true', help='Enable real-time streaming')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_json(args.config)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_wake_word_model(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✓ Model loaded")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sr=config.audio.sample_rate,
        n_dct_filters=config.audio.n_dct_filters,
        n_mels=config.audio.n_mels,
        f_max=config.audio.f_max,
        f_min=config.audio.f_min,
        n_fft=config.audio.n_fft,
        hop_ms=config.audio.hop_ms
    )
    
    # Create detector
    detector = WakeWordDetector(
        model=model,
        preprocessor=preprocessor,
        config=config,
        device=device,
        threshold=args.threshold
    )
    
    # Run inference
    if args.stream:
        # Real-time streaming
        def on_wake_word_detected():
            print("\n🎯 WAKE WORD DETECTED!\n")
        
        streaming_detector = StreamingDetector(
            detector=detector,
            sample_rate=config.audio.sample_rate,
            callback=on_wake_word_detected
        )
        streaming_detector.start()
    
    elif args.audio:
        # Single file inference
        logger.info(f"Processing: {args.audio}")
        is_wake_word = detector.detect_wake_word(args.audio)
        print(f"\n{'✓ WAKE WORD DETECTED' if is_wake_word else '✗ NOT A WAKE WORD'}\n")
    
    else:
        parser.error("Either --audio or --stream must be specified")


if __name__ == "__main__":
    main()
