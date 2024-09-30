import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch

from .audio import load_audio, SAMPLE_RATE


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        embedding_model_name="pyannote/embedding",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)
        # Load speaker embedding model
        self.embedding_model = PretrainedSpeakerEmbedding(embedding_model_name, device=device, use_auth_token=use_auth_token)


    def __call__(self, audio: Union[str, np.ndarray], num_speakers=None, min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        min_segment_duration = 1  # Minimum duration in seconds

        # Filter out short segments
        diarize_df = diarize_df[diarize_df['end'] - diarize_df['start'] >= min_segment_duration]

        # Now extract embeddings
        diarize_df['embedding'] = diarize_df.apply(
            lambda row: self.extract_embedding(audio_data['waveform'], row['start'], row['end']),
            axis=1
        )

        # Generate speaker embeddings
        speaker_embeddings = (
            diarize_df.groupby('speaker')['embedding']
            .apply(lambda x: np.mean(np.stack([emb for emb in x if np.linalg.norm(emb) > 1e-6]), axis=0)
                    if len(x[x.apply(lambda emb: np.linalg.norm(emb) > 1e-6)]) > 0 else np.zeros_like(x.iloc[0]))
            .reset_index(name='speaker_embedding')
        )

        # Merge the average speaker embeddings with the original diarize_df
        diarize_df = pd.merge(diarize_df, speaker_embeddings, on='speaker', how='left')

        return diarize_df


    def extract_embedding(self, waveform: torch.Tensor, start: float, end: float) -> np.ndarray:
        # Crop the waveform to the segment's start and end times
        start_sample = int(start * SAMPLE_RATE)
        end_sample = int(end * SAMPLE_RATE)
        cropped_waveform = waveform[:, start_sample:end_sample]

        # Check if the cropped waveform is valid
        if cropped_waveform.shape[1] < self.embedding_model.min_num_samples:
            return np.zeros((1, self.embedding_model.dimension))  # Return a zero vector for short segments

        # Pass the cropped waveform to the embedding model
        embedding = self.embedding_model(cropped_waveform)
        return embedding


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker
        
    return transcript_result            


class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
