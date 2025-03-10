import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
import nltk
import os
import time
import tempfile
import io
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
from scipy.io import wavfile

# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Voice Agent Analysis Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'speaker_segments' not in st.session_state:
    st.session_state.speaker_segments = None
if 'speaker1_analysis' not in st.session_state:
    st.session_state.speaker1_analysis = None
if 'speaker2_analysis' not in st.session_state:
    st.session_state.speaker2_analysis = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Function to load models (using simpler models that don't require authentication)
@st.cache_resource
def load_models():
    # Use NLTK's VADER for sentiment analysis (no HF download required)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Use a fallback for transcription - don't attempt to load Hugging Face model
    transcriber = None
    
    # We'll use a simple energy-based approach for diarization
    diarization_pipeline = None
    
    # Optional: Try to load emotion classifier
    emotion_classifier = None
    
    # Indicate in logs that we're using fallback methods
    st.info("Using fallback methods for transcription and emotion detection.")
    
    return transcriber, sentiment_analyzer, diarization_pipeline, emotion_classifier

# Function to process audio file
def process_audio(audio_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load audio file
    try:
        audio, sr = librosa.load(tmp_file_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        os.unlink(tmp_file_path)
        return None, None, None
    
    return audio, sr, tmp_file_path

# Function to separate speakers using a simple energy-based approach
def separate_speakers(audio, sr):
    # Use a simple energy-based approach to detect speech segments
    energy = librosa.feature.rms(y=audio)[0]
    threshold = 0.1 * np.max(energy)
    
    # Find speech segments
    speech_segments = []
    is_speech = False
    start_time = 0
    
    for i, e in enumerate(energy):
        frame_time = librosa.frames_to_time(i, sr=sr)
        
        if e > threshold and not is_speech:
            # Start of speech
            is_speech = True
            start_time = frame_time
        elif e <= threshold and is_speech:
            # End of speech
            is_speech = False
            # Alternate speakers in segments longer than 0.5 second
            if frame_time - start_time > 0.5:
                speech_segments.append({
                    'start': start_time,
                    'end': frame_time,
                    'speaker': 'SPEAKER_01' if len(speech_segments) % 2 == 0 else 'SPEAKER_02'
                })
    
    # Handle case where audio ends during speech
    if is_speech:
        speech_segments.append({
            'start': start_time,
            'end': len(audio) / sr,
            'speaker': 'SPEAKER_01' if len(speech_segments) % 2 == 0 else 'SPEAKER_02'
        })
    
    return speech_segments

# Simple mock transcription function without external dependencies
def simple_transcribe(audio_path, transcriber):
    try:
        # Since we're not using the HF transcriber, provide a placeholder
        # In real-world scenarios, consider using a local model or service API
        st.warning("Using mock transcription. For production, consider using a compatible transcription service.")
        
        # Read audio duration to generate a reasonable mock transcript
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        # Generate a placeholder transcript with length proportional to audio duration
        sample_text = "This is a demonstration of the audio analysis application. "
        sample_text += "The actual transcription is not available due to model loading issues. "
        sample_text += "We're analyzing speaking patterns, sentiment, and other metrics. "
        sample_text += "In a production environment, you would connect to a compatible transcription service. "
        
        # Repeat the sample text to match audio duration
        words_per_minute = 150  # Average speaking rate
        words_needed = int((duration / 60) * words_per_minute)
        words = sample_text.split()
        
        # Generate transcript by repeating words as needed
        full_text = " ".join((words * (1 + words_needed // len(words)))[:words_needed])
        
        return full_text
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return "Transcription error occurred."

# Function to segment transcription into speaker turns
def assign_transcription_to_speakers(full_text, speaker_segments, audio_duration):
    # This is a simplified approach since we don't have accurate word timestamps
    # We'll split the text roughly proportional to the speaker segments
    
    words = full_text.split()
    total_words = len(words)
    
    # Calculate total speaking time per speaker
    speaker1_time = sum([s['end'] - s['start'] for s in speaker_segments if s['speaker'] == 'SPEAKER_01'])
    speaker2_time = sum([s['end'] - s['start'] for s in speaker_segments if s['speaker'] == 'SPEAKER_02'])
    total_speaking_time = speaker1_time + speaker2_time
    
    # Ratio of words per speaker based on speaking time
    speaker1_word_ratio = speaker1_time / total_speaking_time if total_speaking_time > 0 else 0.5
    speaker1_words = int(total_words * speaker1_word_ratio)
    
    # Split text between speakers
    speaker1_text = ' '.join(words[:speaker1_words])
    speaker2_text = ' '.join(words[speaker1_words:])
    
    # Create transcript segments
    transcript = []
    
    # Add speaker 1 segments
    speaker1_segments = [s for s in speaker_segments if s['speaker'] == 'SPEAKER_01']
    if speaker1_segments:
        avg_words_per_segment = len(speaker1_text.split()) / len(speaker1_segments)
        word_index = 0
        speaker1_words = speaker1_text.split()
        
        for segment in speaker1_segments:
            segment_word_count = int(avg_words_per_segment)
            if word_index + segment_word_count > len(speaker1_words):
                segment_word_count = len(speaker1_words) - word_index
            
            if segment_word_count > 0:
                segment_text = ' '.join(speaker1_words[word_index:word_index+segment_word_count])
                transcript.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment_text,
                    'speaker': 'SPEAKER_01'
                })
                word_index += segment_word_count
    
    # Add speaker 2 segments
    speaker2_segments = [s for s in speaker_segments if s['speaker'] == 'SPEAKER_02']
    if speaker2_segments:
        avg_words_per_segment = len(speaker2_text.split()) / len(speaker2_segments)
        word_index = 0
        speaker2_words = speaker2_text.split()
        
        for segment in speaker2_segments:
            segment_word_count = int(avg_words_per_segment)
            if word_index + segment_word_count > len(speaker2_words):
                segment_word_count = len(speaker2_words) - word_index
            
            if segment_word_count > 0:
                segment_text = ' '.join(speaker2_words[word_index:word_index+segment_word_count])
                transcript.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment_text,
                    'speaker': 'SPEAKER_02'
                })
                word_index += segment_word_count
    
    # Sort by start time
    transcript.sort(key=lambda x: x['start'])
    
    return transcript

# Function to analyze speaker using VADER sentiment
def analyze_speaker_vader(transcript_segments, sentiment_analyzer, emotion_classifier=None):
    if not transcript_segments:
        return None
    
    # Combine all text from the speaker
    all_text = ' '.join([segment['text'] for segment in transcript_segments])
    
    # Segment into sentences for more accurate sentiment analysis
    sentences = nltk.sent_tokenize(all_text)
    
    # Calculate speaking time and statistics
    total_duration = sum([segment['end'] - segment['start'] for segment in transcript_segments])
    avg_segment_duration = total_duration / len(transcript_segments) if transcript_segments else 0
    
    # Calculate speech rate (words per minute)
    word_count = len(all_text.split())
    speech_rate = (word_count / total_duration) * 60 if total_duration > 0 else 0
    
    # Calculate pauses between segments
    pauses = []
    for i in range(len(transcript_segments) - 1):
        current_end = transcript_segments[i]['end']
        next_start = transcript_segments[i+1]['start']
        if next_start > current_end:
            pauses.append(next_start - current_end)
    
    avg_pause = sum(pauses) / len(pauses) if pauses else 0
    max_pause = max(pauses) if pauses else 0
    
    # Analyze sentiment for each sentence using VADER
    sentiments = []
    for sentence in sentences:
        if sentence.strip():  # Skip empty sentences
            try:
                vs = sentiment_analyzer.polarity_scores(sentence)
                label = "POSITIVE" if vs['compound'] > 0 else "NEGATIVE"
                sentiments.append({
                    'text': sentence,
                    'label': label,
                    'score': abs(vs['compound'])
                })
            except Exception as e:
                st.warning(f"Error analyzing sentiment for sentence: {e}")
    
    # Calculate overall sentiment
    positive_sentences = [s for s in sentiments if s['label'] == 'POSITIVE']
    negative_sentences = [s for s in sentiments if s['label'] == 'NEGATIVE']
    
    sentiment_ratio = len(positive_sentences) / len(sentiments) if sentiments else 0
    
    # Simplified emotion analysis based on sentiment (no external emotion classifier)
    emotions = []
    emotion_counts = {}
    dominant_emotion = "neutral"
    
    # Simplified emotion analysis based on sentiment
    for sentence in sentences:
        if sentence.strip():
            vs = sentiment_analyzer.polarity_scores(sentence)
            
            # Simple mapping of sentiment to emotion
            if vs['compound'] > 0.5:
                emotion = "joy"
            elif vs['compound'] > 0.2:
                emotion = "neutral"
            elif vs['compound'] > -0.2:
                emotion = "surprise"
            elif vs['compound'] > -0.5:
                emotion = "sadness"
            else:
                emotion = "anger"
            
            emotions.append({
                'text': sentence,
                'emotion': emotion,
                'score': abs(vs['compound'])
            })
            
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
    
    if emotion_counts:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate filler words
    filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally']
    filler_count = sum([all_text.lower().count(word) for word in filler_words])
    filler_ratio = filler_count / word_count if word_count > 0 else 0
    
    # Prepare the analysis results
    analysis = {
        'total_speaking_time': total_duration,
        'word_count': word_count,
        'speech_rate': speech_rate,
        'avg_segment_duration': avg_segment_duration,
        'avg_pause': avg_pause,
        'max_pause': max_pause,
        'sentiment_ratio': sentiment_ratio,
        'positive_sentences': len(positive_sentences),
        'negative_sentences': len(negative_sentences),
        'dominant_emotion': dominant_emotion,
        'emotion_counts': emotion_counts,
        'filler_word_count': filler_count,
        'filler_word_ratio': filler_ratio,
        'detailed_sentiments': sentiments,
        'detailed_emotions': emotions,
        'transcript_segments': transcript_segments
    }
    
    return analysis

# Main Streamlit app
def main():
    st.title("Voice Agent Analysis Dashboard")
    
    # Add notice about the demo mode
    st.info("""
    ⚠️ Running in compatibility mode: This version doesn't rely on external ML models.
    Transcription is simulated, and emotion analysis is simplified.
    """)
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("Upload Audio")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
        
        if uploaded_file is not None:
            st.session_state.audio_file = uploaded_file
            
            if st.button("Process Audio", key="process_button"):
                st.session_state.processing_complete = False
                
                # Display progress information
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load models
                status_text.text("Loading models...")
                progress_bar.progress(10)
                transcriber, sentiment_analyzer, _, emotion_classifier = load_models()
                
                # Process audio
                status_text.text("Processing audio...")
                progress_bar.progress(20)
                audio, sr, tmp_file_path = process_audio(uploaded_file)
                
                if audio is not None:
                    # Separate speakers using energy-based approach
                    status_text.text("Separating speakers...")
                    progress_bar.progress(40)
                    speaker_segments = separate_speakers(audio, sr)
                    
                    if speaker_segments:
                        st.session_state.speaker_segments = speaker_segments
                        
                        # Transcribe audio
                        status_text.text("Transcribing audio...")
                        progress_bar.progress(60)
                        full_text = simple_transcribe(tmp_file_path, transcriber)
                        
                        if full_text:
                            # Assign transcription to speakers
                            status_text.text("Assigning transcription to speakers...")
                            progress_bar.progress(70)
                            audio_duration = len(audio) / sr
                            transcription = assign_transcription_to_speakers(full_text, speaker_segments, audio_duration)
                            
                            st.session_state.transcription = transcription
                            
                            # Analyze speakers
                            status_text.text("Analyzing speakers...")
                            progress_bar.progress(80)
                            
                            # Get unique speaker IDs
                            speakers = list(set([segment['speaker'] for segment in transcription]))
                            
                            if len(speakers) < 2:
                                st.error("Could not identify two distinct speakers in the audio. Please try another file.")
                            else:
                                # For simplicity, use the first two speakers
                                speaker1 = speakers[0]
                                speaker2 = speakers[1]
                                
                                # Get segments for each speaker
                                speaker1_segments = [segment for segment in transcription if segment['speaker'] == speaker1]
                                speaker2_segments = [segment for segment in transcription if segment['speaker'] == speaker2]
                                
                                # Analyze each speaker
                                speaker1_analysis = analyze_speaker_vader(speaker1_segments, sentiment_analyzer, emotion_classifier)
                                speaker2_analysis = analyze_speaker_vader(speaker2_segments, sentiment_analyzer, emotion_classifier)
                                
                                # Save analysis to session state
                                st.session_state.speaker1_analysis = speaker1_analysis
                                st.session_state.speaker2_analysis = speaker2_analysis
                                
                                # Clean up temporary file
                                try:
                                    os.unlink(tmp_file_path)
                                except:
                                    pass
                                
                                # Mark processing as complete
                                st.session_state.processing_complete = True
                                progress_bar.progress(100)
                                status_text.text("Processing complete!")
                        else:
                            st.error("Transcription failed. Please try another audio file.")
                    else:
                        st.error("Speaker separation failed. Please try another audio file.")
                
                # Clean up on error
                if not st.session_state.processing_complete:
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
        
        # Add information about models
        st.subheader("About This App")
        st.markdown("""
        This application analyzes conversations between two speakers in an audio file.
        
        **Features:**
        - Speaker separation using energy-based detection
        - Basic speech transcription simulation
        - Sentiment analysis with NLTK's VADER
        - Speaking patterns analysis
        
        **Note:** This version uses simplified approaches that don't rely on external ML models
        to ensure compatibility.
        """)
        
        st.subheader("Troubleshooting")
        st.markdown("""
        If you're experiencing issues:
        
        1. Make sure your Python version is compatible (3.10 recommended)
        2. Check that required libraries are installed correctly
        3. For real transcription, consider installing a local ASR model
        
        For better results with real transcription, you might need to:
        ```
        pip install --upgrade transformers
        huggingface-cli login
        ```
        """)
    
    # Main content area with tabs
    if st.session_state.processing_complete:
        # Create tabs for overall analysis and individual speakers
        tab_overall, tab_speaker1, tab_speaker2 = st.tabs(["Overall Analysis", "Speaker 1", "Speaker 2"])
        
        # Overall Analysis Tab
        with tab_overall:
            st.header("Conversation Analysis")
            
            # Display audio player
            st.audio(st.session_state.audio_file)
            
            # Basic call statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_duration = max([segment['end'] for segment in st.session_state.transcription])
            speaker1_time = st.session_state.speaker1_analysis['total_speaking_time']
            speaker2_time = st.session_state.speaker2_analysis['total_speaking_time']
            
            with col1:
                st.metric("Total Call Duration", f"{total_duration:.2f}s")
            
            with col2:
                st.metric("Speaker 1 Talk Time", f"{speaker1_time:.2f}s")
            
            with col3:
                st.metric("Speaker 2 Talk Time", f"{speaker2_time:.2f}s")
            
            with col4:
                silence_time = total_duration - speaker1_time - speaker2_time
                st.metric("Silence Time", f"{silence_time:.2f}s")
            
            # Talk time visualization
            st.subheader("Talk Time Distribution")
            talk_time_data = pd.DataFrame({
                'Speaker': ['Speaker 1', 'Speaker 2', 'Silence'],
                'Time (s)': [speaker1_time, speaker2_time, silence_time]
            })
            
            fig_talk_time = px.pie(talk_time_data, values='Time (s)', names='Speaker',
                                   color_discrete_sequence=['#3366CC', '#DC3912', '#DDDDDD'])
            st.plotly_chart(fig_talk_time, use_container_width=True)
            
            # Show waveform visualization
            st.subheader("Audio Waveform")
            
            # Generate waveform plot
            fig, ax = plt.subplots(figsize=(10, 2))
            audio, sr = librosa.load(io.BytesIO(st.session_state.audio_file.getvalue()), sr=None)
            librosa.display.waveshow(audio, sr=sr, ax=ax)
            ax.set_title("Audio Waveform")
            st.pyplot(fig)
            
            # Display full transcript with timestamps and speakers
            st.subheader("Complete Transcript")
            
            transcript_df = pd.DataFrame([
                {
                    'Time': f"{segment['start']:.2f}s - {segment['end']:.2f}s",
                    'Speaker': 'Speaker 1' if segment['speaker'] == st.session_state.speaker_segments[0]['speaker'] else 'Speaker 2',
                    'Text': segment['text']
                }
                for segment in st.session_state.transcription
            ])
            
            st.dataframe(transcript_df, use_container_width=True)
        
        # Speaker 1 Analysis Tab
        with tab_speaker1:
            display_speaker_analysis(st.session_state.speaker1_analysis, "Speaker 1")
        
        # Speaker 2 Analysis Tab
        with tab_speaker2:
            display_speaker_analysis(st.session_state.speaker2_analysis, "Speaker 2")
    
    else:
        # Display welcome message if no file has been processed
        if st.session_state.audio_file is None:
            st.info("👈 Please upload an audio file and click 'Process Audio' to begin analysis.")
        else:
            st.info("👈 Click 'Process Audio' to analyze the uploaded file.")

# Function to display speaker analysis
def display_speaker_analysis(analysis, speaker_name):
    if not analysis:
        st.warning(f"No analysis data available for {speaker_name}")
        return
    
    st.header(f"{speaker_name} Analysis")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Speaking Time", f"{analysis['total_speaking_time']:.2f}s")
    
    with col2:
        st.metric("Speech Rate", f"{analysis['speech_rate']:.1f} wpm")
    
    with col3:
        sentiment_pct = analysis['sentiment_ratio'] * 100
        st.metric("Positive Sentiment", f"{sentiment_pct:.1f}%")
    
    with col4:
        st.metric("Dominant Emotion", analysis['dominant_emotion'].capitalize())
    
    # More detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Speaking Patterns")
        
        # Speech patterns metrics
        metrics_df = pd.DataFrame({
            'Metric': [
                'Average Segment Duration',
                'Average Pause Between Segments',
                'Longest Pause',
                'Word Count',
                'Filler Word Count',
                'Filler Word Ratio'
            ],
            'Value': [
                f"{analysis['avg_segment_duration']:.2f}s",
                f"{analysis['avg_pause']:.2f}s",
                f"{analysis['max_pause']:.2f}s",
                str(analysis['word_count']),
                str(analysis['filler_word_count']),
                f"{analysis['filler_word_ratio'] * 100:.2f}%"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Emotional Analysis")
        
        # Emotion distribution chart
        if analysis['emotion_counts']:
            emotions_df = pd.DataFrame({
                'Emotion': list(analysis['emotion_counts'].keys()),
                'Count': list(analysis['emotion_counts'].values())
            })
            
            fig_emotions = px.bar(emotions_df, x='Emotion', y='Count',
                                 color='Emotion',
                                 title=f'Emotional Distribution for {speaker_name}')
            st.plotly_chart(fig_emotions, use_container_width=True)
    
    # Sentiment analysis
    st.subheader("Sentiment Analysis")
    
    # Prepare sentiment summary
    total_sentences = len(analysis['detailed_sentiments'])
    positive_count = analysis['positive_sentences']
    negative_count = analysis['negative_sentences']
    
    # Sentiment distribution chart
    sentiment_summary = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Count': [positive_count, negative_count]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sentiment = px.pie(sentiment_summary, names='Sentiment', values='Count',
                              title=f'Sentiment Distribution for {speaker_name}',
                              color_discrete_sequence=['#28a745', '#dc3545'])
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Alert indicators
        st.subheader("Performance Alerts")
        
        alerts = []
        
        # Check for long pauses
        if analysis['max_pause'] > 3.0:
            alerts.append({
                'type': 'warning',
                'message': f'Long pause detected ({analysis["max_pause"]:.2f}s)'
            })
        
        # Check for high use of filler words
        if analysis['filler_word_ratio'] > 0.05:
            alerts.append({
                'type': 'warning',
                'message': f'High use of filler words ({analysis["filler_word_ratio"] * 100:.1f}%)'
            })
        
        # Check for predominantly negative sentiment
        if analysis['sentiment_ratio'] < 0.4:
            alerts.append({
                'type': 'error',
                'message': 'Predominantly negative sentiment'
            })
        
        # Check for speaking too fast
        if analysis['speech_rate'] > 170:
            alerts.append({
                'type': 'info',
                'message': f'Speaking rate is high ({analysis["speech_rate"]:.1f} wpm)'
            })
        
        # Display alerts
        if alerts:
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(alert['message'])
                elif alert['type'] == 'warning':
                    st.warning(alert['message'])
                else:
                    st.info(alert['message'])
        else:
            st.success("No performance issues detected")
    
    # Transcript for this speaker
    st.subheader(f"{speaker_name} Transcript")
    
    transcript_df = pd.DataFrame([
        {
            'Time': f"{segment['start']:.2f}s - {segment['end']:.2f}s",
            'Text': segment['text']
        }
        for segment in analysis['transcript_segments']
    ])
    
    st.dataframe(transcript_df, use_container_width=True)
    
    # Performance recommendations
    st.subheader("Performance Recommendations")
    
    recommendations = []
    
    # Generate recommendations based on analysis
    if analysis['sentiment_ratio'] < 0.5:
        recommendations.append("Work on using more positive language and tone when interacting with customers.")
    
    if analysis['filler_word_ratio'] > 0.03:
        recommendations.append("Reduce use of filler words (um, uh, like, etc.) to sound more confident and clear.")
    
    if analysis['speech_rate'] > 160:
        recommendations.append("Consider slowing down speech rate for better clarity and comprehension.")
    
    if analysis['speech_rate'] < 120:
        recommendations.append("Consider increasing speech rate to maintain engagement.")
    
    if analysis['avg_pause'] > 1.5:
        recommendations.append("Reduce pauses between segments to maintain conversation flow.")
    
    if 'angry' in analysis['emotion_counts'] or 'sadness' in analysis['emotion_counts']:
        recommendations.append("Work on maintaining a more positive emotional tone.")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")
    else:
        st.success("No specific recommendations at this time. Good job!")

if __name__ == "__main__":
    main()