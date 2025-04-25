import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
import librosa
import librosa.feature
import plotly.express as px
import tempfile
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Set page config and theme
st.set_page_config(
    page_title="Music Recommendation System Based on Acoustic Features",
    page_icon="🎵",
    layout="wide"
)

# Enhanced Professional Dark Theme CSS
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    
    h1, h2, h3, h4 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 500;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }
    
    h1 {
        background: linear-gradient(90deg, #4A90E2, #28D5BD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4A90E2, #28D5BD);
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.2);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    .card {
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        border: 1px solid #333333;
        transition: all 0.4s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        border-color: rgba(74, 144, 226, 0.5);
    }
    
    .highlight {
        color: #28D5BD;
        font-weight: 500;
    }
    
    .pattern-match {
        color: #FF4B91;
        font-weight: 600;
        text-shadow: 0 0 5px rgba(255, 75, 145, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #333333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #4A90E2, #28D5BD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #BBBBBB;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #1A1A1A;
        padding: 0.7rem;
        border-radius: 10px;
        border: 1px solid #333333;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1.2rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4A90E2, #28D5BD);
        color: white;
        border-radius: 6px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    audio {
        width: 100%;
        height: 48px;
        border-radius: 8px;
        margin: 0.7rem 0;
        background: #1A1A1A;
    }
    
    .audio-card {
        padding: 0.5rem 0;
    }
    
    /* Updated styling for select boxes with stronger selectors */
    div[data-baseweb="select"] {
        background-color: #1A1A1A !important;
    }

    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] input {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] * {
        color: #FFFFFF !important;
    }

    /* Style dropdown options */
    div[role="listbox"] {
        background-color: #1A1A1A !important;
        border: 1px solid #4A90E2 !important;
    }

    div[role="option"] {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }

    div[role="option"]:hover {
        background-color: #2A2A2A !important;
    }

    /* Force text color for select box */
    .stSelectbox div[data-baseweb="select"] div {
        color: #FFFFFF !important;
    }

    .stSelectbox div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    /* Additional styling for better visibility */
    .stSelectbox > div {
        border: 1px solid #4A90E2 !important;
    }

    .stSelectbox [data-baseweb="select"] {
        background-color: #1A1A1A !important;
    }

    .stSelectbox [data-baseweb="value-container"] {
        color: #FFFFFF !important;
    }

    .stSelectbox [data-baseweb="value-container"] * {
        color: #FFFFFF !important;
    }
    
    .stSlider {
        padding: 1rem 0 !important;
    }
    
    .stSlider > div {
        height: 6px !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4A90E2, #28D5BD) !important;
        height: 6px !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        border: 2px solid #28D5BD !important;
        box-shadow: 0 0 10px rgba(40, 213, 189, 0.5) !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #0D0D0D !important;
    }
    
    hr {
        border-color: #333333 !important;
        margin: 2rem 0 !important;
    }
    
    /* Custom animations */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(74, 144, 226, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(74, 144, 226, 0); }
        100% { box-shadow: 0 0 0 0 rgba(74, 144, 226, 0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center;'>
            <h1>Music Recommender</h1>
            <p>Discover similar songs based on audio features</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses:
        - Deep Learning (CNN)
        - Audio Feature Analysis
        - Cosine Similarity
        """)
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Search for a song
        2. Select from results
        3. Get recommendations
        4. Analyze similarities
        """)

def display_loading():
    st.markdown("""
    <div class="loading"></div>
    """, unsafe_allow_html=True)

def display_song_info(song_name, df):
    song_data = df[df["Song"] == song_name].iloc[0]
    
    # Handle chords display
    chords_display = "No chord data available"
    if 'Chords' in song_data:
        try:
            if isinstance(song_data['Chords'], str):
                # If chords are stored as string, try to evaluate it
                chords = ast.literal_eval(song_data['Chords'])
            else:
                chords = song_data['Chords']
            
            if isinstance(chords, (list, tuple, set)):
                chords_display = ', '.join(str(chord) for chord in chords)
            else:
                chords_display = str(chords)
        except:
            chords_display = "Error displaying chords"
    
    st.markdown(f"""
    <div class="card">
        <h3>{song_name}</h3>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Tempo: {len(song_data['Tempo'])} features</li>
            <li>Chroma: {len(song_data['Chroma'])} features</li>
            <li>MFCC: {len(song_data['MFCC'])} features</li>
            <li>Chords: {chords_display}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def load_model_and_data():
    """Load data and model with error handling"""
    try:
        # Load scaler
        scaler = joblib.load('feature_scaler.joblib')
        
        # Load data
        df = pd.read_csv("music_features.csv")
        for col in ["Tempo", "Chroma", "MFCC", "Chords"]:
            df[col] = df[col].apply(ast.literal_eval)
            
        # Try to load model, but continue without it if it fails
        try:
            model = tf.keras.models.load_model('music_recommender_model.h5')
            st.success("Model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}. Using similarity-based recommendations only.")
            model = None
            
        return model, scaler, df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def preprocess_features(df, scaler):
    """Preprocess features for model input"""
    # Combine features
    df["Features"] = df.apply(lambda row: row["Tempo"] + row["Chroma"] + row["MFCC"], axis=1)
    X = np.stack(df["Features"].values)
    
    # Ensure X has the correct number of features
    if X.shape[1] != 26:
        st.warning(f"Expected 26 features, got {X.shape[1]}. Truncating/padding to match.")
        # Create a new array with the correct size
        X_adjusted = np.zeros((X.shape[0], 26))
        # Copy the features we have
        X_adjusted[:, :min(X.shape[1], 26)] = X[:, :min(X.shape[1], 26)]
        X = X_adjusted
    
    # Normalize
    X_scaled = scaler.transform(X)
    
    # Pad to 36 for model input
    X_padded = np.zeros((X_scaled.shape[0], 36))
    X_padded[:, :X_scaled.shape[1]] = X_scaled
    
    # Reshape for CNN
    X_cnn = X_padded.reshape(-1, 6, 6, 1)
    
    # Also return flattened features for similarity calculation
    X_flat = X_padded.reshape(X_padded.shape[0], -1)
    
    return X_cnn, X_flat

def get_recommendations(model, X_cnn, X_flat, df, song_index, num_recommendations=5):
    """Get recommendations with fallback to similarity-based approach"""
    try:
        # Calculate similarities using flattened features
        similarities = cosine_similarity([X_flat[song_index]], X_flat)[0]
        top_indices = similarities.argsort()[-(num_recommendations+1):][::-1]
        
        # Get recommendations
        input_song = df.iloc[song_index]["Song"]
        recommendations = []
        
        # If model is available, use it for additional features
        if model is not None:
            try:
                predicted_chords = model.predict(X_cnn)
                # Use predictions to enhance recommendations
                pass
            except Exception as e:
                st.warning(f"Model prediction failed: {str(e)}. Using similarity-based recommendations only.")
        
        # Get recommendations based on similarity
        for idx in top_indices[1:]:  # Skip self
            recommendations.append({
                'song': df.iloc[idx]["Song"],
                'similarity': similarities[idx],
                'features': {
                    'tempo': get_average_tempo(df.iloc[idx]['Tempo']),
                    'chords': len(df.iloc[idx]['Chords'])
                }
            })
        
        return input_song, recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None, []

def get_audio_path(song_name):
    """Get the exact audio file path from compressed_mp3 folder"""
    audio_dir = Path('compressed_mp3')
    audio_path = audio_dir / f"{song_name}.mp3"
    
    # Check if directory exists
    if not audio_dir.exists():
        raise FileNotFoundError(f"Directory '{audio_dir}' not found. Please create the 'compressed_mp3' directory.")
    
    # Check if file exists and is an MP3
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file for '{song_name}' not found in {audio_dir}")
    
    if not audio_path.suffix.lower() == '.mp3':
        raise ValueError(f"File {audio_path} is not an MP3 file")
        
    return str(audio_path)

def display_audio_player(song_name, title_color="#4A90E2"):
    """Display an audio player with song title"""
    try:
        audio_path = get_audio_path(song_name)
        
        st.markdown(f"""
        <div class="audio-card">
            <h4 style='color: {title_color}; margin-bottom: 10px;'>{song_name}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.audio(audio_path)
            
    except FileNotFoundError as e:
        st.markdown(f"""
        <div class="audio-card">
            <h4 style='color: {title_color}; margin-bottom: 10px;'>{song_name}</h4>
            <p style='color: #ff6b6b;'>⚠ {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="audio-card">
            <h4 style='color: {title_color}; margin-bottom: 10px;'>{song_name}</h4>
            <p style='color: #ff6b6b;'>⚠ Error playing audio: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)

def plot_similarity_scores(similarities, songs):
    """Plot similarity scores with enhanced styling"""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(songs))
    
    # Create gradient bars
    colors = plt.cm.viridis(np.linspace(0.2, 1, len(songs)))
    bars = ax.barh(y_pos, similarities, align='center', color=colors)
    
    # Add value labels
    for i, v in enumerate(similarities):
        ax.text(v, i, f' {v:.3f}', color='white', va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(songs)
    ax.invert_yaxis()
    ax.set_xlabel('Similarity Score', color='white')
    ax.set_title('Most Similar Songs', color='white', pad=20)
    
    # Set dark theme
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#1A1A1A')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

def get_unique_chords(df):
    """Get unique chords from the dataset"""
    all_chords = set()
    for chords in df['Chords']:
        try:
            if isinstance(chords, str):
                # If chords are stored as string, try to evaluate it
                chord_list = ast.literal_eval(chords)
            else:
                chord_list = chords
            
            if isinstance(chord_list, (list, tuple, set)):
                all_chords.update(str(chord) for chord in chord_list)
            else:
                all_chords.add(str(chords))
        except:
            continue
    return sorted(list(all_chords))

def get_chord_name(chord_number):
    """Convert chord number to musical note name"""
    chord_names = {
        '0': 'C',
        '1': 'C#/Db',
        '2': 'D',
        '3': 'D#/Eb',
        '4': 'E',
        '5': 'F',
        '6': 'F#/Gb',
        '7': 'G',
        '8': 'G#/Ab',
        '9': 'A',
        '10': 'A#/Bb',
        '11': 'B'
    }
    return f"{chord_number} ({chord_names[str(chord_number)]})"

def get_chord_progression(chords):
    """Convert chord list to progression string with note names"""
    return ' → '.join(get_chord_name(chord) for chord in chords)

def find_chord_pattern(song_chords, pattern):
    """Find if a pattern of chords appears anywhere in the song's progression"""
    if not pattern or not song_chords:
        return 0, []
    
    # Convert everything to strings for comparison
    song_chords = [str(c) for c in song_chords]
    pattern = [str(c) for c in pattern]
    pattern_len = len(pattern)
    
    # Find all occurrences of the pattern
    occurrences = []
    
    # Look for the pattern anywhere in the progression
    for i in range(len(song_chords) - pattern_len + 1):
        # Check if pattern starts at position i
        matches = True
        for j in range(pattern_len):
            if song_chords[i + j] != pattern[j]:
                matches = False
                break
        if matches:
            occurrences.extend(range(i, i + pattern_len))
    
    # Count unique occurrences (some positions might overlap)
    occurrences = list(set(occurrences))
    return len(occurrences) // pattern_len, occurrences

def get_tempo_range(tempo, tolerance=0.2):
    """Get tempo range with tolerance"""
    lower = tempo * (1 - tolerance)
    upper = tempo * (1 + tolerance)
    return lower, upper

def get_average_tempo(tempo_features):
    """Calculate average tempo from tempo features"""
    try:
        if isinstance(tempo_features, str):
            tempo_features = ast.literal_eval(tempo_features)
        return sum(tempo_features) / len(tempo_features)
    except:
        return 0

def get_songs_by_chord_sequence(df, selected_chords, similarity_threshold=0.2, tempo_filter=None):
    """Get songs that contain the selected chord sequence anywhere in their progression"""
    matching_songs = []
    
    for _, row in df.iterrows():
        try:
            # Get song chords
            if isinstance(row['Chords'], str):
                song_chords = ast.literal_eval(row['Chords'])
            else:
                song_chords = row['Chords']
            
            # Convert to list if not already
            if not isinstance(song_chords, (list, tuple)):
                song_chords = list(song_chords)
            
            # Find pattern occurrences
            repeats, positions = find_chord_pattern(song_chords, selected_chords)
            
            # Calculate similarity based on pattern presence
            if positions:  # If pattern is found anywhere
                # Calculate how much of the song contains the pattern
                pattern_coverage = len(positions) / len(song_chords)
                # Base similarity on coverage and number of occurrences
                similarity = min(1.0, pattern_coverage + (repeats * 0.1))
                
                # Check tempo if filter is active
                if tempo_filter:
                    avg_tempo = get_average_tempo(row['Tempo'])
                    tempo_lower, tempo_upper = tempo_filter
                    if not (tempo_lower <= avg_tempo <= tempo_upper):
                        continue
                
                if similarity >= similarity_threshold:
                    matching_songs.append({
                        'song': row['Song'],
                        'similarity': similarity,
                        'progression': song_chords,
                        'pattern_positions': positions,
                        'repeats': repeats,
                        'tempo': get_average_tempo(row['Tempo'])
                    })
        except:
            continue
    
    # Sort by number of repeats first, then similarity
    matching_songs.sort(key=lambda x: (x['repeats'], x['similarity']), reverse=True)
    return matching_songs

def display_chord_progression(song_info):
    """Display chord progression with pattern highlighting"""
    progression = song_info['progression']
    pattern_positions = set(song_info['pattern_positions'])
    
    # Create HTML for the progression
    chord_elements = []
    for i, chord in enumerate(progression):
        chord_name = get_chord_name(chord)
        if i in pattern_positions:
            # Highlight matching pattern
            chord_elements.append(f'<span style="color: #FF4B91; font-weight: bold;">{chord_name}</span>')
        else:
            chord_elements.append(chord_name)
    
    progression_html = ' → '.join(chord_elements)
    
    st.markdown(f"""
    <div class="card">
        <h4>{song_info['song']}</h4>
        <p><strong>Pattern Found:</strong> {song_info['repeats']} time(s)</p>
        <p><strong>Match Score:</strong> {song_info['similarity']*100:.1f}%</p>
        <p><strong>Tempo:</strong> {song_info['tempo']:.1f} BPM</p>
        <p><strong>Full Progression:</strong></p>
        <p style="font-size: 1.1em; margin-top: 5px;">{progression_html}</p>
        <p><em style="color: #AAAAAA;">Highlighted text shows your chord sequence in the progression</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_chord_selector(df):
    """Display chord selection interface with ordered selection and tempo filtering"""
    st.markdown("### Select Chords")
    st.markdown("Choose the chords you want to find similar songs with. The order of selection matters!")
    
    # Get unique chords
    unique_chords = get_unique_chords(df)
    
    # Use session state to track chord selection order
    if 'selected_chords_order' not in st.session_state:
        st.session_state.selected_chords_order = []
    
    # Create columns for chord selection
    cols = st.columns(4)
    
    # Track changes in checkboxes
    for i, chord in enumerate(unique_chords):
        with cols[i % 4]:
            was_selected = chord in st.session_state.selected_chords_order
            is_selected = st.checkbox(get_chord_name(chord), key=f"chord_{i}", value=was_selected)
            
            if is_selected and chord not in st.session_state.selected_chords_order:
                st.session_state.selected_chords_order.append(chord)
            elif not is_selected and chord in st.session_state.selected_chords_order:
                st.session_state.selected_chords_order.remove(chord)
    
    # Add clear selection button
    if st.button("Clear Selection"):
        st.session_state.selected_chords_order = []
        st.rerun()
    
    # Add tempo filtering
    st.markdown("### Tempo Filter")
    use_tempo = st.checkbox("Filter by Tempo", value=False)
    tempo_filter = None
    
    if use_tempo:
        col1, col2 = st.columns(2)
        with col1:
            target_tempo = st.number_input("Target Tempo (BPM)", min_value=1, max_value=300, value=120)
        with col2:
            tempo_tolerance = st.slider("Tempo Tolerance", min_value=0.1, max_value=0.5, value=0.2, 
                                     format="±%.0f%%", help="How much the tempo can vary from target")
        tempo_filter = get_tempo_range(target_tempo, tempo_tolerance)
        
        st.markdown(f"""
        Looking for songs with tempo between 
        {tempo_filter[0]:.1f} and {tempo_filter[1]:.1f} BPM
        """)
    
    # Add similarity threshold slider
    if st.session_state.selected_chords_order:
        st.markdown("### Similarity Threshold")
        similarity_threshold = st.slider(
            "Minimum chord similarity percentage",
            min_value=0.2,
            max_value=1.0,
            value=0.5,
            step=0.1,
            format="%.0f%%"
        )
        return st.session_state.selected_chords_order, similarity_threshold, tempo_filter
    
    return [], 0.5, None

def plot_chord_distribution(df):
    """Plot distribution of chord usage across the dataset"""
    all_chords = []
    for chords in df['Chords']:
        if isinstance(chords, str):
            chords = ast.literal_eval(chords)
        all_chords.extend(chords)
    
    chord_counts = pd.Series(all_chords).value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=chord_counts.index, y=chord_counts.values, hue=chord_counts.index, legend=False, ax=ax)
    
    plt.title('Chord Distribution in Dataset', pad=20)
    plt.xlabel('Chord')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # Style the plot
    ax.set_facecolor('#1A1A1A')
    fig.patch.set_facecolor('#121212')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.tick_params(colors='#E0E0E0')
    ax.xaxis.label.set_color('#E0E0E0')
    ax.yaxis.label.set_color('#E0E0E0')
    ax.title.set_color('#E0E0E0')
    
    plt.tight_layout()
    return fig

def plot_tempo_distribution(df):
    """Plot distribution of tempos across the dataset"""
    tempos = [get_average_tempo(tempo) for tempo in df['Tempo']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(tempos, bins=30, ax=ax, color='#4A90E2')
    
    plt.title('Tempo Distribution in Dataset', pad=20)
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Count')
    
    # Style the plot
    ax.set_facecolor('#1A1A1A')
    fig.patch.set_facecolor('#121212')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['right'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.tick_params(colors='#E0E0E0')
    ax.xaxis.label.set_color('#E0E0E0')
    ax.yaxis.label.set_color('#E0E0E0')
    ax.title.set_color('#E0E0E0')
    
    plt.tight_layout()
    return fig

def analyze_song_features(song_data):
    """Analyze musical features of a song"""
    tempo = get_average_tempo(song_data['Tempo'])
    chord_progression = song_data['Chords']
    if isinstance(chord_progression, str):
        chord_progression = ast.literal_eval(chord_progression)
    
    unique_chords = len(set(chord_progression))
    progression_length = len(chord_progression)
    
    return {
        'tempo': tempo,
        'unique_chords': unique_chords,
        'progression_length': progression_length,
        'chord_progression': chord_progression
    }

def display_song_analysis(song_name, df):
    """Display detailed analysis of a song"""
    song_data = df[df['Song'] == song_name].iloc[0]
    analysis = analyze_song_features(song_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Tempo (BPM)</div>
        </div>
        """.format(analysis['tempo']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Unique Chords</div>
        </div>
        """.format(analysis['unique_chords']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Progression Length</div>
        </div>
        """.format(analysis['progression_length']), unsafe_allow_html=True)
    
    st.markdown("### Chord Progression")
    progression = ' → '.join(get_chord_name(chord) for chord in analysis['chord_progression'])
    st.markdown(f"""
    <div class="card">
        <p style="font-family: monospace; font-size: 1.1em;">{progression}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return analysis

def extract_features(audio_path):
    """Extract audio features from the given audio file."""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, duration=30)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Calculate statistics for each feature
        features = []
        for feature in [mfcc, spectral_centroid, chroma_stft, spectral_contrast, 
                       spectral_bandwidth, spectral_rolloff, zero_crossing_rate]:
            features.extend([
                np.mean(feature),
                np.std(feature),
                np.min(feature),
                np.max(feature)
            ])
        
        # Reshape features to match the model's expected input shape (6, 6, 1)
        features = np.array(features)
        # Pad or truncate to 36 features (6x6)
        if len(features) < 36:
            features = np.pad(features, (0, 36 - len(features)))
        else:
            features = features[:36]
            
        return features.reshape(1, 6, 6, 1)  # Reshape for CNN input
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def process_uploaded_audio(uploaded_file):
    """Process uploaded audio file and extract features"""
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp_upload.mp3")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features
        features = extract_features(temp_path)
        
        if features is not None:
            st.success("Audio features extracted successfully!")
            
            # Display debug information
            with st.expander("Feature Extraction Details"):
                st.write("Debug Information:")
                st.write("- Features Shape:", features.shape)
            
            return features
        
        return None
        
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()

def display_recommendation(rec):
    """Display a single recommendation with details"""
    st.markdown(f"""
    <div class="card">
        <h4>{rec['song']}</h4>
        <p>Similarity Score: {rec['similarity']:.2f}</p>
        <p>Tempo: {rec['features']['tempo']:.1f} BPM</p>
        <p>Chord Complexity: {rec['features']['chords']} unique chords</p>
    </div>
    """, unsafe_allow_html=True)
    display_audio_player(rec['song'])

# Define genre labels
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def main():
    # Show sidebar
    show_sidebar()
    
    # Create tabs
    tabs = st.tabs(["Song Similarity", "Chord Analysis", "Upload Song", "Dataset Insights"])
    
    # Song Similarity Tab
    with tabs[0]:
        st.title("🔍 Song Similarity Search")
        st.write("Find similar songs from our dataset")
        
        try:
            model, scaler, df = load_model_and_data()
            if model is not None and df is not None:
                # Prepare dataset features
                X_cnn, X_flat = preprocess_features(df, scaler)
                
                selected_song = st.selectbox("Select a song from dataset", df['Song'].tolist())
                
                if selected_song:
                    st.markdown("### Selected Song Analysis")
                    analysis = display_song_analysis(selected_song, df)
                    
                    if st.button("Find Similar Songs", key="dataset_similar"):
                        # Get song index
                        song_index = df[df['Song'] == selected_song].index[0]
                        
                        # Get recommendations
                        _, recommendations = get_recommendations(model, X_cnn, X_flat, df, song_index)
                        
                        if recommendations:
                            st.markdown("### Similar Songs")
                            for rec in recommendations:
                                display_recommendation(rec)
            else:
                st.error("Could not load model or dataset. Please check if all required files are present.")
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")

    # Chord Analysis Tab
    with tabs[1]:
        st.title("🎸 Chord Analysis")
        st.write("Analyze chord progressions and find similar songs")
        
        # Load data for chord analysis
        try:
            _, _, df = load_model_and_data()
            if df is not None:
                selected_chords, similarity_threshold, tempo_filter = display_chord_selector(df)
                
                if selected_chords:
                    st.markdown("### Selected Chord Progression")
                    st.markdown(get_chord_progression(selected_chords))
                    
                    matching_songs = get_songs_by_chord_sequence(
                        df, selected_chords, 
                        similarity_threshold=similarity_threshold,
                        tempo_filter=tempo_filter
                    )
                    
                    if matching_songs:
                        st.markdown(f"### Found {len(matching_songs)} Matching Songs")
                        for song in matching_songs:
                            display_chord_progression(song)
                            display_audio_player(song['song'])
                    else:
                        st.info("No songs found with this chord progression. Try adjusting the similarity threshold or tempo filter.")
        except Exception as e:
            st.error(f"Error in chord analysis: {str(e)}")

    # Upload Song Tab (moved to third position)
    with tabs[2]:
        st.title("🎵 Find Similar Songs by Upload")
        st.write("Upload your song to find similar music in our dataset")
        
        # Load model and data
        try:
            model, scaler, df = load_model_and_data()
            if model is not None and df is not None:
                # Prepare dataset features once
                X_cnn, X_flat = preprocess_features(df, scaler)
                
                # File uploader
                uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

                if uploaded_file is not None:
                    # Create a temporary file to save the uploaded audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    # Display audio player
                    st.audio(uploaded_file)

                    # Add a find similar songs button
                    if st.button("Find Similar Songs"):
                        with st.spinner("Analyzing audio and finding similar songs..."):
                            # Extract features
                            features = extract_features(temp_path)
                            
                            if features is not None:
                                try:
                                    # Get similarity scores using the model
                                    uploaded_song_features = features
                                    similarities = cosine_similarity(uploaded_song_features.reshape(1, -1), X_flat)[0]
                                    
                                    # Get top 5 most similar songs
                                    top_indices = similarities.argsort()[-5:][::-1]
                                    
                                    # Display results
                                    st.subheader("Similar Songs")
                                    
                                    for idx in top_indices:
                                        song_data = df.iloc[idx]
                                        similarity_score = similarities[idx]
                                        
                                        st.markdown(f"""
                                        <div class="card">
                                            <h4>{song_data['Song']}</h4>
                                            <p>Similarity Score: {similarity_score:.2f}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Display audio player for the similar song
                                        display_audio_player(song_data['Song'])
                                        
                                        # Display chord progression
                                        if 'Chords' in song_data:
                                            chords = song_data['Chords']
                                            if isinstance(chords, str):
                                                chords = ast.literal_eval(chords)
                                            st.markdown("#### Chord Progression")
                                            st.markdown(get_chord_progression(chords))
                                        
                                        st.markdown("---")
                                    
                                except Exception as e:
                                    st.error(f"Error finding similar songs: {str(e)}")
                                    st.write("Debug info:", str(e._class.name_))
                    
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            else:
                st.error("Could not load model or dataset. Please check if all required files are present.")
        except Exception as e:
            st.error(f"Error loading model and data: {str(e)}")

    # Dataset Insights Tab (remains as fourth tab)
    with tabs[3]:
        st.title("📊 Dataset Insights")
        st.write("Explore patterns and distributions in the music dataset")
        
        try:
            _, _, df = load_model_and_data()
            if df is not None:
                # Dataset statistics
                st.markdown("### Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Songs", len(df))
                
                with col2:
                    avg_tempo = np.mean([get_average_tempo(tempo) for tempo in df['Tempo']])
                    st.metric("Average Tempo", f"{avg_tempo:.1f} BPM")
                
                with col3:
                    avg_chords = np.mean([len(set(ast.literal_eval(chords) if isinstance(chords, str) else chords)) 
                                        for chords in df['Chords']])
                    st.metric("Avg. Unique Chords", f"{avg_chords:.1f}")
                
                # Chord distribution
                st.markdown("### Chord Distribution")
                chord_dist_fig = plot_chord_distribution(df)
                st.pyplot(chord_dist_fig)
                
                # Tempo distribution
                st.markdown("### Tempo Distribution")
                tempo_dist_fig = plot_tempo_distribution(df)
                st.pyplot(tempo_dist_fig)
                
        except Exception as e:
            st.error(f"Error loading dataset insights: {str(e)}")

if __name__ == "__main__":
    main()
