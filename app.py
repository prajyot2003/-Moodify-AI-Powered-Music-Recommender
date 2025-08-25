import os
import time
import streamlit as st
from transformers import pipeline
from youtubesearchpython import VideosSearch

# -----------------------------
# App Setup / Config
# -----------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # quiet Mac MPS warnings
st.set_page_config(page_title="Moodify 🎶", page_icon="🎵", layout="centered")

st.title("🎵 Moodify – AI Music Recommender")
st.caption("Type how you feel (or speak locally), I'll detect your emotion and play matching music 🎶")

# Detect if we're on Hugging Face Space
RUNNING_IN_SPACE = "SPACE_ID" in os.environ

# -----------------------------
# Optional voice input (only local)
# -----------------------------
use_voice = False
if not RUNNING_IN_SPACE:
    try:
        import speech_recognition as sr  # optional local dependency
        use_voice = True
    except Exception:
        # Keep running; voice is optional
        pass

# -----------------------------
# Load emotion model (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    # DistilRoBERTa emotion model (lightweight & accurate)
    # Returns top label by default (pipeline default top_k=1)
    return pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base", device=-1)

emotion_model = load_emotion_model()

# -----------------------------
# Utilities
# -----------------------------
def detect_emotion(text: str) -> str:
    """Return lowercase emotion label from model."""
    try:
        res = emotion_model(text)[0]
        label = res["label"].lower()
        return label
    except Exception:
        return "neutral"

# Map detected emotions → search moods/genres
EMOTION_TO_MOOD = {
    "joy": "happy",
    "love": "romantic",
    "anger": "calm",        # steer to calming music
    "sadness": "sad",
    "fear": "relaxing",
    "surprise": "energetic",
    "disgust": "moody",
    "neutral": "chill"
}

# Extra moods you can pick manually
MOOD_OPTIONS = sorted(list(set(EMOTION_TO_MOOD.values()) | {"lofi", "focus", "party", "workout", "sleep"}))

@st.cache_data(show_spinner=False)
def yt_search(query: str, limit: int = 6):
    vs = VideosSearch(query, limit=limit)
    data = vs.result()
    items = data.get("result", [])
    # Return (title, url, id)
    parsed = []
    for v in items:
        title = v.get("title", "Untitled")
        link = v.get("link")
        vid_id = None
        # Most links are standard watch?v=..; if not, just pass link to st.video()
        if link and "watch?v=" in link:
            vid_id = link.split("watch?v=")[-1].split("&")[0]
        parsed.append((title, link, vid_id))
    return parsed

def get_voice_input() -> str | None:
    """Record a short phrase and transcribe with Google (local only)."""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Speak now…")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
        text = recognizer.recognize_google(audio)
        st.success(f"You said: **{text}**")
        return text
    except sr.UnknownValueError:
        st.error("I couldn't understand—try again.")
    except sr.WaitTimeoutError:
        st.error("No voice detected—try again.")
    except Exception as e:
        st.error(f"Voice error: {e}")
    return None

# -----------------------------
# UI
# -----------------------------
st.write("### How do you want to share your mood?")
if use_voice:
    input_mode = st.radio("Input method", ["Type", "Speak"], horizontal=True)
else:
    input_mode = "Type"
    if RUNNING_IN_SPACE:
        st.info("🎙️ Voice input is disabled on Hugging Face Spaces. Use text input below.")

user_text = ""
if input_mode == "Type":
    user_text = st.text_input("Tell me how you feel (e.g., “I feel lonely”, “I’m excited for tonight”)")
else:
    if st.button("🎤 Tap to Speak"):
        said = get_voice_input()
        if said:
            user_text = said

# If user gave any text, detect emotion → propose mood; else let them manually pick
detected_emotion = None
detected_mood = None

if user_text.strip():
    with st.spinner("Analyzing your feelings…"):
        detected_emotion = detect_emotion(user_text)
    st.write(f"**Detected emotion:** `{detected_emotion}`")
    detected_mood = EMOTION_TO_MOOD.get(detected_emotion, "chill")

# Manual override dropdown (preselect detected mood if we have it)
st.write("### Choose/adjust mood")
default_index = MOOD_OPTIONS.index(detected_mood) if detected_mood in MOOD_OPTIONS else MOOD_OPTIONS.index("chill")
chosen_mood = st.selectbox(
    "Adjust if needed (auto set from your feelings):",
    MOOD_OPTIONS,
    index=default_index
)

# Generate playlist
if st.button("🎵 Generate Playlist"):
    if not user_text.strip() and not detected_mood:
        st.warning("Type how you feel or use the dropdown to pick a mood.")
    else:
        query = f"{chosen_mood} songs playlist"
        with st.spinner(f"Finding {chosen_mood} tracks on YouTube…"):
            songs = yt_search(query, limit=6)

        if not songs:
            st.error("No songs found. Try a different mood.")
        else:
            st.subheader("🎶 Your Moodify Playlist")
            # Autoplay first track if possible
            first = songs[0]
            if first[2]:  # have video id
                # Embed playlist style (first as main, others as queue)
                ids = [s[2] for s in songs if s[2]]
                if len(ids) > 1:
                    embed_url = f"https://www.youtube.com/embed/{ids[0]}?autoplay=1&playlist={','.join(ids[1:])}"
                else:
                    embed_url = f"https://www.youtube.com/embed/{ids[0]}?autoplay=1"
                st.components.v1.iframe(embed_url, height=420, width=720)
            else:
                # Fallback to direct video link
                st.video(first[1])

            with st.expander("Show track list"):
                for title, link, _ in songs:
                    st.markdown(f"- [{title}]({link})")

            st.success("Enjoy your music! 🎧")
