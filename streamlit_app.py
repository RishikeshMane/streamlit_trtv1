
from datetime import datetime, timedelta
from audiorecorder import audiorecorder
from code_editor import code_editor
from pathlib import Path
from io import BytesIO
import tempfile
import shutil
import uuid
import re
import io

import streamlit as st
import whisper
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS

#text to speech model declearation
from transformers import pipeline
from datasets import load_dataset
from openai import OpenAI
import tempfile
import torch
import os
from openai import OpenAI
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk

client = OpenAI()
client = OpenAI()


client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

modelspech = "speecht5_tts"
synthesiser = pipeline("text-to-speech", modelspech)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

st.set_page_config(page_title='Openai Whisper Transcriber',
                page_icon='📜',
                layout="wide",
                initial_sidebar_state="expanded")

# apply custom css
with open(Path('utils/style.css')) as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


@st.cache_resource(ttl=60*60*24, show_spinner=False)
def cleanup_tempdir() -> None:
    '''Cleanup temp dir for all user sessions.
    Filters the temp dir for uuid4 subdirs.
    Deletes them if they exist and are older than 1 day.
    '''
    deleteTime = datetime.now() - timedelta(days=1)
    uuid4_regex = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    uuid4_regex = re.compile(uuid4_regex)
    tempfiledir = Path(tempfile.gettempdir())
    if tempfiledir.exists():
        subdirs = [x for x in tempfiledir.iterdir() if x.is_dir()]
        subdirs_match = [x for x in subdirs if uuid4_regex.match(x.name)]
        for subdir in subdirs_match:
            itemTime = datetime.fromtimestamp(subdir.stat().st_mtime)
            if itemTime < deleteTime:
                shutil.rmtree(subdir)


def get_all_uuid4_subdirs_from_tempdir() -> None:
    '''Get all uuid4 subdirs from temp dir for all user sessions.'''
    uuid4_regex = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    uuid4_regex = re.compile(uuid4_regex)
    tempfiledir = Path(tempfile.gettempdir())
    if tempfiledir.exists():
        subdirs = [x for x in tempfiledir.iterdir() if x.is_dir()]
        subdirs_match = [x for x in subdirs if uuid4_regex.match(x.name)]
        subdirs_match = [str(x) for x in subdirs_match]
        return subdirs_match
    else:
        return None


@st.cache_data(show_spinner=False)
def make_tempdir() -> Path:
    '''Make temp dir for each user session and return path to it
    returns: Path to temp dir
    '''
    if 'tempfiledir' not in st.session_state:
        tempfiledir = Path(tempfile.gettempdir())
        tempfiledir = tempfiledir.joinpath(f"{uuid.uuid4()}")   # make unique subdir
        tempfiledir.mkdir(parents=True, exist_ok=True)  # make dir if not exists
        st.session_state['tempfiledir'] = tempfiledir
    return st.session_state['tempfiledir']


def store_file_in_tempdir(tmpdirname: Path, uploaded_file: BytesIO) -> Path:
    '''Store file in temp dir and return path to it
    params: tmpdirname: Path to temp dir
            uploaded_file: BytesIO object
    returns: Path to stored file
    '''
    # store file in temp dir
    tmpfile = tmpdirname.joinpath(uploaded_file.name)
    with open(tmpfile, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return tmpfile


def write_file_in_tempdir(tmpdirname: Path, uploaded_file: bytearray, filename: str) -> Path:
    '''Write file in temp dir and return path to it
    params: tmpdirname: Path to temp dir
            uploaded_file: bytearray
            filename: str
    returns: Path to stored file
    '''
    # store file in temp dir
    tmpfile = tmpdirname.joinpath(filename)
    with open(tmpfile, 'wb') as f:
        f.write(uploaded_file)
    return tmpfile


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    model = whisper.load_model(name)
    return model


@st.cache_data(show_spinner=False)
def transcribe(_model, audio_file):
    transcription = _model.transcribe(audio_file, fp16=False)
    return transcription["text"]

temp_data = ""

if __name__ == "__main__":

    st.title(''':skyblue[***UNRUFFLEDFEATHERS***]''')
    st.divider()
    
    tab1, tab2, tab3 = st.tabs([" Live Transciption "," Read files ", " Interact with Data "])
     

     
    # with tab2:
       
    cleanup_tempdir()  # cleanup temp dir from previous user sessions
    tmpdirname = make_tempdir()  # make temp dir for each user session
    st.title("Chat Promt")

    client = OpenAI(api_key=OpenAI.api_key)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter prompt"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    with tab1:
        audio_bytes = audio_recorder(pause_threshold=40)
        if audio_bytes:
            # Check if audio is of sufficient length
            if len(audio_bytes) > 8000:
                st.success('Audio captured correctly')
            else:
                st.warning('Audio captured incorrectly, please try again.')
            st.audio(audio_bytes, format="audio/wav")
            st.session_state.audio_bytes = audio_bytes

            # Form for real-time translation
        with st.form('input_form'):
            submit_button = st.form_submit_button(label='Translate', type='primary')
            if submit_button and 'audio_bytes' in st.session_state and len(st.session_state.audio_bytes) > 0:
                # Translate audio bytes into English
                audio_file = io.BytesIO(st.session_state.audio_bytes)
                audio_file.name = "temp_audio_file.wav"
                transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
                )
                temp_data=transcript
                print(temp_data)
                
                st.markdown("***Translation Transcript***")
                st.text_area('transcription', transcript, label_visibility='collapsed')
                #enable if to show Synthesized speech
                # if transcript:
                #     # Convert text to speech
                #     sound_file = BytesIO()
                #     tts = gTTS(transcript, lang='en')
                #     tts.write_to_fp(sound_file)
                #     st.markdown("***Synthesized Speech Translation***")
                #     st.audio(sound_file)
                # else:
                #     st.warning('No text to convert to speech.')
                
            else:
                    st.warning('No audio recorded, please ensure your audio was recorded correctly.')
        with st.expander("See explanation"):
            temprole1 = st.selectbox(
            'Select your role?',
            ('Web Project Manager', 'Web Designer', 'Frontend Developer','Backend Developer'))
            
            req1 = st.selectbox(
            'Select your requirement?',
            ('Business requirement template', 'Software or website requirement specification', 'Feature list','all of above'))

            # if st.button('Data'):
            #     print(temp_data)
            #     st.write('Given role :', temprole1 ,' and requirement '+ req1 , 'give solution on ' + temp_data)
                
            if 'button' not in st.session_state:
                st.session_state.button = False

            def click_button():
                st.session_state.button = not st.session_state.button
                # The message and nested widget will remain on the page
                def recognize_from_microphone():
                    
                    speech_config.speech_recognition_language="en-US"

                    #To recognize speech from an audio file, use `filename` instead of `use_default_microphone`:
                    #audio_config = speechsdk.audio.AudioConfig(filename="YourAudioFile.wav")
                    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
                    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

                    st.text("Speak into your microphone.")
                    speech_recognition_result = speech_recognizer.recognize_once_async().get()

                    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                        st.text("Recognized: {}".format(speech_recognition_result.text))
                    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                        st.text("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                        cancellation_details = speech_recognition_result.cancellation_details
                        st.text("Speech Recognition canceled: {}".format(cancellation_details.reason))
                        if cancellation_details.reason == speechsdk.CancellationReason.Error:
                            st.text("Error details: {}".format(cancellation_details.error_details))
                            st.text("Did you set the speech resource key and region values?")
                recognize_from_microphone()

            st.button('Click me', on_click=click_button)

            if st.session_state.button:
                st.write('Button is on!')
            else:
                st.write('Button is off!')
            
                    
    
    # with tab3:
    #     temprole1 = st.selectbox(
    #     'Select your role?',
    #     ('Web Project Manager', 'Web Designer', 'Frontend Developer','Backend Developer'))
        
    #     req1 = st.selectbox(
    #     'Select your requirement?',
    #     ('Business requirement template', 'Software or website requirement specification', 'Feature list','all of above'))

        #st.write('Given role :', temprole1 ,' and requirement '+ req1 , 'give solution on ' + transcript['text'])
        
        # with st.expander("See explanation"):
        #     txt = st.text_area(
        #     "Text to analyze",
        #     "Role  - as a expert \n" 
        #     "conversation - " + temp_data,
        #     )
    
    with tab2:
        
        with st.expander("Read File"):
            file = st.file_uploader("Choose a file",type=["txt",'csv','xlsx'])
            if file:
                # this will write UploadedFile(id=2, name='test.txt', type='text/plain', size=666)
                st.write(file)
                if file.type=='text/plain':
                    from io import StringIO
                    stringio=StringIO(file.getvalue().decode('utf-8'))
                    read_data=stringio.read()
                    temp_data=temp_data.join(read_data)
                    st.write(read_data)
              
        with st.expander("Speech To Text"):
            st.title("Speech to Text ")
       
            with st.spinner("Loading Openai Model for the first time, please wait..."):
                model = load_model(name="small")
                st.subheader("Audio File - input ")
                audio_file = st.file_uploader("Upload Audio",
                                            type=["wav", "mp3", "m4a", "ogg"],
                                            label_visibility="collapsed")

                audio = audiorecorder("Click to record", "Recording...")

                # if audio_file is not None:
                #     tmpfile = store_file_in_tempdir(tmpdirname, audio_file)
                #     audio_file_path = Path(audio_file.name)
                # elif len(audio) > 0:
                #     audio_file_bytes = audio.tobytes()
                #     tmpfile = write_file_in_tempdir(tmpdirname, audio_file_bytes, filename="audio.mp3")
                #     audio_file_path = Path("audio.mp3")
                # else:
                #     #bottom_image = st.file_uploader('', type='jpg', key=6)
                #     # imagee = 'logo.jpg'
                #     # if imagee is not None:
                #     #     image = Image.open(imagee)
                #     #     new_image = image.resize((100, 100))
                #     #     st.sidebar.image(new_image,width=400)
                #     st.sidebar.warning("Please Upload an Audio File")
                #     # st.sidebar.image('logo.jpg',width=400)
                #     st.stop()
                if audio_file is not None:
                    tmpfile = store_file_in_tempdir(tmpdirname, audio_file)
                    audio_file_path = Path(audio_file.name)
                elif len(audio) > 0:
                    audio_file_bytes = audio.tobytes()
                    tmpfile = write_file_in_tempdir(tmpdirname, audio_file_bytes, filename="audio.mp3")
                    audio_file_path = Path("audio.mp3")
                else:
                    st.sidebar.warning("Please Upload an Audio File")
                    st.stop()
                
                st.sidebar.header("Play Original Audio File")
                st.sidebar.audio(audio_file)
                if st.sidebar.button("Transcribe Audio ✍️"):
                    st.sidebar.info("Transcribing Audio...")
                    with st.spinner("Running Transcribe, please wait..."):
                        transcription = transcribe(model, str(tmpfile.resolve()))
                    st.sidebar.success("Transcription Complete")
                    st.subheader("Transcription Result ✍️")
                    st.success(audio_file.name)
                    st.markdown(transcription)
                    st.download_button(label='Download Text File ',
                                data=transcription,
                                file_name=audio_file_path.with_suffix(".txt").name)          