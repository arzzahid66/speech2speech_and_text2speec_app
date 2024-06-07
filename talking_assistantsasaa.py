import assemblyai as aai
from qdrant_client import QdrantClient
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from io import BytesIO
from st_audiorec import st_audiorec
from typing import IO
import streamlit as st
from langchain_community.document_loaders.assemblyai import TranscriptFormat
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
aai.settings.api_key = "open ai key "
GOOGLE_API_KEY ="gemini key "
api_key = GOOGLE_API_KEY

def qdrant_client():
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        qdrant_key = "api key "
        URL = "url qdrant"
        qdrant_client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
        )
        qdrant_store = Qdrant(qdrant_client,"my_first_xeven_collection" ,embedding_model)
        return qdrant_store

qdrant_store = qdrant_client()

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)
API_KEY = os.getenv("OPENAI_API_KEY")


def assembly_ai_voice_to_text(audio_location):
    loader = AssemblyAIAudioTranscriptLoader(file_path=audio_location)
    transcript = loader.load()
    text = transcript[0].page_content
    return text

def transcribe_voice_to_text(audio_location):
    client = OpenAI(api_key=API_KEY)
    audio_file= open(audio_location, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

def chat_completion_call(text):
    client = OpenAI(api_key=API_KEY)
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content

# for rag uncomment this 
# def chat_completion_call(text):
#         response = qa_ret(qdrant_store,text)
#         return response

def text_to_speech_ai(speech_file_path,response):
    client = OpenAI(api_key=API_KEY)
    response = client.audio.speech.create(model="tts-1",voice="nova",input=response)
    response.stream_to_file(speech_file_path)

def text_to_speech_ai_with_elevenlab(speech_file_path,text: str) -> IO[bytes]:
    """
    Converts text to speech and returns the audio data as a byte stream.

    This function invokes a text-to-speech conversion API with specified parameters, including
    voice ID and various voice settings, to generate speech from the provided text. Instead of
    saving the output to a file, it streams the audio data into a BytesIO object.
    Args:
        text (str): The text content to be converted into speech.
    Returns:
        IO[bytes]: A BytesIO stream containing the audio data.
    """
    # Perform the text-to-speech conversion
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    print("Streaming audio data...")
    speech_file_path = speech_file_path

    # Writing the audio to a file
    with open(speech_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    # print(f"{save_file_path}: A new audio file was saved successfully!"
    # Return the path of the saved audio file
    return speech_file_path
# for rag 
# def qdrant_client():
#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         qdrant_key = "key"
#         URL = "url"
#         qdrant_client = QdrantClient(
#         url=URL,
#         api_key=qdrant_key,
#         )
#         qdrant_store = Qdrant(qdrant_client,"my_first_xeven_collection" ,embedding_model)
#         return qdrant_store

# qdrant_store = qdrant_client()

# def qa_ret(qdrant_store,text):
#     try:
#         template = """You are AI assistant that assisant user by providing answer to the question of user by extracting information from provided context:
#         {context} and chat_history if user question is related to chat_history take chat history as context .
#         if you donot find any relevant information from context for given question just say ask me another quuestion. you are ai assistant.
#         Answer should not be greater than 3 lines.
#         Question: {question}
#         """
#         prompt = ChatPromptTemplate.from_template(template)
#         retriever= qdrant_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
#         setup_and_retrieval = RunnableParallel(
#                 {"context": retriever, "question": RunnablePassthrough()}
#                 )
#             # Load QA Chain
#         model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key =api_key)
#         output_parser= StrOutputParser()
#         rag_chain = (
#         setup_and_retrieval
#         | prompt
#         | model
#         | output_parser
#         )
#         respone=rag_chain.invoke(text)
#         return respone
#     except Exception as ex:
#         return ex

method = st.radio("Select Method",("Openai_4o","Assembly_ai_openai_Elevenlab"))
st.title("QA Retrieval with speech to text and text to speech")
"""
Hi just click on the voice recorder and let me know how I can help you today ?
"""
wav_audio_data = st_audiorec()
text = st.text_input("Enter your Question here")
if method == "Assembly_ai_openai_Elevenlab":
    if st.button("submit"):
            st.write(text)
            api_response = chat_completion_call(text)
            st.write(api_response)
            speech_file_path = 'audio_response.mp3'
            text_to_speech_ai_with_elevenlab(speech_file_path, api_response)
            st.audio(speech_file_path)
if wav_audio_data is not None:
    # st.audio(wav_audio_data, format='audio/wav')
    ##Save the Recorded File
    audio_location = "audio_file.wav"
    # st.audio(wav_audio_data,format=".wav")
    with open(audio_location, "wb") as f:
        f.write(wav_audio_data)
    if method == "Openai_4o":
        text = transcribe_voice_to_text(audio_location)
        st.write(text)
        api_response = chat_completion_call(text)
        st.write(api_response)
        speech_file_path = 'audio_response.mp3'
        text_to_speech_ai(speech_file_path, api_response)
        st.audio(speech_file_path)
    if method == "Assembly_ai_openai_Elevenlab":
        text = assembly_ai_voice_to_text(audio_location)
        st.write(text)
        api_response = chat_completion_call(text)
        st.write(api_response)
        speech_file_path = 'audio_response.mp3'
        text_to_speech_ai_with_elevenlab(speech_file_path, api_response)
        st.audio(speech_file_path)

# Display the image with pricing plans
from PIL import Image

image_path = 'img_price.png'
image = Image.open(image_path)
image = image.resize((900, 400))
st.image(image, caption='Pricing plans for audio models')



      




