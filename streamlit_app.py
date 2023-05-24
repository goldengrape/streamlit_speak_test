# 下面这个程序可以正常输出语音，但是它每次会说两遍，这是为什么？
from langchain.callbacks.base import BaseCallbackHandler
import azure.cognitiveservices.speech as speechsdk
import os
# from pydub import AudioSegment  
import base64
import time 


class StreamDisplayHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response, **kwargs) -> None:
        self.text=""


class StreamSpeakHandler(BaseCallbackHandler):
    def __init__(self, container, synthesis="zh-CN-XiaoxiaoNeural", rate="+50.00%"):
        self.container = container
        self.new_sentence = ""
        # Initialize the speech synthesizer
        self.synthesis=synthesis
        self.rate=rate
        self.speech_synthesizer = self.settings(synthesis)

    def settings(self, synthesis):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get('SPEECH_KEY'), 
            region=os.environ.get('SPEECH_REGION')
        )
        audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        
        speech_config.speech_synthesis_voice_name=synthesis

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
        return speech_synthesizer

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.new_sentence += token
        # Check if the new token forms a sentence.
        if token in ".:!?。：！？\n":
            # Synthesize the new sentence
            speak_this = self.new_sentence
            if len(speak_this) > 0:
                ssml_text=f"""<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" version="1.0" xml:lang="en-US">
    <voice name="{self.synthesis}">
    <prosody rate="{self.rate}">
            {speak_this}
    </prosody>
    </voice>
</speak>"""



                # self.speak_ssml_async(ssml_text)
                self.speak_text_to_streamlit(ssml_text)
            self.new_sentence = ""

    def on_llm_end(self, response, **kwargs) -> None:
        self.new_sentence = ""

    def speak_ssml_async(self, text):
        speech_synthesis_result = self.speech_synthesizer.speak_ssml_async(text).get()
        if speech_synthesis_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f'Error synthesizing speech: {speech_synthesis_result.reason}')

    def speak_text_to_streamlit(self, text):
        result = self.speech_synthesizer.speak_ssml_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_stream = result.audio_data
            audio_base64 = base64.b64encode(audio_stream).decode('utf-8')
            audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
            self.container.markdown(audio_tag, unsafe_allow_html=True)
            print(text)
            print(result.audio_duration)
            time.sleep(result.audio_duration)


#### demo ####
# from StreamHandler import StreamDisplayHandler, StreamSpeakHandler

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["SPEECH_KEY"] = st.secrets["speech_key"]
os.environ['SPEECH_REGION']=st.secrets["speech_region"]

query = st.text_input("input your query", value="Tell me a joke")
ask_button = st.button("ask")

st.markdown("### streaming box")
chat_box = st.empty()
speak_box=st.empty()
display_handler = StreamDisplayHandler(
    chat_box,
    display_method='write')
speak_handler = StreamSpeakHandler(container=speak_box, synthesis="en-US-AriaNeural",rate="+30.00%")
chat = ChatOpenAI(
        max_tokens=100, streaming=True,
        callbacks=[display_handler, speak_handler])


st.markdown("### together box")

if query and ask_button:
    response = chat([HumanMessage(content=query)])
    llm_response = response.content
    st.markdown(llm_response)