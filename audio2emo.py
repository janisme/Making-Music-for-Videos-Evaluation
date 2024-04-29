from pydub import AudioSegment
from speechbrain.inference.interfaces import foreign_class
import os
import random
# https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/discussions


''''This function takes in .mp3 file path and convert .mp3 to .wav'''
def conversion(mp3_file_path):
    # mp3_file_path = "vmcpdata/Q4_10sec.mp3"
    wav_file_path = "vmcpdata/Q4.wav"

    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)

    # Export the audio to WAV format
    audio.export(wav_file_path, format="wav")

    print("Conversion complete!")

''''This function takes in path of .wav file and comp. the related emotion'''
def audio2emot(audio_path):
    # audio_path= "vmcpdata/Q4.wav"
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
    print("complete1")
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    # print(out_prob)
    # print(score)
    # print(text_lab)

    return text_lab

'''This python file is to find the emotio  of the audio >> drama2emo.txt'''
if __name__ == '__main__':
    # emo:["sad", "happy", "angry", "neutral"]
    dir_path = "vmcpdata/exp1-2"

    print("loading model ...")
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
    print("complete1")

    ct = 0
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        if file_path.lower().endswith('.wav'):
            ct +=1

            _, score, _, emo = classifier.classify_file(file_path)

            print(f'the {ct} clip is {emo}')
            print(score)
