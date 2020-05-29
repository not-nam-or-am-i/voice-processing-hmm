import tkinter as tk
import threading
import pyaudio
import wave
import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import pickle

# lấy mfcc của tất cả các file wav trong wav
def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
    return mfcc


# read file
def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix



# for cname in class_names:
#     print('Accuracy:', cname, n_correct1[cname]/n_test1[cname])

class Recorder():
    models = pickle.load(open('hmm_models.sav', 'rb'))
    kmeans = pickle.load(open('kmeans.sav', 'rb'))
    predict_word = ''
    chunk = 1024 
    sample_format = pyaudio.paInt16 
    channels = 2
    fs = 44100
    sentence_counter = 0 #đếm câu
    
    frames = []  
    def __init__(self, master):
        self.isrecording = False
        self.text = tk.StringVar()
        self.text.set("sentence #0")
        
        self.label = tk.Label(master, textvariable=self.text)
        self.button1 = tk.Button(master, text='rec',command=self.start_recording, padx=50)
        self.button2 = tk.Button(master, text='stop',command=self.stop_recording, padx=50)
        self.button2["state"] = "disabled"
        
        self.label.pack()
        self.button1.pack()
        self.button2.pack()
        

    def start_recording(self):
        self.p = pyaudio.PyAudio()  
        self.stream = self.p.open(format=self.sample_format,channels=self.channels,rate=self.fs,frames_per_buffer=self.chunk,input=True)
        self.isrecording = True
        self.button1["state"] = "disabled"
        self.button2["state"] = "normal"
        
        print('Recording')
        t = threading.Thread(target=self.record)
        t.start()
    
    def change_label(self): #đổi số câu
        # self.text.set("sentence #"+str(self.sentence_counter))
        self.text.set(self.predict_word)

    def stop_recording(self):
        self.isrecording = False
        print('recording complete')
        self.button1["state"] = "normal"
        self.button2["state"] = "disable"
        
        self.filename= "sentence" + str(self.sentence_counter) + ".wav" #tên file
        self.sentence_counter = self.sentence_counter + 1
        # self.change_label()
        
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.frames = [] #reset lại dữ liệu ghi âm

        #file
        # testfile = get_class_data('./')
        # testfile = list([kmeans.predict(testfile).reshape(-1, 1)])

        testset1 = {}
        n_test1 = {}
        class_names = ['bệnh nhân', 'chúng ta', 'có thể', 'người','Việt Nam']
        for cname in class_names:
            print(f"Load {cname} test")
            testset1[cname] = get_class_data('./')
            n_test1[cname] = len(testset1[cname])

        for cname in class_names:
            testset1[cname] = list([self.kmeans.predict(v).reshape(-1, 1) for v in testset1[cname]])

        print("Testing")
        n_correct1 = {'bệnh nhân': 0, 'chúng ta': 0, 'có thể': 0, 'người': 0,'Việt Nam': 0}
        for true_cname in class_names:
            for O in testset1[true_cname]:
                score = {cname: model.score(O, [len(O)]) for cname, model in self.models.items()}
                if (true_cname == max(score, key=score.get)): n_correct1[true_cname] += 1
                print(true_cname, score, 'predict:', max(score, key=score.get))
                self.predict_word = max(score, key=score.get)

        self.change_label()
        
    def record(self):
        while self.isrecording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            

main = tk.Tk()
main.title('recorder')
main.geometry('400x500')
app = Recorder(main)
main.mainloop()