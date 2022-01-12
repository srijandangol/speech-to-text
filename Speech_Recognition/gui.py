from tkinter import *
import tkinter as tk
import argparse
import os
import ast
import queue
import sounddevice
import vosk
import sys
import _thread
from datetime import datetime

from PIL import ImageTk, Image
from itertools import count, cycle
from speech import get_text


root = Tk()
root.title('Hello Assistant')

#application width and height
appWidth = 480
appHeight = 600

#screen width and height
screenWidth = root.winfo_screenwidth()
screenHeight = root.winfo_screenheight()

x = int((screenWidth / 2) - (appWidth / 2))
y = int((screenHeight / 2) - (appHeight / 2))

# window pops up on center of the screen
root.geometry(f'{appWidth}x{appHeight}+{int(x)}+{int(y)}')

root.resizable(False,False)
root.configure(bg='black')
is_running=True
final_text  =[]
fls = StringVar()
fls2 = StringVar()
fls3 = StringVar()
fls_diff = StringVar()




#start recording
def record():
    global is_running
    is_running=True
    fls.set('Recording') # Update
    _thread.start_new_thread(startrecord,())

def stop():
    global is_running,final_text
    is_running = False
    fls.set('Start Recording')
    time_stamp = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.txt"
    time_stamp = 'textfile.txt'
    with open(time_stamp, 'a') as f:
        f.write(' '.join(final_text))
        f.close()

# recorder
global s_audio

def startrecord():
    global s_audio
    s_audio=0
    class ImageLabel(tk.Label):
        """
        A Label that displays images, and plays them if they are gifs
        :im: A PIL Image instance or a string filename
        """

        def load(self, im):
            if isinstance(im, str):
                im = Image.open(im)
            frames = []

            try:
                for i in count(1):
                    frames.append(ImageTk.PhotoImage(im.copy()))
                    im.seek(i)
            except EOFError:
                pass
            self.frames = cycle(frames)

            try:
                self.delay = im.info['duration']
            except:
                self.delay = 100

            if len(frames) == 1:
                self.config(image=next(self.frames))
            else:
                self.next_frame()

        def unload(self):
            self.config(image=None)
            self.frames = None

        def next_frame(self):
            global after_id, s_audio
            if self.frames:
                self.config(image=next(self.frames))
                if s_audio == 0:
                    after_id = self.after(self.delay, self.next_frame)

    lbl = ImageLabel(root)

    lbl.place(relx=0.07, rely=0.65)
    lbl.load('wave.gif')
    q = queue.Queue()
    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')

    args, remaining = parser.parse_known_args()

    if args.list_devices:
        print(sounddevice.query_devices())
        parser.exit(0)
        parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])

    parser.add_argument(
        '-f', '--filename', type=str, metavar='FILENAME',
        help='audio file to store recording to')

    parser.add_argument(
        '-m', '--model', type=str, metavar='MODEL_PATH',
        help='Path to the model')

    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')

    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    args = parser.parse_args(remaining)
    try:
        if args.model is None:
            args.model = "model"
        if not os.path.exists(args.model):
            print ("Please download a model for your language from https://alphacephei.com/vosk/models")
            print ("and unpack as 'model' in the current folder.")
            parser.exit(0)
        if args.samplerate is None:
            device_info = sounddevice.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])

        model = vosk.Model(args.model)

        if args.filename:
            dump_fn = open(args.filename, "wb") f
        else:
            dump_fn = None

        with sounddevice.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device, dtype='int16',
                                channels=1, callback=callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('#' * 80)

                rec = vosk.KaldiRecognizer(model, args.samplerate)
                while is_running:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        text = ast.literal_eval(rec.FinalResult())['text']
                        global final_text
                        final_text.append(text)
                        new_text = ' '.join(final_text)
                        new_text.insert(END, final_text[-1] + '\n')
                        new_text = [new_text[i:i+110] for i in range(0, len(new_text), 110)]
                        fls3.set('\n'.join(new_text))
                        print(new_text)
                        pass
                    else:
                        # print(rec.PartialResult())
                        text = ast.literal_eval(rec.PartialResult())['partial']
                        text =  [text[i:i+110] for i in range(0, len(text), 110)]
                        inputtxt1.delete("1.0","end")
                        inputtxt.delete("1.0", "end")
                        inputtxt1.insert("end-1c", text)
                        print(text)

                    if dump_fn is not None:

                        dump_fn.write(data)
                else:
                    parser.exit(0)

        #inputtxt.insert("end-1c", "Performing your command")
    except Exception as e:
        s_audio=1
        c = Canvas(bg="black", height=105, width=405,highlightthickness=0)
        c.place(relx=0.07, rely=0.65)
        inputtxt.insert("end-1c", "Performing your command")
        parser.exit(type(e).__name__ + ': ' + str(e))







# label widget
myLabel= Label(root, text="Voice Assistant",
                 font=("Calibri", 20), fg="brown", anchor=N)  # Title
myLabel.place(relx=0.3,rely=0.1)

# label1 widget
myLabel1 = Label(root, text="Owner",
                 font=("Calibri", 8), fg="black", anchor=N)  # Title
myLabel1.place(relx=0.9,rely=0.175)

# label2 widget
myLabel2 = Label(root, text="Assistant",
                 font=("Calibri", 8), fg="black", anchor=N)  # Title
myLabel2.place(relx=0.01,rely=0.375)

# input text
inputtxt1= Text(root, height=7, width=60)
inputtxt1.place(relx=0.125, rely=0.2)

# input text
inputtxt = Text(root, height=7, width=60)
inputtxt.place(relx=0, rely=0.4)


#button
originalImg = Image.open("micro.png")
resized = originalImg.resize((60, 60), Image.ANTIALIAS)
img = ImageTk.PhotoImage(resized)

button_record = Button(root, padx=20, pady=5,fg="Red",image=img,command=record,highlightthickness = 0, bd = 0,bg="lightgrey")

button_record.place(relx=0.45,rely=0.87)

root.mainloop()