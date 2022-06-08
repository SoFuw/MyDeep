from tkinter import *
import tkinter as tk
from tkinter import filedialog
import os
from MyInference import MyInference
from MyMLP import MLP
import time


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master.geometry("500x400")
        self.master.title("body posture estimation")
        self.log_text = Text(self.master)
        self.sb = Scrollbar(self.master)
        self.sb.pack(side=RIGHT, fill=BOTH)
        self.log_text.configure(state="disabled", yscrollcommand=self.sb.set)
        self.sb.config(command=self.log_text.yview)

        self.log_text.pack()
        self.image_file_search_button = Button(
            self.master, width=100, text='search a image file', command=self.__go_inference_image__)
        self.image_file_search_button.pack()
        self.image_dir_search_button = Button(
            self.master, width=100, text='search a image folder', command=self.__go_inference_images__)
        self.image_dir_search_button.pack()
        self.video_file_search_button = tk.Button(self.master, width=100, text="search a video file",
                                                  command=self.__go_inference__)
        self.video_file_search_button.pack()

    def __go_inference__(self):
        self.log_text.focus()
        self.filename = filedialog.askopenfile(initialdir=os.getcwd(
        ), title="비디오 파일 찾기", filetypes=[('video files', '.mp4 .avi')])
        self.__set_input__(f'selected file : {self.filename.name}\n')
        inference = MyInference()
        inference.load_video(self.filename.name)
        start = time.time()
        self.__set_input__(f'processing... : {self.filename.name}\n')
        inference.inference_video(8)
        self.__set_input__(
            f'Done! go to result_video folder : {self.filename.name}\n')
        self.__set_input__(f'processing time : {time.time()-start}s')

    def __go_inference_image__(self):
        self.log_text.focus()
        self.filename = filedialog.askopenfile(initialdir=os.getcwd(
        ), title="이미지 파일 찾기", filetypes=[('image files', '.jpg .png .jpeg')])
        self.__set_input__(f'selected file : {self.filename.name}\n')
        inference = MyInference()
        start = time.time()
        self.__set_input__(f'processing... : {self.filename.name}\n')
        inference.inference_image(self.filename.name)
        self.__set_input__(
            f'Done! go to result_images folder : {self.filename.name}\n')
        self.__set_input__(f'processing time : {time.time()-start}s')

    def __go_inference_images__(self):
        self.log_text.focus()
        self.foldername = filedialog.askdirectory()
        self.__set_input__(f'selected folder : {self.foldername}\n')
        inference = MyInference()
        inference.load_images(self.foldername)
        start = time.time()
        self.__set_input__(f'processing... : {self.foldername}\n')
        inference.inference_images(8)
        self.__set_input__(
            f'Done! go to result_images folder : {self.foldername}\n')
        self.__set_input__(f'processing time : {time.time()-start}s')

    def __set_input__(self, str: str):
        self.log_text.config(state='normal')
        self.log_text.insert('end', str)
        self.log_text.configure(state="disabled")
        self.master.update()


root = Tk()
app = App(root)
app.master.mainloop()

#tkinter로 진행...
