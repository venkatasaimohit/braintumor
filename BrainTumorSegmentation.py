from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import cv2

main = tkinter.Tk()
main.title("Microwave Medical Image Segmentation for Brain Stroke Diagnosis: Imaging-Process-Informed Image Processing") #designing main screen
main.geometry("1000x650")


global filename, image

def loadImage():
    global filename, image
    filename = filedialog.askopenfilename(initialdir="testImages")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")

    image = cv2.imread("testImages/1.png")
    image = cv2.resize(image, (250, 250))
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
   
def OtsuSegment():
    global image
    enhanced = cv2.detailEnhance(image, sigma_s=5, sigma_r=0.05)
    gray_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY, 0.7)
    #blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    thresh = cv2.threshold(gray_img, 155, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    otsu = cv2.bitwise_and(image, image, mask=thresh)
    cv2.imshow("Original Image", image)
    cv2.imshow("OTSU Segmented Image", otsu)
    cv2.waitKey(0)

def dbim(img, sigma_value = 0.33):
    median = np.median(img)
    lower_value = int(max(0, (1.0 - sigma_value) * median))
    upper_value = int(min(255, (1.0 + sigma_value) * median))
    dbim = cv2.Canny(img, lower_value, upper_value)
    return dbim

def dbimSegment():
    global image
    enhance = cv2.detailEnhance(image, sigma_s=5, sigma_r=0.05)
    enhanced = cv2.detailEnhance(image, sigma_s=5, sigma_r=0.05)
    tumor_image = np.zeros((250, 250, 3), np.uint8)
    tumor_image[:]=(110,50,50)
    gray_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY, 0.7)
    (T_value, thresh_value) = cv2.threshold(gray_img, 170, 220, cv2.THRESH_BINARY)
    (T_value, threshInv_value) = cv2.threshold(gray_img, 170, 220,cv2.THRESH_BINARY_INV)
    kernel_value = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    process_img = cv2.morphologyEx(thresh_value, cv2.MORPH_CLOSE, kernel_value)
    process_img = cv2.erode(process_img, None, iterations = 14)
    process_img = cv2.dilate(process_img, None, iterations = 13)
    dbim_image = dbim(process_img)
    (contour, _) = cv2.findContours(dbim_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contour[0])
    roi = enhanced[y:y + h, x:x + w]
    tumor_image[y:y + h, x:x + w] = roi
    cv2.drawContours(enhanced, contour, -1, (0, 0, 255), 2)
    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", enhance)
    cv2.imshow('Tumor Image', tumor_image)
    cv2.imshow('Segmented Image', enhanced)
    cv2.waitKey(0)

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Microwave Medical Image Segmentation for Brain Stroke Diagnosis: Imaging-Process-Informed Image Processing', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Microwave Medical Image", command=loadImage)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

otsuButton = Button(main, text="OTSU Based Segmentation", command=OtsuSegment)
otsuButton.place(x=330,y=100)
otsuButton.config(font=font1) 

dbimButton = Button(main, text="Proposed DBIM Based Segmentation", command=dbimSegment)
dbimButton.place(x=620,y=100)
dbimButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=10,y=150)
exitButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
