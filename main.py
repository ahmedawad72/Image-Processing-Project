import tkinter
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
import PIL
from PIL import Image, ImageTk
from PIL import ImageFilter
import numpy as np
import cv2 as cv
from numpy import asarray

root = Tk()
root.title("Image Processing")
# set full window
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.wm_maxsize(height=750, width=1355)

# --------------------------- welcome label and exit button --------------------------------#

welcome_label = Label(root, text="Image Processing", height=3, width=80, font=('Helvetica', 24), bg='gray50', )
welcome_label.place(x=-90, y=0)
# myFont = font.Font(family='Helvetica')
# welcome_label.pack()

Exit_btn = Button(root, text="Exit", font=('Times', 16), activebackground='red', command=root.quit)
Exit_btn.place(x=1290, y=74)
Exit_btn.config(height=1,width=5)

flag=False
Save_btn = Button(root, text="Save Picture", font=('Times', 16), activebackground='green', command= lambda:save() )
Save_btn.place(x=200, y=74)
Save_btn.config(height=1,width=12)

def save():
    global flag
    flag=True

# --------------------------------------UPlOADING PICTURE--------------------------------------------------------------#
# # choosing picture Button
# frame_height=626
# frame_width=1365
# pic_frame = Frame(root, width=frame_width, height=frame_height).place(x=0, y=113)
# choose_pic = Button(root, text="choose picture", font=('Times', 16), activebackground='green', command=lambda: choose())
# choose_pic.place(x=0, y=74)
# choose_pic.config(height=1,width=12)
# # Choosing Picture Method
#
# check = False
# IMG=PIL.Image.open("background.jpg")   #GLOBAL VARIABLE TO STORE IMG
# def choose():
#     f_types = [('JPG files', '*.jpg'), ('PNG files', '*.png')]
#     filename = filedialog.askopenfilename(filetypes=f_types)
#     img = Image.open(filename)
#     img = img.resize((frame_width, frame_height))
#     img = ImageTk.PhotoImage(img)
#     e1 = Label(pic_frame)
#     e1.place(x=200, y=113)
#     e1.image = img
#     e1['image'] = img
#
# ---------------------------------------------------------------------------------------------------------------------#

def upload_img():
    f_types = [('JPG files', '*.jpg'), ('PNG files', '*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    return filename



def save_img(img):
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    edge = Image.fromarray(img)
    edge.save(filename)
    tk_edge = ImageTk.PhotoImage(edge)
    label = tkinter.Label(root, image=tk_edge)
    label.place(x=100, y=100)
    global flag
    flag = False

# -----------------------------------------------BUTTON MENU-----------------------------------------------------------#
idendity = Button(root,text="Idendity Transformation", font=('Times', 14),activebackground='green',command=lambda :Idendity_Transformation())
idendity.place(x=0, y=170)
idendity.config(height=2,width=18)

def Idendity_Transformation():  # black and whight
    fileName=upload_img()
    original = cv.imread(fileName)
    img = cv.imread(fileName,0)
    cv.imshow("before", original)
    cv.imshow("after", img)
    cv.waitKey()
    global flag
    if flag:
        save_img(img)
    cv.destroyAllWindows()
# ---------------------------------------------------------------------------------------------------------------------#

log = Button(root,text="Log Transformation", font=('Times', 14), activebackground='green',command=lambda:Log_Transformation())
log.place(x=0, y=222)
log.config(height=2,width=18)

                           # DONEEE
def Log_Transformation():
    fileName=upload_img()   #path
    original = cv.imread(fileName)
    c = 255 / np.log(1+ np.max(original))
    log_image = c * (np.log(original + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    cv.imshow("before", original)
    cv.imshow("after",log_image)
    cv.waitKey()
    global flag
    if flag:
        save_img(log_image)
    cv.destroyAllWindows()
# ---------------------------------------------------------------------------------------------------------------------#

negative = Button(root,text="Negative Transformation", font=('Times', 14), activebackground='green',command=lambda:Negative_Transformation())
negative.place(x=0, y=274)
negative.config(height=2,width=18)

            #DONEEEEEEEEE
def Negative_Transformation():                                               #RRRRGGGGGGBBBBB
    fileName = upload_img()
    original = cv.imread(fileName)
    img_neg = 255 - original
    cv.imshow("before", original)
    cv.imshow("after", img_neg)
    cv.waitKey()
    global flag
    if flag:
        save_img(img_neg)
    cv.destroyAllWindows()
      #
      # fileName = upload_img()
      # original = cv.imread(fileName)
      # img= cv.imread(fileName,0)
          # h,w=img.shape
      # for row in range(h):
      #     for col in range(w):
      #         img[row][col] = 255 - img[row][col]
      # cv.imshow("before", original)
      # cv.imshow("after", img)
      # cv.waitKey()
      # global flag
      # if flag:
      # save_img(img)
      # cv.destroyAllWindows()

#---------------------------------------------------------------------------------------------------------------------#

contrast = Button(root,text="Change Contrast", font=('Times', 14), activebackground = 'green',command=lambda:Change_Contrast())
contrast.place(x=0, y=326)
contrast.config(height=2,width=18)

def Change_Contrast():
    fileName = upload_img()
    original = cv.imread(fileName)
    img = cv.imread(fileName,0)
    h, w = img.shape
    a = np.min(img)
    b = np.max(img)
    R = b - a
    for row in range(h):
        for col in range(w):
            img[row][col] = ((img[row][col] - a) / R) * 255
            img[row][col] = np.rint(img[row][col])
    cv.imshow("before", original)
    cv.imshow("after", img)
    cv.waitKey()
    global flag
    if flag:
        save_img(img)
    cv.destroyAllWindows()
# ---------------------------------------------------------------------------------------------------------------------#

power = Button(root,text="Power Law", font=('Times', 14), activebackground = 'green',command=lambda:Power_law_transformation())
power.place(x=0, y=378)
power.config(height=2,width=18)

def Power_law_transformation():
    fileName = upload_img()
    original = cv.imread(fileName)
    img= cv.imread(fileName)
    img=img/255.0
    im_power_law_transformation = cv.pow(img, 0.6)
    cv.imshow("before", original)
    cv.imshow("after", im_power_law_transformation)
    cv.waitKey()
    global flag
    if flag:
        save_img(img)
    cv.destroyAllWindows()
    # f_types = [('JPG files', '*.jpg'), ('PNG files', '*.png')]

#   filename = filedialog.askopenfilename(filetypes=f_types)
#   original=cv.imread(filename)
#   img= cv.imread(filename,0)
#   h,w=img.shape
#   for row in range(h):
#       for col in range(w):
#           img[row][col] = 255 * (img[row][col] / 255) ** 0.5
#   cv.imshow("before", original)
#   cv.imshow("after", img)
#   cv.waitKey()
#   cv.destroyAllWindows()


# --------------------------------------------------------------------------------------------------#

median = Button(root,text="Median Filter", font=('Times', 14), activebackground='green',command=lambda :Median_Filter())
median.place(x=0, y=430)
median.config(height=2,width=18)

def Median_Filter():
    fileName = upload_img()
    original = cv.imread(fileName)
    median = cv.medianBlur(original, 9)  # remove salt and pepper noise
    cv.imshow("before", original)
    cv.imshow("after", median)
    cv.waitKey()
    global flag
    if flag:
        save_img(median)
    cv.destroyAllWindows()

# --------------------------------------------------------------------------------------------------#

min = Button(root,text="Min Filter", font=('Times', 14), activebackground='green',command=lambda :Min_Filter())
min.place(x=0, y=482)
min.config(height=2,width=18)

def Min_Filter():
    # original = upload_img()
    # original=convert_from_cv2_to_image(original)
    # min = original.filter(ImageFilter.MinFilter(size=5))
    # original.show()
    # min.show()
    # cv.waitKey()
    # global flag
    # if flag:
    #     save_img(min)
    # cv.destroyAllWindows()
    original = Image.open(r"C:\Users\AHMED\Pictures/min max.jpg")
    min = original.filter(ImageFilter.MinFilter(size=5))
    original.show()
    min.show()
    cv.waitKey()
    global flag
    if flag:
        save_img(min)
    cv.destroyAllWindows()


# --------------------------------------------------------------------------------------------------#

max= Button(root,text="Max Filter", font=('Times', 14), activebackground='green',command=lambda :Max_Filter())
max.place(x=0, y=534)
max.config(height=2,width=18)

def Max_Filter():
    original = Image.open(r"C:\Users\AHMED\Pictures/min max.jpg")
    max = original.filter(ImageFilter.MaxFilter(size=5))
    original.show()
    max.show()
    cv.waitKey()
    global flag
    if flag:
        save_img(max)
    cv.destroyAllWindows()

# --------------------------------------------------------------------------------------------------#

laplace= Button(root,text="Laplacian Filter", font=('Times', 14), activebackground='green',command=lambda :Laplacian_Filter())
laplace.place(x=0, y=586)
laplace.config(height=2,width=18)

def Laplacian_Filter():
    fileName = upload_img()
    original = cv.imread(fileName)
    laplace = cv.Laplacian(original, cv.CV_64F, ksize=3)
    kernelArray = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplacianImageScratch = cv.filter2D(src=original, ddepth=-1, kernel=kernelArray)
    laplace2 = np.uint8(np.absolute(laplace))             # Normalise the laplace image.
    cv.imshow("original", original)
    cv.imshow("laplacian", laplace)
    cv.imshow("normalized", laplace2)
    cv.imshow("laplacian scratch", laplacianImageScratch)
    cv.waitKey()
    global flag
    if flag:
        save_img(laplacianImageScratch)
        save_img(laplace)
        save_img(laplace2)
    cv.destroyAllWindows()

# --------------------------------------------------------------------------------------------------#

gaussian= Button(root,text="Gaussian Filter", font=('Times', 14), activebackground='green',command=lambda :Gaussian_Filter())
gaussian.place(x=0, y=638)
gaussian.config(height=2,width=18)

def Gaussian_Filter():
    fileName = upload_img()
    original = cv.imread(fileName)
    gaussian_blur = cv.GaussianBlur(original,(1,1),0)                   # First step : Apply Gaussian Blur
    LoG_image = cv.Laplacian(gaussian_blur, cv.CV_64F, ksize=3)  #Apply Laplace function
    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    img_prewittx = cv.filter2D(gaussian_blur, -1, kernelx)
    img_prewitty = cv.filter2D(gaussian_blur, -1, kernely)
    sobelx = cv.Sobel(original, cv.CV_64F, 1, 0, ksize=3)  # x
    sobely = cv.Sobel(original, cv.CV_64F, 0, 1, ksize=3)  # y.
    sobel = sobelx + sobely
    cv.imshow("gaussian x", img_prewittx)
    cv.imshow("gaussian y", img_prewitty)
    cv.waitKey()
    global flag
    if flag:
        save_img(img_prewittx)
        save_img(img_prewitty)
    cv.destroyAllWindows()

# --------------------------------------------------------------------------------------------------#

# set window color
root.configure(bg='white')
root.mainloop()


