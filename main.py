from sys import exit
import numpy as np
from keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from time import sleep
import tkinter.messagebox as messagebox
from tkinter import *
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
global draw_area, root, model
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def terminated_program():
    exit()


def paint(event):
    x = event.x
    y = event.y
    rad = 8
    x_low = x - rad
    x_up = x + rad
    y_low = y - rad
    y_up = y + rad
    draw_area.create_oval(x_low, y_low, x_up, y_up,
                          fill='black', outline='black')


def preparing_to_neural():
    global root
    if os.path.exists("digit.ps"):
        os.remove("digit.ps")
    if os.path.exists("digit.jpg"):
        os.remove("digit.jpg")

    draw_area.postscript(file="digit.ps")
    inp = os.path.abspath('digit.ps').replace('\\', '/')
    out = inp[:len(inp) - 3] + '.jpg'
    cmd = fr"{os.getcwd()}\ImageMagick\magick.exe {inp} {out}"
    Popen(cmd, stdout=PIPE, stderr=STDOUT)
    while not os.path.exists("digit.jpg"):
        sleep(0.1)
    os.remove(fr"{os.getcwd()}\digit.ps")

    print("Saved")


def win_predict(predict):
    messagebox.showinfo("Нейросеть", f"Нейросеть считает, что это цифра {predict}")


def run_neural():
    preparing_to_neural()
    res = get_predict()
    win_predict(res)
    root.destroy()


def get_interface():
    global root
    root = Tk()
    root.title("Paint")
    root.geometry('245x270+700+10')
    # запрещаем изменять размер окна
    root.resizable(False, False)
    root.protocol('WM_DELETE_WINDOW', terminated_program)

    # создание холста
    global draw_area
    draw_area = Canvas(root, width=240, height=240, bg='white')
    draw_area.bind("<B1-Motion>", paint)
    draw_area.grid(row=2, column=0, columnspan=7)

    check_neural = Button(text='Распознать', command=run_neural)
    check_neural.grid(row=0, column=2)
    delete_all = Button(text='Стереть всё', command=lambda: draw_area.delete("all"))
    delete_all.grid(row=0, column=4)

    root.mainloop()


def create_network():
    global model
    if os.path.exists("snn.h"):
        model = load_model("snn.h5")
        return model
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # стандартизация входных данных
        x_train = x_train / 255
        x_test = x_test / 255

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # создание нейронной сети
        input_shape = (28, 28, 1)
        model = Sequential([
            Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2), strides=2),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            MaxPooling2D((2, 2), strides=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.3),
            Dense(30, activation='relu'),
            Dropout(0.1),
            Dense(10, activation='softmax')
        ])
        model.summary()

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        his = model.fit(x_train, y_train, batch_size=16, epochs=6, validation_split=0.2, shuffle=0.2)
        plt.plot(his.history['loss'])
        plt.plot(his.history['val_loss'])
        plt.show()

        model.evaluate(x_test, y_test)
        model.save("snn.h5")
        return model


def get_predict():
    img = Image.open("digit.jpg").resize((28, 28), Image.ANTIALIAS)
    img = np.asarray(img.convert('L'))
    img = 1 - (img.reshape(784) / 255.0)
    img = np.expand_dims(img.reshape(28, 28), axis=0)

    res = model.predict(img)
    res = np.argmax(res)

    os.remove(fr"{os.getcwd()}\digit.jpg")
    return res


if __name__ == '__main__':
    model = create_network()
    while True:
        get_interface()
