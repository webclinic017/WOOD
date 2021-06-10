# root.configure(background='#1E90FF')
# TICKER_LIST = ['EUR/USD','EUR/AUD','GBP/AUD','GBP/NZD','GBP/CAD','GBP/JPY','USD/CAD','USD/JPY']
# PERIOD_LIST = ['m1','m5','m15','m30','H1','H4','D1']

import tkinter as tk
import tkinter.font as tkFont

class App:
    

    def __init__(self, root):
        #setting title
        root.title("Go!em FX")
        #setting window size
        width=600
        height=500
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.configure(background='#1E90FF')
        root.resizable(width=False, height=False)

        GListBox_642=tk.Listbox(root)
        GListBox_642["bg"] = "#ffffff"
        GListBox_642["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times',size=10)
        GListBox_642["font"] = ft
        GListBox_642["fg"] = "#1e90ff"
        GListBox_642["justify"] = "center"
        GListBox_642["relief"] = "ridge"
        GListBox_642.place(x=10,y=10,width=80,height=25)
        # GListBox_642["listvariable"] = ['EUR/USD','EUR/AUD','GBP/AUD','GBP/NZD','GBP/CAD','GBP/JPY','USD/CAD','USD/JPY']
        GListBox_642.insert(0, "EUR/USD")
        GListBox_642.insert(1, "EUR/AUD")
        GListBox_642.insert(2, "GBP/NZD")
        GListBox_642.insert(3, "GBP/AUD")
        GListBox_642.insert(4, "GBP/CAD")
        GListBox_642.insert(5, "GBP/JPY")
        GListBox_642.insert(6, "USD/CAD")
        GListBox_642.insert(7, "USD/JPY")

        GListBox_783=tk.Listbox(root)
        GListBox_783["bg"] = "#ffffff"
        GListBox_783["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times',size=10)
        GListBox_783["font"] = ft
        GListBox_783["fg"] = "#1e90ff"
        GListBox_783["justify"] = "center"
        GListBox_783.place(x=110,y=10,width=80,height=25)
        # GListBox_783["listvariable"] = ['m1','m5','m15','m30','H1','H4','D1']
        GListBox_783.insert(0, "m1")
        GListBox_783.insert(1, "m5")
        GListBox_783.insert(2, "m15")
        GListBox_783.insert(3, "m30")
        GListBox_783.insert(4, "H1")
        GListBox_783.insert(5, "H4")
        GListBox_783.insert(6, "D1")

        GButton_532=tk.Button(root)
        GButton_532["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_532["font"] = ft
        GButton_532["fg"] = "#000000"
        GButton_532["justify"] = "center"
        GButton_532["text"] = "Init Base"
        GButton_532.place(x=10,y=100,width=70,height=25)
        GButton_532["command"] = self.GButton_532_command

        GButton_460=tk.Button(root)
        GButton_460["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_460["font"] = ft
        GButton_460["fg"] = "#000000"
        GButton_460["justify"] = "center"
        GButton_460["text"] = "Backtest"
        GButton_460.place(x=10,y=220,width=70,height=25)
        GButton_460["command"] = self.GButton_460_command

        GButton_389=tk.Button(root)
        GButton_389["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_389["font"] = ft
        GButton_389["fg"] = "#000000"
        GButton_389["justify"] = "center"
        GButton_389["text"] = "Learning"
        GButton_389.place(x=10,y=140,width=70,height=25)
        GButton_389["command"] = self.GButton_389_command

        GButton_359=tk.Button(root)
        GButton_359["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_359["font"] = ft
        GButton_359["fg"] = "#000000"
        GButton_359["justify"] = "center"
        GButton_359["text"] = "Live"
        GButton_359.place(x=10,y=180,width=70,height=25)
        GButton_359["command"] = self.GButton_359_command

        GButton_92=tk.Button(root)
        GButton_92["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_92["font"] = ft
        GButton_92["fg"] = "#000000"
        GButton_92["justify"] = "center"
        GButton_92["text"] = "Verification"
        GButton_92.place(x=10,y=260,width=70,height=25)
        GButton_92["command"] = self.GButton_92_command

        '''GRadio_435=tk.Radiobutton(root)
        ft = tkFont.Font(family='Times',size=10)
        GRadio_435["font"] = ft
        GRadio_435["fg"] = "#333333"
        GRadio_435["justify"] = "center"
        GRadio_435["text"] = " Save Model"
        GRadio_435.place(x=90,y=140,width=85,height=25)
        GRadio_435["command"] = self.GRadio_435_command'''



    def GButton_532_command(self):
        print("command")


    def GButton_460_command(self):
        print("command")


    def GButton_389_command(self):
        print("command")


    def GButton_359_command(self):
        print("command")


    def GButton_92_command(self):
        print("command")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

