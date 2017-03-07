import os
from pyten.UI import basic, auxiliary, dynamic
from Tkinter import *


class TensorUI:
    def __init__(self, parent):
        # Define basic methods and scenarios as lists and dictionaries
        self.scenarios=["Basic Tensor", "With Auxiliary Info", "Online Tensor Completion"]
        self.methods={"Basic Tensor":["Tucker(ALS)","CP(ALS)","NNCP"],"With Auxiliary Info":["AirCP","CMTF"],"Online Tensor Completion":["OnlineCP","OLSGD"]}
        self.methodsValue={"Tucker(ALS)":'1',"CP(ALS)":'2',"NNCP":'3',"AirCP":'1',"CMTF":'2',"OnlineCP":'2',"OLSGD":'3'}
        self.OPERATIONS=[("Recover", '1'),("Decompose", '2')]
        self.file_path=''
        self.selectedScenario=self.scenarios[0]
        self.selectedMethods=self.methods["Basic Tensor"]
        self.selectedMethod="Tucker(ALS)"

        # Define the top level container
        self.myContainer1 = Frame(parent)
        self.myContainer1.pack()

        #Label and Drop Down for Scenarios
        self.label1=Label(self.myContainer1,text="Select Scenario")
        self.label1.grid(row=0,column=0)
        var1 = StringVar()
        var1.set(self.scenarios[0])
        scenario_option=OptionMenu(self.myContainer1, var1, *self.scenarios, command=self.event_scenarioChange)
        scenario_option.grid(row=0,column=1)

        #Label and Dynamic Dropdown for Methods
        self.label2 = Label(self.myContainer1, text="Select Method")
        self.label2.grid(row=1, column=0)
        var2 = StringVar()
        var2.set(self.selectedMethods[0])
        self.method_option = OptionMenu(self.myContainer1, var2, *self.selectedMethods,command=self.getSelectedMethod)
        self.method_option.grid(row=1, column=1)

        # Label and select file option
        self.label3 = Label(self.myContainer1, text="Select File")
        self.label3.grid(row=2, column=0)
        self.entry = Entry(self.myContainer1, width=50, textvariable=self.file_path)
        self.entry.grid(row=2, column=1)
        self.bbutton = Button(self.myContainer1, text="Browse", command=self.browsecsv)
        self.bbutton.grid(row=2, column=2)

        # Operation Lable and Radio Buttons
        self.label4 = Label(self.myContainer1, text="Operation")
        self.label4.grid(row=3, column=0)
        self.op=StringVar()
        self.op.set('1')
        for text, mode in self.OPERATIONS:
            b = Radiobutton(self.myContainer1, text=text, variable=self.op, value=mode)
            b.grid(row=3, column=mode)


        #Submit button
        self.button1 = Button(self.myContainer1, text="Submit",command=self.submitClick)
        self.button1.grid(row=4, column=1)

    # Event Handler for Submit Button
    def submitClick(self):
        if self.selectedScenario == self.scenarios[0]:
            value='1'
            basic(self.file_path, self.methodsValue[self.selectedMethod], self.op.get())
        elif self.selectedScenario == self.scenarios[1]:
            value='2'
            auxiliary(self.file_path,self.methodsValue[self.selectedMethod], self.op.get())
        else:
            value='3'
            dynamic(self.file_path,self.methodsValue[self.selectedMethod], self.op.get())

    # Event Handler for Changing Scenarios from Dropdown
    def event_scenarioChange(self,value):
        self.selectedScenario=value
        var2 = StringVar()
        if self.selectedScenario == self.scenarios[0]:
            self.selectedMethods = self.methods[self.scenarios[0]]
        elif self.selectedScenario == self.scenarios[1]:
            self.selectedMethods = self.methods[self.scenarios[1]]
        else:
            self.selectedMethods = self.methods[self.scenarios[2]]

        var2.set(self.selectedMethods[0])
        self.method_option.grid_forget()
        self.method_option=OptionMenu(self.myContainer1, var2, *self.selectedMethods)

        self.method_option.grid(row=1, column=1)

    def browsecsv(self):
        from tkFileDialog import askopenfilename

        Tk().withdraw()
        self.filename = askopenfilename()
        self.file_path = os.path.abspath(self.filename)
        self.entry.delete(0, END)
        self.entry.insert(0, self.file_path)

    def getSelectedMethod(self, value):
        self.selectedMethod=value



root = Tk()
root.wm_title("PyTen")
myapp = TensorUI(root)
root.mainloop()
