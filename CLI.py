class frontend:
    def __init__(self):
        import Main as p 
        import numpy as np
        self.np = np
        self.model = p
        self.listnames = ['cylinders','displacement','horsepower','weight','acceleration','model_year','origin']
        self.asker()
    def asker(self):
        while True:
            try:
                x = int(input('''What do you want to do :-
                                 1)Enter 1 for making the prediction 
                                 2)Enter 2 for Getting cross_val score of our model
                                 3)Enter any other number for exiting \n'''))
            except Exception as e:
                print("Please enter a suitable number!!!")
                print("\n")
                self.asker()
            
            if(x!=1 and x!=2):
                print("Have a nice day!!")
                exit()
            elif(x==2):
                self.crossval()
            else:
                self.predictor()
    
    def crossval(self):
        print(self.model.cross)
        print("\n")
    
    def predictor(self):
        inputs = []
        for i in self.listnames:
            try:
                y = float(input(f"Enter the value for {i} of your vehicle : "))
            except Exception as e:
                print("You didnt enter a valid value!! Try again")
                print("\n")
                self.predictor()
            inputs.append(y)
        inputarray = self.np.array(inputs).reshape(1,len(inputs))
        print("\n")
        print(f"The approximated value of Miles-per-Gallon according to your data is {self.model.mainpipe.predict(inputarray)}")
        print("\n")
        
        
        
model = frontend()
