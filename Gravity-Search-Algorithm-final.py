import numpy as np 
import matplotlib.pyplot as plt
from numpy.random import rand, randn, randint
from numpy.linalg import norm

class GravitySearchAlgorithm(object):
    def __init__(self, function, is_maximization=False, num_agents=20, iterations=1000, gravity=100, 
                 beta=20 , plot = False, show_points_and_score = True):
        self.function_level = function.__code__.co_nlocals
        self.num_of_agents = num_agents #Setting the number of agetts
        self.masses = np.zeros([self.num_of_agents, 1]) #Creating an empty mass matrix for each agent
        self.pos = np.array(randint(-10, 10, [self.num_of_agents, self.function_level]))#Drawing of starting positions
        self.velocity = np.zeros([self.num_of_agents, self.function_level]) #Creating an empty velocity matrix for each agent
        self.acceleration = np.zeros([self.num_of_agents, self.function_level])#Creating an empty acceleration matrix for each agent
        self.force = np.zeros([self.num_of_agents, self.function_level])#Creating an empty force matrix for each agent
        self.evaluation_func_value = np.zeros(self.num_of_agents) #Creating an empty scoring matrix
        self.G = 0 # initializing constant G. It will be change later in function update_G
        self.grav = gravity #Starting value  of gravitational constant
        self.current_iteration = 1
        self.num_of_iterations = iterations #Number of iterations
        self.worst_agents = np.zeros(self.num_of_agents) #Number of the worst agents
        self.n_of_worst_agents = 0
        self.number_of_bad_agents = self.num_of_agents - np.floor(self.num_of_agents/10)
        self.is_maximization = is_maximization #Declaration of minimum or maximum
        self.optimization_function = function #Function 
        self.beta = beta 
        self.plot = plot
        self.Points = np.zeros(self.function_level)
        self.Score = 0
        self.show = show_points_and_score
        self.start() #Starting the algorithm
        
    def physics(self):
        self.acceleration = self.force / self.masses #Calculation acceleration
        self.velocity = rand() * self.velocity + self.acceleration #Calculation velocity
        self.pos = self.pos + self.velocity #Calculation positions

    

    def forces_between_objects(self, obj1, obj2, random_value): #Calculation forces between agents
        self.force[obj1] += random_value * (((self.G * self.masses[obj1] * self.masses[obj2]) *
                                             (self.pos[obj2] - self.pos[obj1])) / (
                                                norm(self.pos[obj1] - self.pos[obj2]) + rand())) 

    def gravity(self): #Determining the gravity of agents
        self.force = np.zeros([self.num_of_agents, self.function_level]) #Setting all forces to zero
        for obj1 in range(self.num_of_agents):
            random_value = rand()
            for obj2 in range(self.num_of_agents):
                if obj1 is not obj2 or self.worst_agents[obj1] is True:
                    self.forces_between_objects(obj1, obj2, random_value)
                    
    def worst_agents_over_time(self): #Increasing the number of bad agents and their selection
        if self.current_iteration % (np.floor(self.num_of_iterations / (self.number_of_bad_agents - 1))) == 0:
            self.worst_agents = np.argsort(self.evaluation_func_value) >= self.n_of_worst_agents
            self.n_of_worst_agents += 1

    def evaluation_function(self):
        temp_masses = np.ones(self.num_of_agents) #Creating a temporary mass
        self.evaluation_func_value = list(map(self.optimization_function,self.pos[:,:(self.function_level-1)])) #Calculation of the function value for each agent

        best, worst = self.get_best_worst() #Selection of the worst and higher value of the function

        for obj in range(self.num_of_agents): #Calculation of new mass for agents
            if self.evaluation_func_value[obj] != 0 and self.evaluation_func_value[obj] != worst:
                temp_masses[obj] = (self.evaluation_func_value[obj] - worst) / (best - worst)

        for obj in range(self.num_of_agents): #Change to new mass
            self.masses[obj] = temp_masses[obj] / np.sum(temp_masses)

    def get_best_worst(self): #Choosing the best and worst function value
        if not self.is_maximization: #For maximization
            best = np.min(self.evaluation_func_value) #Calculation of the best function value
            worst = np.max(self.evaluation_func_value) #Calculation of the worst function value
        else: #For minimalization
            best = np.max(self.evaluation_func_value) #Calculation of the best function value
            worst = np.min(self.evaluation_func_value) #Calculation of the worst function value
        return best, worst

    def update_G(self): #Update of the gravitational force
        self.G = self.grav *np.exp(-self.beta*(self.current_iteration / self.num_of_iterations))
        self.current_iteration += 1

    def start(self): #The function that starts the algorithm
        if self.is_maximization == False and self.show == True:
            print("Function minimalization - please wait ")
        elif self.is_maximization == True and self.show == True:
            print("Function maximalization - please wait ")
        
        update_plot = 5
        plt.figure()

        while self.num_of_iterations > self.current_iteration:
            #Calling all functions declared above
            self.evaluation_function()
            self.gravity()
            self.physics()
            self.update_G()
            
            #The code below is for plotting the algorithm's progression
            #Plotting  works of course only for functions of 2 or 3 dimentions. 
            if self.function_level == 3 and self.plot == True:
                plt.clf()
                if update_plot == 5 :
                    frame_x = np.array([np.max(self.pos[:,0]),np.min(self.pos[:,0])])
                    frame_y = np.array([np.max(self.pos[:,1]),np.min(self.pos[:,1])])
                    x = np.linspace(-5+frame_x[1],5+frame_x[0])
                    y = np.linspace(-5+frame_y[1],5+frame_y[0])
                    X, Y = np.meshgrid(x, y)
                    update_plot = 0

                update_plot += 1
                plt.contour(X, Y, self.optimization_function([X,Y]) )
                plt.scatter(self.pos[:,0], self.pos[:,1],color='red')
                plt.axis([-5+frame_x[1],5+frame_x[0], -5+frame_y[1],5+frame_y[0]])
                plt.pause(0.01)
                
            if self.function_level == 2 and self.plot == True :
                plt.clf()
                if update_plot == 5 :
                    frame_x = np.array([np.max(self.pos[:,0]),np.min(self.pos[:,0])])
                    frame_y = np.array([np.max(self.optimization_function([self.pos[:,0]])),np.min( self.optimization_function([self.pos[:,0]]))])
                    x = np.linspace(-5+frame_x[1],5+frame_x[0],500)
                    update_plot = 0
                update_plot += 1

                plt.plot(x, self.optimization_function([x]) , color = 'black',linewidth=2)
                plt.scatter(self.pos[:,0], self.optimization_function([self.pos[:,0]]),color='blue', s= 50)
                plt.axis([-5+frame_x[1],5+frame_x[0], -5+frame_y[1],5+frame_y[0]])
                plt.pause(0.01)

        if (self.function_level == 3 or self.function_level == 2) and self.plot == True :
            plt.show()            
            
        #Displaying and saving results
        if self.is_maximization == False : 
            self.Score = np.min(list(map(self.optimization_function,self.pos[:,:(self.function_level-1)])))
            self.Points = self.pos[np.argmin(list(map(self.optimization_function,self.pos[:,:(self.function_level-1)])))][:(self.function_level-1)]
            if self.show == True:
              print("Score: " + str(self.Score))
              print("Points: " + str(self.Points))
        else:
            self.Score = np.max(list(map(self.optimization_function,self.pos[:,:(self.function_level-1)])))
            self.Points = self.pos[np.argmax(list(map(self.optimization_function,self.pos[:,:(self.function_level-1)])))][:(self.function_level-1)]
            if self.show == True:
              print("Score: " + str(self.Score))
              print("Points: " + str(self.Points))

if __name__ == "__main__":
    def function(variable):
    #Declaration of variables
        x = variable[0]
        y = variable[1]
    
    #Example of declaring more variables
    #  x = variable[0]
    #  y = variable[1]
    #  z = variable[2]
    #  c = variable[3]
    
        return x**2 + y**2  #Function

    #Calling the algorithm
    GravitySearchAlgorithm(function) 

    #If you want get a result and save it:
    # algorithm = GravitySearchAlgorithm(function) 
    # points = algorithm.Points
    # score = algorithm.Score

    # If you want to change parameters:
    # GravitySearchAlgorithm(function,is_maximization=False, num_agents=20, iterations=1000, gravity=100, beta=20 , plot = False, show_points_and_score = True)
