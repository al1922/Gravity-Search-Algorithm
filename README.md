# Gravity Search Algorithm

## General info
The project is an implementation of a gravity search algorithm.

More detail about gravity search algorithm: 
* https://www.hindawi.com/journals/amp/2017/2131862/
* https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200

## Technologies
Project is created with:
* Python version: 3.7 
* Numpy version: 1.19.3 
* MatplotLib: 3.3.3 

## Setup 
To run this project you need install libraries:
```
python -m pip install numpy
```
```
python -m pip install matplotlib
```

## Use 

```
if __name__ == "__main__":
  def function(variable):
    x = variable[0] #variables
    y = variable[1] #variables
    return x**2 + y**2  #Function
    
#Calling the algorithm: 
GravitySearchAlgorithm(function) 
```

Example of declaring more variables:<br />
```
x = variable[0]<br />
y = variable[1]<br />
z = variable[2]<br />
c = variable[3]<br />
```

If you want get a result and save it: <br />
```
algorithm = GravitySearchAlgorithm(function)
points = algorithm.Points 
score = algorithm.Score 
```

If you want to change parameters: <br />
```
  GravitySearchAlgorithm(function,is_maximization=False, num_agents=20, iterations=1000, gravity=100, beta=20 , plot = False, show_points_and_score = True)
```
