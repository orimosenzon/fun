import random 

# ground truth 
WX, WY, DD = 30, 60 , 10 # WX * x + WY * y = DD

def linear_sep(wx, x, wy, y): 
    return wx * x + wy *y 

def train(steps=1000):
    wx, wy = 0, 0

    for step in range(steps):
        x, y = random.randint(0,100), random.randint(0,100)
        distance = linear_sep(WX, x, WY, y) - DD
        dd = linear_sep(wx, x, wy, y)  

        ... 