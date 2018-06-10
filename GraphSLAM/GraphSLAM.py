import numpy as np

def expand(mat,dimx, dimy, list1, list2 = []):
        if list2 == []:
            list2 = list1

        res=np.zeros((dimx, dimy))
        for i in range(len(list1)):
            for j in range(len(list2)):
                res[list1[i]][list2[j]] = mat[i][j]
        return res

def graph_slam(data, N, num_landmarks, motion_noise, measurement_noise):
    dim = 2 * (1 + num_landmarks) 

    w = np.zeros((dim,dim))
    w[0][0] = 1.0
    w[1][1] = 1.0
    
    chi = np.zeros((dim,1))
    chi[0][0] = world_size / 2.0
    chi[1][0] = world_size / 2.0
    
    for k in range(len(data)):

        measurement = data[k][0]
        motion      = data[k][1]
    
        for i in range(len(measurement)):
            m = 2 * (1 + measurement[i][0])
    
            for b in range(2):    
                w[b][b]          +=  1.0 / measurement_noise
                w[m+b][m+b]      +=  1.0 / measurement_noise
                w[b][m+b]        += -1.0 / measurement_noise
                w[m+b][b]        += -1.0 / measurement_noise
                chi[b][0]        += -measurement[i][1+b] / measurement_noise
                chi[m+b][0]      +=  measurement[i][1+b] / measurement_noise
        
        w=expand(w,dim+2, dim+2, [0,1] + list(range(4, dim+2)), [0,1] + list(range(4, dim+2)))
        chi = expand(chi,dim+2, 1, [0,1] + list(range(4, dim+2)), [0])
        
        for b in range(4):
            w[b][b] +=  1.0 / motion_noise
        for b in range(2):        
            w[b  ][b+2]        += -1.0 / motion_noise
            w[b+2][b  ]        += -1.0 / motion_noise
            chi[b  ][0]        += -motion[b] / motion_noise
            chi[b+2][0]        +=  motion[b] / motion_noise
            
        matA     = w[np.ix_([0,1],range(2,dim+2))]
        matB     = w[np.ix_([0,1],[0,1])]
        matC     = chi[np.ix_([0,1],[0])]
        wPrime   = w[np.ix_(range(2,dim+2),range(2,dim+2))]
        chiPrime = chi[np.ix_(range(2,dim+2), [0])]

        w   = wPrime - np.dot(np.dot(matA.transpose(),np.linalg.inv(matB)),matA)
        chi = chiPrime - np.dot(np.dot(matA.transpose(),np.linalg.inv(matB)), matC)
        

    mu = np.dot(np.linalg.inv(w) ,chi)

    return mu, w 

#test
import random
import math
from Robot import robot 

num_landmarks      = 5        # number of landmarks
N                  = 20       # time steps
world_size         = 100.0    # size of world
measurement_range  = 50.0     # range at which we can sense landmarks
motion_noise       = 2.0      # noise in robot motion
measurement_noise  = 2.0      # noise in the measurements
distance           = 20.0     # distance by which robot (intends to) move each iteratation 


complete = False

while not complete:

    data = []

    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)
    seen = [False for row in range(num_landmarks)]
    
    orientation = random.random() * 2.0 * math.pi
    dx = math.cos(orientation) * distance
    dy = math.sin(orientation) * distance
    
    for k in range(N-1):
    
        Z = r.sense()

        for i in range(len(Z)):
            seen[Z[i][0]] = True

        while not r.move(dx, dy):
            orientation = random.random() * 2.0 * math.pi
            dx = math.cos(orientation) * distance
            dy = math.sin(orientation) * distance
            
        data.append([Z, [dx, dy]])

    complete = (sum(seen) == num_landmarks)

print('---actial value---')
print('Landmarks: ', r.landmarks)
print(r)


mu, w  = graph_slam(data, N, num_landmarks, motion_noise, measurement_noise)      
print('---estimated value---')
#print('x=',mu[0],'y=',mu[1])    
print('Robot: [x=%.5f y=%.5f]'  % (mu[0], mu[1]))
for i in range(num_landmarks):
    print('LM',i,':x=',mu[2*(i+1)],'y=',mu[2*(i+1)+1])
