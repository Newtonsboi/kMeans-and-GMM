import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

# ######################################## PREAMBLE #################################################
# Script uses an unsupervised algorithim, K-Means, to group points into K clusters. When script is  #
# run in the interpreter, the output will return centroid value for each cluster,and the percentage # 
# of points in each cluster for the training set. Change to kMeans(True) to run on validation.      #
# ###################################################################################################

####################################################################
#       QUESTION 2 (10,000 DATASET)        #      QUESTION 3       #
#  K    K1%     K2%    K3%    K4%   K5%    #  VALIDATION SET LOSS  #   
#  1    100                                #     12,870.10         #
#  2    50.5   49.5                        #      2,960.67         #        
#  3    38.2   23.8   38.0                 #      1,629.21         #
#  4    37.1   12.1   37.3   13.5          #      1,054.54         #
#  5    36.8   11.1   37.0    7.6   7.5    #       907.21          #
####################################################################
#                     100 DIMENSION QUESTION                       #   
#  K         TRAINING SET LOSS K MEANS    TRAINING SET LOSS MoG    #
#  5               215,509                     1,091,210           #   
#  10              215,268                       834,024           #
#  15              215,361                       834,038           #
#  20              212,945                       486,583           #
#  30              211,200                       484,477           #
####################################################################

def loadData(valid = False):
    global k, epochs, trainData, validData, num_pts, dim, trainLoss, validLoss
    k = 3 # Set number of clusters
    epochs = 500 # Set number of epochs. Set to 800 for 100D

    # trainData = np.load('data100D.npy')  # Comment either this or 2D 
    trainData = np.load('data2D.npy')
    [num_pts, dim] = np.shape(trainData)  
    trainLoss = np.full((epochs, 1), np.inf) # Define loss vector to store values of training loss

    if valid: # Split data to training and validation
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        validData = trainData[rnd_idx[:valid_batch]]
        trainData = trainData[rnd_idx[valid_batch:]] # Re-define trainData if valid is True
        [num_pts, dim] = np.shape(trainData)
        validLoss = np.full((epochs, 1), np.inf)

def buildGraph():
    tf.reset_default_graph() # Clear any previous junk
    tf.set_random_seed(45689)

    trainingInput = tf.placeholder(tf.float32, shape=(None, dim))  # Data placeholder
    centroid = tf.get_variable('mean', shape=(k,dim), initializer=tf.initializers.random_normal())  # Mean placeholder
    distanceSquared = distanceFunc(trainingInput,centroid) # Finds the euclidean norm 
    loss = tf.math.reduce_sum(tf.math.reduce_min(distanceSquared,0)) # Choose the smallest distance for each point and sum them
    optimizer =tf.train.AdamOptimizer(learning_rate= 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss) # Optimize
    return optimizer, loss,  distanceSquared, centroid, trainingInput

def kMeans(valid=False):
    loadData(valid) # Load the Data
    optimizer, loss,  distanceSquared, centroid, trainingInput = buildGraph() # Build the graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0,epochs):
            _, trainLoss[i], dist, mU = sess.run([optimizer, loss,  distanceSquared, centroid], feed_dict = {trainingInput:trainData})  

            if valid: # Find validation loss and distance if this is True
                validLoss[i],distV = sess.run([loss, distanceSquared], feed_dict = {trainingInput: validData})
        
        assign = np.argmin(dist,0) # Assign the point to the nearest cluster
        inCluster = np.mean(np.eye(k)[assign],0)  # Find the average number of points in each cluster
        plotter(valid) # Plot the Loss vs Epochs graph
        scatter(trainData, assign, mU) # Draw a 2D scatter plot. For dim = 2 only

        if valid: 
            assignV = np.argmin(distV,0) # If we are validating, use the distances of the validation points to assign the cluster
            inClusterV = np.mean(np.eye(k)[assignV],0) # Find the average number of points in each cluster
            scatter(validData, assignV, mU) # Draw a 2D scatter plot for the validation points. For dim = 2 only
            return mU, inCluster, inClusterV
    return mU, inCluster, None


def distanceFunc(X, mu): # Returns distance squared
    expandPoints = tf.expand_dims(X, 0)
    expandCentroid = tf.expand_dims(mu, 1)
    return tf.reduce_sum(tf.square(tf.subtract(expandPoints, expandCentroid)), 2)


def plotter(valid=False): # Plotting Functiom
    plt.figure(1)
    plt.cla()
    plt.title("K = %i Loss vs Epoch" % k, fontsize = 32)
    plt.ylabel("Loss", fontsize = 30)
    plt.xlabel("Epoch", fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlim((0,epochs))
    plt.grid(which = 'both', axis = 'both')

    if valid == False:
        plt.plot(trainLoss)
    else: # Normalize Plot to see how training and validation are related
        plt.plot(trainLoss/trainData.shape[0])
        plt.plot(validLoss/validData.shape[0])

def scatter(X, cluster, mU):
    if dim == 2: # Would be pretty bizarre to show 100D on 2D
        plt.figure()            
        plt.title("K = %i Scatter Plot" % k, fontsize = 32)
        plt.xlabel("$x_1$", fontsize = 30)
        plt.ylabel("$x_2$", fontsize = 30)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.scatter(X[:, 0], X[:, 1], c= cluster, s=1, cmap='viridis')
        plt.scatter(mU[:, 0], mU[:, 1], c='black', s=50, alpha=0.5)

mU, inCluster, inClusterV = kMeans(False)  # Change to True if you want to run validation. 

