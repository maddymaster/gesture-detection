# gesture-detection
Gesture detection using RNN
tep1: After importing we quickly check the data, the images, the folders. Its important to know what we are dealing with.

Step2: Generators! 
This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, you are going to preprocess the images as you have images of 2 different dimensions as well as create a batch of video frames. You have to experiment with `img_idx`, `y`,`z` and normalization such that you get high accuracy.

They help with yield option that helps us keep going back without exiting the function which you do with a return. Hence Generators are like heavily used, especially when you are training or building a pipeline for training. Using the basic generator function than using a keras which come with their own inbuilt generators that masks all the hardship and learning underneath, we built our own custom awesome generator that will feed the frames.

Inside our generator, we have a few samples, lets say 23 and we pick a batch size of 10. That will get divided into Batch1: 10, Batch2: 10 and Batch3: 3.

We then read the video as many number of frames, so we sort the frames to retain the flow of the temporals.

The function given in the starter doc helps 


Step3: Reading Video as Frames

Note that in our project, each gesture is a broken into indivdual frame. Each gesture consists of 30 individual frames. While loading this data via the generator there is need to sort the frames if we want to maintain the temporal information.

The order of the images loaded might be random and so it is necessary to apply sort on the list of files before reading each frame.

Step 4: Model building & Implementation and a lot of reading the documentation! 

3D Convolutional Network, or Conv3D

Conv3D convolution layer (e.g. spatial convolution over volumes).

This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally,if activation is not None, it is applied to the outputs as well.

When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 128, 1) for 128x128x128 volumes with a single channel, in data_format="channels_last".

Step 5:  Converting 2D to 3D and model building and experimenting
To use 2D convolutions, we first convert every image into a 3D shape : width, height, channels. 
Apart from the width and height, channels refer to the RGB layers (Red Green and Blue), and hence it gets set as 3. When we talk about 4D according to documentation, length, breadth, height and channel (r/g/b) gets added and so on. This is so awesome! The time we start to begin to feel like superman! 

Lets create the model architecture. The architecture is described below:
We tried many filters, batches, and some took time so had to reduce with even 3,3 filters. 

Then we used Adam optimizer with its default settings. And as per instructions ReduceLROnPlateau has been used to reduce the learning alpha after the epochs result plateauing.

Then with dense perceptrons also we experimented

Step 6: final model
After  the final model building, we are all set! We achieved an accuracy of over 80% and have achieved the objective
