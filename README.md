# CS6910_Assignment2

We have three parts of this assignment:
Part A 
->deals with building a CNN model from scratch having 5 layers and training it on inaturalist dataset to build a classifier.
Part B
->deals with transfer learning and using pretrained models like Xception, InceptionV3, InceptionresnetV2, Resnet50 to build the classifier.
Part C
->deals with object detection in real time and we have directly used a pretrained model YoloV3 to demonstrate object detection on roads/highways to reduce the number of accidents.
We have used various libraries such as numpy, keras, tensorflow and wandb to represent the metrics.

PART-A

Building a 5 layered CNN model from scratch using Keras to classify the inaturalist dataset. Since the dataset were large, we used data generators to generate data batch by batch during runtime.
We used the following sweep configurations for tuning the hyperparameters:
Method: Bayes
Metric: Maximize validation accuarcy
Dimentions of each filters(kernel_size)=[[(2,2),(2,2),(2,2),(2,2),(2,2)], [(3,3),(3,3),(3,3),(3,3),(3,3)],[(6,6),(5,5),(4,4),(3,3),(2,2)],[(2,2),(3,3),(4,4),(5,5),(6,6)]]
L2 Regularization(weight_decay)=[0, 0.0005, 0.005]
Dropout(dropout)=[0, 0.2, 0.3, 0.4]
Learning_rate=[1e-3, 1e-4]
Activation functions =['relu','selu','elu']
Batch_Normalization=[True,False]
Filters in 5 different layers=[[32,32,32,32,32],[32,64,128,256,512],[32,16,8,4,2],[512,256,128,64,32]]
augment_data=[True,False]
Batch_size=[32, 64, 128, 256]
Dense_size(Number of neurons in the dense layer)=[64, 128, 256, 512]
We trained the model with different combinations of hyperparameters using bayes method with >30 runs.
Then based on the high validation accuracy we chose 3 combinations of hyperparameters which gave >35% validation accuracy and ran again to get the final best hyperparameter combination.
Best hyperparameter configuration:
best_kernel_size= [(3,3),(3,3),(3,3),(3,3),(3,3)]
best_filters= [32,32,32,32,32]
best_weight_decay= 0
best_dropout= 0.4
best_learning_rate= 1e-3
best_batch_size= 128
best_dense_size= 64
best_batch_norm= True
best_data_augment= False
best_activation= 'selu'

Using the above hyperparameter configuration, we got the highest validation accuracy to be 39.14% after 20 epochs and on the test dataset, test accuracy of 34.73% was obtained. Out of 30 random images, the model was predicting 11-13 images correctly.
Guided Backpropagation was introduced in the conv5 layer to visualise 10 neurons with the images that excite them.

PART-B

Here we used the transfer learning, where we are using an already trained model on a larger dataset like Imagenet dataset, we are using the weights and biases of the pretrained model to build our own classifier. We have discarded the last layer since these pretrained models are supposed to classify 1000 classes.
We have used Xception, InceptionV3, InceptionresnetV2, Resnet50 models on training data to fit the model. We have added a final output layer of 10 neurons. In the training, we have used the following sweep configurations:
Method: Bayes
Metric: Maximize validation accuarcy
L2 Regularization(weight_decay):[0, 0.0005, 0.005]
Pre_trained_models: ['Inceptionv3','Inceptionresnetv2','Xception','Resnet50']
Dropout: [0, 0.2, 0.3, 0.4]
Batch_normalization: [True,False]
Augment_data: [True,False]
Batch_size: [32, 64, 128, 256]        
Dense_size: [64, 128, 256, 512]

After running for more than 20 sweep runs, we found that InceptionresnetV2 performs well for inaturalist dataset.
The best configuration came out to be:
best_weight_decay= 0
best_dropout= 0.4        
best_batch_size= 128
best_dense_size= 256        
best_batch_norm= False
best_data_augment= False
best_pre_trained_model= 'Inceptionresnetv2'        
image_size=299
With the above best configuration, test accuracy was found to be 81.70%.

PART -C

We have directly used a pretrained deep learning model called YoloV3 to perform object detection in real time. Our idea is to alert the driver of a nearby object by drawing his attention to a nearby person/car/cyclist etc. by alerting him beforehand, thus reducing the number of road accidents considerably. The demo video can be found here: https://www.youtube.com/watch?v=titUlkF_a9g
