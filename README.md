# age_detector
An age detector of a person in a photo

### English
## IMAGE CLASSIFIER FOR THE DETECTION OF A PERSON'S AGE
By Luis Miguel García Marín

This python 3 project was done in order to test my skills learned in the Nvidia course "Fundamentals of Deep Learning".

The content of this repository includes test images such as "leonor-15.jpg" and "elderly-90.jpg" and the notebook where the neural network is developed "age_detector.pynb" (a Spanish version is also included as "detector_de_edad.pynb").

The neural network that is developed in the notebook "age_detector.pynb" is trained for detecting the age of a person that is passed as an image at the input of the network.
The image bank used together with its labels is already treated to be worked as csv file and can be downloaded at: https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv

For the correct reproduction of the training of this network and then to be able to put it to the test, you will need to download the "age_gender.csv" dataset specified in the previous link (190 MB are required). One way to execute this is by uploading the notebook "age_detector.pynb" to the cloud, such as Colab or Amazon Web Services. We must also upload the "age_gender.csv" dataset and separate images to make predictions, such as "leonor-15.jpg" (photo of Princess Leonor at 15 years old) and "anciana-90.jpg" (photo of a 90-year-old woman). The directory I chose to upload so much the dataset like the images was inside the "data" folder.

Once all the preparations have been made, the execution of the steps can begin.
The steps of what is done in the notebook are developed below:

1. We import the necessary libraries for the development of our neural network.
2. We load our dataset of 27305 48x48 pixel facial images with their labels, in
our case in .csv format. The dataset can be downloaded at: https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv
3. We look at the first 5 rows (image with labels) of the dataset.
4. As we explore the data, we see that the pixels are expressed as strings separated by
spaces. To handle this data better, we are going to convert it into an array of numbers, helping us
of a lambda function, which uses the x.split () functions (to separate the elements for each
space) and np.array () (to construct the array, with float number format of precision 32).
5. We see that now the pixels are a numerical array.
6. We now preview about 20 images of the dataset, accompanied by all the labels.
However, we will only use the 'age' label.
7. We obtain the pixels in x, converting them into a tuple to be able to access correctly
to the .shape attribute
8. We normalize the pixels so that the model can work better with floating values between 0 and 1.
Knowing that the maximum value of a pixel is 255.
9. In order to have information about the nearby pixels and to be able to carry out convolution, we now carry out
a reshape to the pixels to go from working with one dimension to three dimensions (width in
pixels, pixel height and number of color channels). In this way, the entrance of our
neural network will also have these dimensions.
10. We get in and age labels. Since there is not a total number of classes
predefined (the maximum age of a person does not have a strict limit), it does not suit us
categorize.
11. Using the train_test_split function, we now partition the entire dataset into train
and test.
12. We build the neural network model, in this case based a little on that of the
booklet 3. The input must be 3-dimensional as specified above (48,48,1).
However, we want the output to be a real number that indicates the predicted age of the
person in the photo, so the output will be a single unit with relu activation, since there is no
as other times we have a specific number of categories among which to distribute the percentages
solution.
13. We summarize the model.
14. In the compilation of the model we indicate to use 'adam' (method of the descent of the gradient
stochastic) as an optimizer, the root mean square error as a loss function and the error
mean absolute as metrics.
15. We proceed to carry out the training of the model, of about 20 epochs.
16. We create the make_predictions function to make separate predictions and test
our model.
17. We finally create the age_detector function, which makes use of the make_predictions function
and presents the result to us in a more legible way.
18. We clean the memory, in case we need it.
