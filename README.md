# Mood-Detection
The purpose of this python code is to be able to recognize the mood of a person from a live video stream. This could be adapted into robotics design later by allowing the robot to respond to how the people around it are feeling. For example, if a worker shows fear everytime the robot gets close then the robot could learn to give that worker more space.
Currently the code is designed to recognize anger, disgust, fear, happiness, sadness, surprise, and a neutral default. While training the data set takes time, realistically this would only need to be done once, and the live recognition is very rapid.
The dataset used is saved in a folder called 'dataset' with a separate folder for each mood (anger, disgust, fear, happy, neutral, sadness, and surprise; if you change the names of the folders or have additional folders then change the 'emotions' list at the beginning of the code). The dataset I used is the Cohn-Kanade database (Ask for access here: http://www.consortium.ri.cmu.edu/ckagree/) This code is also adapted from code shared by Paul van Gent and Adrian Rosebrock.

# References
– Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG’00), Grenoble, France, 46-53.
– Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
- van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from:
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
- Rosebrock, A. Real-time facial landmark detection with OpenCV, Python, and dlib. Retrieved from:
https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
