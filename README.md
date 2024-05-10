<div align="center">
<h1>GrandmAI</h1>
  <img src="https://github.com/menezess42/grandmaai/assets/67249275/4e50e215-1329-49ba-8546-44cdf48401cb" alt="grandmai" width="500">
<h3>A security system powered by AI.</h3>
<p>You can find a video demonstration <a href="https://www.youtube.com/shorts/ZmDJwb9250c">here</a></p>
<hr/>
</div>

<h2>Description</h2>
  A project that uses computer vision and artificial intelligence to detect one or more people in the video and identify their behavior, if the behavior is identified as abnormal, the system alerts.

- <h3>Technologies</h3>

  - OpenCV
  - Yolov8Pose
  - My own <a href="">dataset</a>
  - MLP
  - Cellphone

<hr/>
<h2>About</h2>
I had this idea after watching a video (couldn't find the video to link here) in which a prison guard was passing between cells and was attacked by an inmate. She was saved by the other inmates who ran to her aid. In the video, I noticed that there was a considerable amount of time between the start of the attack and the guards arriving at the scene, with the attack already prevented and the aggressive inmate already restrained by the other inmates. 
So I thought, if there was a system that could read the security camera feed, detect the type of behavior, and if it was abnormal behavior, for example, the inmate's attack, the system would emit an alert to the guards, this type of situation could be avoided.
But obtaining a dataset from prison security cameras would be a complicated and time-consuming bureaucracy, so I decided to direct the project to residential security systems. And even directing the project to residential security, I did not find a dataset on behavior, so I had to create my own which can be found on my Kaggle profile at this <a href="">link</a>.
<hr/>
<h2>Development</h2>

- <h3>Creating video dataset</h3>
  I started the project by creating the training and testing dataset. Since I didn't have the necessary funds to acquire some security cameras, I positioned my cellphone on the outer wall of my house.
  I recorded 152 videos of normal behaviors and 152 of abnormal behaviors. I considered the following behaviors abnormal:

  - Turn towards the residence and stare at it.
  - Grab the gate.
  - Try to climb the gate.
  - Garage gate:
    - Mess with the lock.
    - Try to lift it.
    - Stand in front of it.
  <p>
  I separated the raw recording into videos of around 10 seconds each, hence 152. For normal behaviors, I simply walked from one point to another on the street, entering and leaving the frame. For abnormal behaviors, I performed them timing each for 10 seconds.
  </p>

- <h3>Extracting data</h3>
  After creating the dataset of videos separated into Normal and Abnormal folders, I decided to create a numerical dataset, which is the job of the <a href="">make_dataset.py</a> code. This code is responsible for reading a video frame by frame, passing each frame through the YOLOv8-Pose AI which identifies the keypoints of the people in the frame, saving the result of each frame to a json file inside a folder. For example, reading the abnormal_1 video will create the directory ./Abnormal/abnormal_1/frame_x.json where x is the frame number.

- <h3>Creating and training the detection AI</h3>
  The machine learning model is a MLP with 3 hidden layers and 1 output layer, its architecture is tapered, the first layer has 512 neurons, the second has 256, and the third has 128. The output layer has 1 binary result neuron, where 1 indicates Abnormal and 0 indicates normal. There are dropout and batch normalization between the hidden layers, the training code is <a href="">ml_training.py</a>.

  The input takes a vector of 340 positions. You might be wondering, but if YOLOv8-Pose returns 17 key points for each person, how is the input 340?

  Well, the input is 340 because each key point is an XY coordinate. But the total count still doesn't add up, right? The total input being 340 was a choice so that the MLP could understand a movement, with all its fluidity. If I passed only one frame at a time, it would learn as if it were seeing photos. To correct this, the code takes the first 10 frames from the moment the person enters the video, these 10 frames * 17XY create a 340 input. After getting the first 10 frames, the sliding window moves one frame forward, meaning if it took from 0 to 9 on the first iteration, on the second it will take from 1 to 10. I chose the sliding window of size 10 because 10 frames are a few seconds of video, which allows me to get a good reading of the video without a noticeable delay.
<p align="center"> <img src="https://github.com/Menezess42/GrandmaAI/assets/67249275/43681e1d-ba2d-4ca9-adf3-0eada612ab35" alt="GrandmAI model" width="500"> </p>
<hr/>
<h2>GrandmAI in the real world</h2>
Well, the previous topic talks about development, simpler, the training and test videos contain only one actor (myself) performing the movements, what about GrandmAI in practice?
Tests and data collection on real-world execution are still being collected. In terms of execution, what differs from the training code to the "final" code is the fact that we use YOLOv8's track to track each individual in the video because in a real scenario it's not just one person at a time entering and leaving the frame. With this track, we save in a dictionary the frames of each tracked person, where the tracking id is the key and the value is a vector of frames, when the vector reaches len=10, an AI model is instantiated and the vector is passed through this instance. After that, the first position of the vector is discarded, and the next YOLOv8 reading for that person is awaited.
