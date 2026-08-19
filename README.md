# Description
This is a Fastapi app for identifying closest pose match between uploaded image and indexed database.
Currently using Google's mediapipe pose landmark detection model. Best matches are identified through the Euclidean distance between extracted pose vectors.

## Goal
Theoretically this app can be used for any generic pose detection case. My personal interest is to use this tool to identify the most closely matching pose from the Manga JoJo's Bizzare Adventure by Hirohiko Araki. Araki is known for his use of dynamic and evocative poses in his artwork. Being able to compare the similarity of live photographs to this distinctive art direction will be a challenging computer vision task.


# TO DO
- Implement pose detection model that can better handle poses present in drawn or artistic images
- Implement effective manga panel segmentation algorithm to extract individual panels from the Manga volumes of choice and populate the database
- Implement detection for multiple subjects within an image, currently mediapipe can only handle single persons.
- Add a "suggest pose" page which will return a pose image from the database for the user to replicate.
- Allow suggest pose page to filter for "number of people".
