
# Emotion Recognition

People have come to comprehend or appreciate how difficult it is to decipher human social signals based on nonverbal behaviour through the use of the internet as a medium of communication. Face-to-face interactions, rather than internet recordings or activities, are better for analysing complex behavioural patterns. This fascination with behaviour patterns is well understood in a facial expression approach, as evidenced by numerous previous research. Detecting images using images is not an easy or simple operation. To increase prediction accuracy, a lot of study has been done on this area, which has resulted in the introduction of novel designs and approaches. The Deep CNN architecture is the model that has been used in this project.



## Environment Requirements

To run this project, you will need to add certain packages to your python environment

Install packages with pip

```bash
  pip install -r requirements.txt
```




## Run Locally

Clone the project

```bash
  git clone https://github.com/pritomK78459/Emotion-Recognition
```

Go to the project directory

```bash
  cd Emotion-Recognition
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run a script

```bash
  python .\script\emotion_recognition_on_media.py --image /path/to/image
```


## Running Tests

To run tests on images, run the following command

```bash
  python .\script\emotion_recognition_on_media.py --image /path/to/image
```


## Sample Images

![App Screenshot](https://github.com/pritomK78459/Emotion-Recognition/blob/master/emotion_recognition_api/sample_images/happy.jpg)

![App Screenshot](https://github.com/pritomK78459/Emotion-Recognition/blob/master/emotion_recognition_api/sample_images/fear.jpg)

![App Screenshot](https://github.com/pritomK78459/Emotion-Recognition/blob/master/emotion_recognition_api/sample_images/surprise.jpg)
## Running Tests

To run tests on ivideos, run the following command

```bash
  python .\script\emotion_recognition_on_media.py --video /path/to/video
```


## Sample Video

![App Screenshot](https://github.com/pritomK78459/Emotion-Recognition/blob/master/emotion_recognition_api/sample_videos/result_video.gif)


## Use models

```Python
import tensorflow as tf

model = tf.keras.models.load_model('models/1')
```


## Run api server

Go to the project directory

```bash
  cd emotion_recognition_api
```

Start backend server

```bash
  python main.py
```

## Run api server through tf-serving

```bash
  cd emotion_recognition_api
```

Start backend server

```bash
  python main-tf-serving.py
```

Start tf-serving docker instance

```bash
docker run --rm -p 8501:8501 --init -v [absolute path to saved model]:[relative path to saved model] tensorflow/serving --model_name=[model name] --model_base_path=[path to model]
```

```bash
  docker run --rm -p 8501:8501 --init -v F:/emotion_recognition/emotion_recognition_api/models/1:/models/1 tensorflow/serving --model_name=emotion_recognition --model_base_path=/models/
```
## API Reference


#### Predict Image

```http
  POST /localhost:8000/predict/${image_file}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`      | `file` | **Required**. Image in which you want to run the model |

Takes an image file and returns the predicted image and confidence score


## Run Webapp

Go to the project directory

```bash
  cd fontend
```

Install dependencies

```bash
  npm install
```

Start frontend server

```bash
  npm start
```

![webapp screenshot](https://github.com/pritomK78459/Emotion-Recognition/blob/master/emotion_recognition_api/sample_images/webapp_screenshot.png)

## Authors

- [@pritomK78459](https://github.com/pritomK78459)

