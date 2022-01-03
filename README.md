### Animal Sound Classification (Cats Vrs Dogs Audio Sentiment Classification)

This is a simple audio classification `api` build to classify the sound of an audio, weather it is the `cat` or `dog` sound.

<img src="images/dog-cats.jpg" width="100%" alt="alt"/>

### Response

Given a `.wav` audio the model will classify what does the sound the audio belongs to either `cat` or `dog`.

```json
{
  "predictions": {
    "class": "dog",
    "label": 1,
    "probability": 1.0
  },
  "success": true
}
```

### Starting the server

To start server and start `audio` classification first you need to make sure you are in the `server` folder and run the following commands:

1. creating a virtual environment

```shell
virtualenv venv && .\venv\Scripts\activate.bat
```

2. installing packages

```shell
pip install -r requirements.txt
```

3. Starting the server

```shell
python api/app.py
```

> The server will start on a default port of `3001` and you will be able to make api request to the server to do audio classification.

### Model Metrics

The following table shows all the metrics summary we get after training the model for few `15` epochs.

<table border="1">
    <thead>
      <tr>
        <th>model name</th>
        <th>model description</th>
        <th>test accuracy</th>
        <th>validation accuracy</th>
        <th>train accuracy</th>
         <th>test loss</th>
        <th>validation loss</th>
        <th>train loss</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>cats-dogs-sound-cnn.pt</td>
        <td>audio sentiment classification for dogs and cats CNN.</td>
        <td>90.7%</td>
        <td>90.7%</td>
        <td>93.5%</td>
        <td>0.621</td>
        <td>0.218</td>
        <td>0.209</td>
      </tr>
       </tbody>
  </table>

### Classification report

The following is the classification report for the model on the `test` dataset.

<table border="1">
    <thead>
      <tr>
        <th>#</th>
        <th>precision</th>
        <th>recall</th>
        <th>f1-score</th>
        <th>support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>accuracy</td>
        <td>-</td>
        <td>-</td>
        <td>90%</td>
        <td>2305</td>
      </tr>
      <tr>
        <td>macro avg</td>
        <td>91%</td>
        <td>90%</td>
        <td>90%</td>
        <td>2305</td>
      </tr>
      <tr>
        <td>weighted avg</td>
        <td>92%</td>
        <td>89%</td>
        <td>90%</td>
        <td>2305</td>
      </tr>
    </tbody>
  </table>

### Confusion matrix

The following figure shows a confusion matrix for the classification model.

<p align="center" with="100%"><img src="images/cm.png" width="100%" alt=""/>
</p>

### Audio Sentiment classification

If you hit the server at `http://localhost:3001/classify` you will be able to get the following expected response that is if the request method is `POST` and you provide the file expected by the server.

### Expected Response

The expected response at `http://localhost:3001/classify` with a file `audio` of the right format will yield the following `json` response to the client.

```json
{
  "predictions": {
    "class": "dog",
    "label": 1,
    "probability": 1.0
  },
  "success": true
}
```

### Using `curl`

Make sure that you have the audio named `cat.wav` in the current folder that you are running your `cmd` otherwise you have to provide an absolute or relative path to the audio.

> To make a `curl` `POST` request at `http://localhost:3001/classify` with the file `cat.wav` we run the following command.

```shell
# for cat
curl -X POST -F audio=@cat.wav http://127.0.0.1:3001/classify

# for dog
curl -X POST -F audio=@dog.wav http://127.0.0.1:3001/classify
```

### Using Postman client

To make this request with postman we do it as follows:

1. Change the request method to `POST` at http://127.0.0.1:3001/classify
2. Click on `form-data`
3. Select type to be `file` on the `KEY` attribute
4. For the `KEY` type `audio` and select the audio you want to predict under `value`
5. Click send

If everything went well you will get the following response depending on the face you have selected:

```json
{
  "predictions": { "class": "dog", "label": 1, "probability": 1.0 },
  "success": true
}
```

### Using JavaScript `fetch` api.

1. First you need to get the input from `html`
2. Create a `formData` object
3. make a POST requests

```js
const input = document.getElementById("input").files[0];
let formData = new FormData();
formData.append("audio", input);
fetch("http://127.0.0.1:3001/classify", {
  method: "POST",
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

If everything went well you will be able to get expected response.

```json
{
  "predictions": { "class": "dog", "label": 1, "probability": 1.0 },
  "success": true
}
```

### Notebooks

- All notebooks for training and saving the models are found in the `notebooks` folder of this repository.
