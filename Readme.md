# Stock Price Prediction Using Multi-Modal Features (Audio + Text Transcripts)

###### Data
###### Approach
###### Model
The model follows the general architecture of the model mentioned in the paper in that it also first creates a within unimodal encoding and then creates a between modals encoding which is then passed to a regression model that predicts the next 30 days of prices.
But instead of using 27 core features of the speech, i used encodings generated from a pretrained speech emotion classifier model to generate the speech features, and instead of using glove encodings, i used bert encodings.
Also, the unimodal encodings are generated from a similar architecture which is a transformer and are concatenated and passed to another transformer to learn multi-modal correlations.
The output of this transformer is passed to a linear model which generates linear predictions for the next 30 days of stock prices.
I compare 7-day, 15-day and 30-day activations for generating the scores.
###### Further improvements
1. Instead of directly concatenating the audio and text encodings, use the approach similar to the paper [multimodal learning]().
2. Use a better speech encoder like [pase]()
###### References
###### Usage
1. Download the data from the [link](https://drive.google.com/file/d/15wtWZvSJicF_Ur2V45lCyCjNJQ7QfXth/view) mentioned in the paper.
2. Extract the data
3. Run the script prepare_data.py : it'll create train, validation and test folders, convert the mp3 audios to wav files, and download the yahoo finance data.
4. Run the training using train.py
5. To predict, pass the directory where the data is stored and run test.py

###### Results
<table><thead>
<tr>
<td colspan=5><code class="prettyprint"><b>MSE Scores</b></code></td>
</tr>
<tr>
<th>Model</th>
<th>3-days</th>
<th>7-days</th>
<th>15-days</th>
<th>30-days</th>
</tr>
</thead><tbody>
<tr>
<td>Paper</td>
<td>0.78</td>
<td>0.78</td>
<td>0.78</td>
<td>0.78</td>
</tr>
<tr>
<td>Mine</td>
<td>0.78</td>
<td>0.78</td>
<td>0.78</td>
<td>0.78</td>
</tr>
</table>