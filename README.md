# Fake News Classification

Do you know how much is fake news influencing our lives? How can we predict if the news is fake or real? How can we apply Natural Language Processing for that? 

This dataset has 45k news in different subjects being real and fake news. Data were collected from the reuters.com website (true news) and the fake news where collected from several websites not reliable by Politifact and Wikipedia. 

The dataset consists of two files: "True.csv" with 21417 news considered as true; "Fake.csv" with 23481 fake news. 
Each row contains information about a new: title, text, subject, and publication date. Each news was collected and classified into different subjects (news, politics, left-news, government news, US news, middle-east). 

Considering only the text of the news, I created several pipelines to compare distinct prediction models. I have cleaned all the text, made some analyzes (as you can read in the link below), and then I have removed the stop words and made stemming. I have compared three types of converters of a text collection to a matrix of token counts: CountVectorizer, TfidfVectorizer, and HashVectorizer. When comparing the multinomial Naive Bayes classifier (MultinomialNB) with the Linear Support Vector Classification (LinearSVC), my results shows that TfidfVectorizer combined with the LinearSVC achieves better accuracy compared to other models.

- CountVectorizer_multinomialNB: 97.65%
- CountVectorizer_linearSVC: 98.24%
- TfidfVectorizer_multinomialNB: 96.92%
- TfidfVectorizer_linearSVC: 99.07%
- HashVectorizer_multinomialNB: 96.37%
- HashVectorizer_linearSVC: 95.03%

In future works, I will analyze overfitting since in my analysis the dataset may be biased due to the creation of fake news.
