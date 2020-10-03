## Analyzing Fake News Based on Article Clustering

![Image](infographic.png)

### Introduction

The aim of this project is to utilize different features of articles and social media posts regarding current events, politics, history, and economics to predict if these posts are instances of reliable or fake news. Fake news refers to the deliberate distribution of misinformation—news whose main purpose is to distort the truth for the intention of persuasion seeking to drive action. Coupled with the fact that misinformation has the tendency to spread faster online than real news by a substantial margin (*Vosoughi*), it is crucial that we provide readers with the tools to recognize the validity of their news so that we can collectively use accurate information to make the best decisions possible on policies that impact us all.
 
The spread of misinformation should be viewed as a threat to the digital landscape. We will build a tool that automatically scans through articles and posts, using feature patterns commonly associated with real and fake news to recognize these works as either truthful or deceptive. A tool like this would be of great use as it could warn the people viewing this content if it falls into a potentially deceptive category, signaling that they should not trust the source or at least be highly skeptical of its validity.

### Methods
Our method for this project relies on two major steps. Step one will be performing unsupervised learning on large datasets of articles. Our proposed process for this step will be to tokenize the articles and cluster them into groups using DBSCAN. We will do this process both with a dataset of generic articles and with a dataset that has also been labeled as fake news or not.
 
The second step will be to create a supervised classifier based on the label dataset. We will use this classifier to label the items in the clusters from step one. Our analysis for this project will revolve around whether certain groupings of articles (of which we will examine for similarities) have a higher rate of fake news.

### Results

### Discussion

### References

_This project proposal is produced for Georgia Tech CS 4641 - Fall 2020_
