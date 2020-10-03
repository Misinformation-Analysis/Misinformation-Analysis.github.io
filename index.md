![Image](infographic.png)

### Introduction

The aim of this project is to utilize different features of articles and social media posts regarding current events, politics, history, and economics to predict if these posts are instances of reliable or fake news. Fake news refers to the deliberate distribution of misinformation—news whose main purpose is to distort the truth for the intention of persuasion seeking to drive action. Coupled with the fact that misinformation has the tendency to spread faster online than real news by a substantial margin [(*Vosoughi*)](http://science.sciencemag.org/content/359/6380/1146), it is crucial that we provide readers with the tools to recognize the validity of their news so that we can collectively use accurate information to make the best decisions possible on policies that impact us all.
 
The spread of misinformation should be viewed as a threat to the digital landscape. Our hope is to better understand what "types" of articles are subjected to fake news. Types here is defined broadly as content, length, source, etc. Analysis like this would be of great use as it could help identify where fact checking resources are needed the most.

### Methods
Our method for this project relies on two major steps. Step one will be performing unsupervised learning on a large [dataset](https://www.kaggle.com/snapcrack/all-the-news) of articles. Our proposed process for this step will be to vectorize the articles and cluster them into groups using clustering algorithms like DBSCAN or KMeans [(Mustakim)](https://iopscience.iop.org/article/10.1088/1742-6596/1363/1/012001) and then to choose the best clustering with clustering evaluation measures.
 
The second step of the project will be to create a supervised classifier based on this labeled fake news [dataset](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset). We will use this classifier to attempt to label the articles in the clusters from step one. Our analysis for this project will revolve around whether certain clusterings of articles (of which we will examine for similarities) have a higher rate of fake news.

### Results
Based on the clustering results given by the unsupervised clustering, we hope to be able to identify clusters and characteristics of those clusters that are highly associated with fake news. For example, if we discover that articles that have calls to action or are written with inflammatory words tend to be more inaccurate, while articles that describe multiple points of view or are written using academic language are typically legitimate, we could use these clusters as good indicators of fake news and accurate information. Using our trained model, we then hope to be able to predict the veracity of articles in the large unlabeled dataset.

### Discussion

The best outcome of this project would be a successful prediction of the accuracy of a news article. Misinformation could be unintentional, false stories, or news developed for the purpose of influencing the reader in a certain direction [(_Thota_)](https://scholar.smu.edu/datasciencereview/vol1/iss3/10/). We hope to then analyze these predictions to better identify key indicators and sources. This information will be useful for fact checkers and people developing robust fake news detection systems. This will, in turn, lead to a slowed spread of fake news. A better-informed reader will lead to them making decisions based on facts rather than misinformation that they might have been consuming otherwise. If mainstream media outlets and platforms were able to better screen for accurate information, the public discourse around controversial issues would be heavily improved.

### References

1. Vosoughi, Soroush. “The Spread of True and False News Online.” Science, vol. 359, no. 6380, 2018, pp. 1146-1151. Science, [science.sciencemag.org/content/359/6380/1146](http://science.sciencemag.org/content/359/6380/1146).

2. Thota, Aswini, et al. “Fake News Detection: A Deep Learning Approach.” SMU Data Science Review, vol. 1, no. 3, 2018. [https://scholar.smu.edu/datasciencereview/vol1/iss3/10/](https://scholar.smu.edu/datasciencereview/vol1/iss3/10/).

3. Mustakim et al 2019 J. Phys.: Conf. Ser. 1363 012001. "DBSCAN algorithm: twitter text clustering of trend topic pilkadapekanbaru." [https://iopscience.iop.org/article/10.1088/1742-6596/1363/1/012001](https://iopscience.iop.org/article/10.1088/1742-6596/1363/1/012001)

_This project proposal is produced for Georgia Tech CS 4641 - Fall 2020_
