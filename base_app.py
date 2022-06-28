"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from sklearn.feature_extraction.text import CountVectorizer
# Data dependencies
import pandas as pd
import numpy as np
#import pprint
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
# Set plot style for data visualisation
sns.set()
from PIL import Image
#import spacy
#nlp= spacy.load('en')
#NLP Packages
#from wordcloud import WordCloud
#from textblob import TextBlob

# Vectorizer
news_vectorizer = open("resources/tfidf_ndu.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
#Models
#def load_predictions_models(model_file):
    #load_prediction_models = joblib.load(open(os.path.join(model_file),"rb"))
    #return load_prediction_models

#creating dict keys for predictions
def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")
    # Using PIL

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Exploratory Data Analysis", "Prediction", "Profile"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        image = Image.open('img5.png')
        st.image(image, caption=None, use_column_width=True)
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")
        #st.subheader("Who should use this tool")
        st.subheader("==========================================================")
        st.subheader("Benefits of using this classification tool")
        st.subheader("==========================================================")
        st.markdown("1. Quicker near real-time results compared to a survey without having to pay for expensive reports")
        st.markdown("2. Easily accessible on your internet browser")
        st.markdown("3. You don't need to understand complicated statistical techniques, just need to understand insights")
        st.markdown("If you are a startup or even an established business looking to launch a new product, are you aware of your potential customers sentiments regarding climate change? As a not for profit organisation looking for donors for environmental projects, do you know what your donors thoughts are regarding climate change? Knowing this information can help you better prepare to take your organisations strategy forward. Not knowing this information can make you seem irrelevant to your target market and cause you to miss out on an opporunity of a lifetime. The tweet classifier will help you be more prepared and relevant to your audience.")
        st.subheader("==========================================================")
        st.subheader("Instructions for using this tool")
        st.subheader("==========================================================")
        st.markdown("Let us help you turn insights from your potential customers to action.")
        st.markdown("Get started by:")
        st.markdown("1. Navigating to the sidebar at the top left of this page")
        st.markdown("2. Choose an option by clicking the 'Choose Option' dropdown")
        st.markdown("3. Select the option you wish to view")
        st.markdown("4. Get insights that will help you be better prepared")
        if st.checkbox('Choose to Preview Example'):
            st.subheader("Example chart: What are the most frequent opinions regarding climate change?")

            import plotly.figure_factory as ff
            # Add histogram data
            x1 = np.random.randn(200) + 1
            x2 = np.random.randn(200)
            x3 = np.random.randn(200)
            x4 = np.random.randn(200) - 1


            # Group data together
            hist_data = [x1, x2, x3, x4]

            group_labels = ['Anti: -1', 'Neutral: 0', 'Pro: 1', 'News: 2']

             # Create distplot with custom bin_size
            fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5, .75])

            # Plot!
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("Your conclusion from the above example chart may be that those who are against ('Anti:-1') climate change are likely to have tweet more about  their opinions. The resulting insight from this tool may help you with your next step for formulating your brand messaging and positioning etc.")
            #chart_data1 = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
            #st.line_chart(chart_data1)

        chart_data2 = {'Class':['2', '1', '0', '-1'], 'Description':['News: the tweet links to factual news about climate change', 'Pro: the tweet supports the belief of man-made climate change', 'Neutral: the tweet neither supports nor refutes the belief of man-made climate change', 'Anti: the tweet does not believe in man-made climate change']}
        df = pd.DataFrame(chart_data2)
        blankIndex=[''] * len(df)
        df.index=blankIndex
        st.subheader("==========================================================")
        st.subheader("Table: Data dictionary")
        st.subheader("==========================================================")
        st.table(df)
        st.subheader("==========================================================")
        st.subheader("Raw Twitter data and label")
        st.subheader("==========================================================")
        st.markdown("The collection of this data which we use as our data source was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes in the table below:")


        st.markdown("Select tickbox to view raw data where the 'sentiment'column denotes the 'class' column in the above data dictionary table which is associated to a description ranging from 'Anti' to'News' as descriptions for tweet messages that classify a range people that do not believe to those that believe in climate change respectively.")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page


    # Building out the Exploratory Data Analysis page
    if selection == "Exploratory Data Analysis":
        st.markdown("Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and diagramatic representations")
        st.info("Tweet Data Insights")

        st.subheader("General Insights")
        st.markdown("A subset of the dataset denoted by the 'Head' and the 'Tail' shows what raw tweet data looks like when placed in rows and columns. Building on our definition of the four sentiment classes defined in the information page, a handful of graphical plots are used to show the distribution of tweets by deniers and belivers of climate change. This includes:")
        st.markdown("1. Which proportion of tweets belong to the believer versus the denier group (or neutral) group. This can show which views are most widely help if we use tweets as our proxy for climate change belief sentiment i.e. Use Bar chart plot, Pie chart and Word Count plots to get a sense of the more widely held beliefs of climate change.")
        st.markdown("2. Which common words exist and were predominantly in describing climate change i.e. Use the Word Cloud and ngram (Unigram, Bigram etc) plots to see what words were most prevelant amongst the climate change sentiment tweets.")
    #my_dataset = 'resources/train.csv'

    #Loading Dataset
    #@st.cache(persist=True)
    #def explore_data(dataset):
        #train_df = pd.read_csv(os.path.join(dataset))
        #return train_df

    #data = explore_data(my_dataset)

        DATA_URL = ('resources/train.csv')

        @st.cache(allow_output_mutation=True)
        def load_data(nrows):
            data = pd.read_csv(DATA_URL, nrows=nrows)
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis='columns', inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(40000)
        data_load_state.text("Done! (using st.cache)")

        #Minimal cleaning
        import re

        #showing what our dataset is
        if st.checkbox('Choose to Preview Dataset and Visualisations'):
            #data = explore_data(my_dataset)
            @st.cache(allow_output_mutation=True)
            def remove_URL(message):
                url = re.compile(r'https?://\S+|www\.\S+')
                return url.sub(r'',message)
            if st.checkbox('Choose to remove URLs from data (may take long to run)'):
                st.markdown("***Unselect the box and stop running if it takes longer than 5 minutes***.")
                st.text("This selection allows you to clean URL's for the tweets but may take longer to run.")
                st.text("Leave this selection unselected for better run times.")
                data['message']=data['message'].apply(lambda x : remove_URL(x))
            if st.button('Head'):
                st.write(data.head())
            elif st.button('Tail'):
                st.write(data.tail())
            else:
                st.write(data.head(2))
            #Show entire Dataset
            if st.checkbox('Show all Dataset'):
                st.write(data)

            #Show Column Names
            if st.checkbox('Show Column Names'):
                st.write(data.columns)

            #Show Dimensions
            data_dim = st.selectbox('What Dimensions Do You Want to See?',['Rows','Columns'])
            if data_dim == 'Rows':
                st.text('Showing Rows')
                st.write(data.shape[0])
            elif data_dim == 'Columns':
                st.text('Showing Columns')
                st.write(data.shape[1])

            #Select a Column
            col = st.selectbox('Select Column', ('sentiment','message','tweetid'))
            if col == 'sentiment':
                st.write(data['sentiment'])
            elif col == 'message':
                st.write(data['message'])
            elif col == 'tweetid':
                st.write(data['tweetid'])
            else:
                st.write('Select Column')

            #Add Plots

            #Bar Chart
            if st.checkbox('Show Bar Chart Plot for distribution of tweets across sentiment classes'):
                st.write(data['sentiment'].value_counts())
                st.bar_chart(data['sentiment'].value_counts())

            #Add Pie Chart
            if st.checkbox('Show Pie Chart for proportion of each sentiment class'):
                pie_labels = ['Pro', 'News', 'Neutral', 'Anti']
                fig1 = go.Figure (data=[go.Pie(labels=pie_labels, values = data['sentiment'].value_counts())])
                st.plotly_chart(fig1)

            #Word count for distribution
                    # Define subplot to see graphs side by side
            if st.checkbox('Show Word Count Plot for distribution of tweets across sentiment classes'):
                data['word_count'] = data['message'].apply(lambda x: len(x.split()))

                # Split so we can use updated train set with new feature
                data = data[:len(data)]

                # Define subplot to see graphs side by side
                fig, ax = plt.subplots(figsize = (10, 5))

                #create graphs
                sns.kdeplot(data['word_count'][data['sentiment'] == 0], shade = True, label = 'Neutral')
                sns.kdeplot(data['word_count'][data['sentiment'] == 1], shade = True, label = 'Pro')
                sns.kdeplot(data['word_count'][data['sentiment'] == 2], shade = True, label = 'News')
                sns.kdeplot(data['word_count'][data['sentiment'] == -1], shade = True, label = 'Anti')

                # Set title and plot
                plt.title('Distribution of Tweet Word Count')
                plt.xlabel('Word Count')
                plt.ylabel('Sentiments Proportions')
                st.pyplot()
            #Show word cloud chart
            if st.checkbox('Show Word Cloud to see the most words in the climate change tweets'):
                from wordcloud import WordCloud
                Allwords= ' '.join( [tweets for tweets in data['message']] )
                wordCloud= WordCloud(width= 700, height= 500, random_state= 21, max_font_size= 150).generate(Allwords)
                plt.imshow(wordCloud, interpolation= 'bilinear')
                plt.axis('off')
                st.pyplot()

            #Show Ngram bar chart
            @st.cache(allow_output_mutation=True)
            def get_top_tweet_unigrams(corpus, n=None):
                vec = CountVectorizer(ngram_range=(1, 1)).fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0)
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]
            if st.checkbox('Show Unigram Plot for top 20 common single words in tweets'):
                plt.figure(figsize=(10,5))
                top_tweet_unigrams=get_top_tweet_unigrams(data['message'])[:20]
                x,y=map(list,zip(*top_tweet_unigrams))
                sns.barplot(x=y,y=x)
                st.pyplot()



            #anyone can add from here

    # Building out the predication 
    #loading models
    def load_prediction_models(models):
        loaded_models =joblib.load(open(os.path.join(models),"rb"))
        return loaded_models

    if selection == "Prediction":
        st.info("Prediction with ML Models")
        st.markdown("Predictive modeling is a process that uses data and statistics to predict outcomes with classification data models. These models can also be used to predict our twitter data. We get predictions from models such as Logistic Regression, LinearSVC, Naive Bayes Classifier and many more.")
        st.markdown("LogisticRegression- Is used to obtain odds ratio in the presence of more than one exploratory variable. It explain the relationship between one dependent binary variable and one or more independent variables")
        st.markdown("Support Vector Machine- It analyzes data used for classification and rehression analysis. It separates data points using hyperplane with the largest amount of margin")
        st.markdown("Naive Bayes Classifier- It uses the principle of Bayes Theorem to make classification. This model is quick and simple to build but does it is not the most accurate. Its advantage is that it is useful on large dataset")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        #load some models used
        all_models= ["LinearSVC", "LogisticRegression", "Naive Bayes Classifier"]
        model_choice= st.selectbox("Choose ML Model", all_models)

        def load_prediction_models(models):
                loaded_models =joblib.load(open(os.path.join(models),"rb"))
                return loaded_models

        if st.button("Classify"):
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            if model_choice == 'LinearSVC':
                predictor = joblib.load(open(os.path.join("resources/clf_ndu.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
            
            elif model_choice == 'LogisticRegression':
                predictor = joblib.load(open(os.path.join("resources/lr_model_ndu.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
            
            elif model_choice == 'Naive Bayes Classifier':
                predictor = joblib.load(open(os.path.join("resources/nb_model_ndu.pkl"),"rb"))
                prediction = predictor.predict(vect_text)

            def getAnalysis(prediction):
                if prediction == -1:
                    return "Anti  (i.e. a denier of man-made climate change)"
                elif prediction ==0:
                    return "Neutral  (i.e. Neutral in beliefs about man-made climate change)"
                elif prediction == 1:
                    return "Pro  (i.e. Believes that climate change is man-made)"
                else:
                    return "News  (i.e. Has factual sources that climate change is man-made)"

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(getAnalysis(prediction)))

    #Building Profile Page
    if selection == "Profile":
        st.info("Founders Information")
        st.markdown("Kamogelo Masekwa")
        st.markdown("Alex Thotse")
        st.markdown("Mandisa Buthelezi")
        st.markdown("Nokubongwa Ndlela")
        st.markdown("Vumbhoni Maluleke")
	st.markdown("Lefa Matlala")
	
       

    


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
