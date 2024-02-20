import requests
from selenium import webdriver
import time
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
import json
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def interval_calibrator(min:int, max:int, porcentage=0.75):
    """
    Calibrates a given interval into two subintervals with a specified proportion.

    Args:
        min: The minimum value of the original interval.
        max: The maximum value of the original interval.
        porcentage: The proportion of the first subinterval to the entire interval.
            Defaults to 0.75.

    Returns:
        A list containing two subintervals:
        - The first subinterval, covering the specified proportion of the original interval.
        - The second subinterval, covering the remaining proportion.

    Examples:
        >>> interval_calibrator(10, 20)
        [[10, 15], [15, 20]]
        >>> interval_calibrator(5, 30, porcentage=0.6)
        [[5, 17], [17, 30]]
    """
    interval = max-min 
    interval_1 = [min, round(min + interval*(1-porcentage))]
    interval_2 = [interval_1[1], max]
    return [interval_1,interval_2]

def steam_ids_scrapper():
    """
    Scrapes Steam game IDs from the specified URL using Selenium WebDriver.

    Args:
        url (str, optional): The URL of the Steam store page to scrape.
            Defaults to 'https://store.steampowered.com/games/?l=latam&flavor=contenthub_all&offset=24'.

    Returns:
        List[int]: A list of extracted Steam game IDs.

    Raises:
        WebDriverException: If an error occurs during WebDriver interaction.
        JSONDecodeError: If JSON data extraction fails.

    Notes:
        - This function relies on external libraries: `selenium` and `json`.
        - The URL and selectors might change over time, requiring code adjustments.
        - Consider ethical implications and potential website terms of service before scraping.
        - Responsible scraping practices involve rate limiting, user-agent rotation, and avoiding excessive requests.

    Examples:
        >>> steam_ids_scrapper()  # Extract IDs from default URL
        [123456, 789012, ...]

        >>> steam_ids_scrapper('https://store.steampowered.com/genre/Action')  # Scrape action games
        [987654, 321098, ...]
    """
    
    URL = 'https://store.steampowered.com/games/?l=latam&flavor=contenthub_all&offset=24'
    path_chrome = './chromedriver-win64/chromedriver.exe'
    options = ChromeOptions()
    options.add_argument("--headless=new")
    service = webdriver.ChromeService(executable_path=path_chrome)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(URL)
    time.sleep(3)
    element = driver.find_element(By.CLASS_NAME, 'responsive_page_template_content')

    app = element.find_element(By.ID,'application_config') #contains in json information alongside with IDs

    #IDs
    id1 = json.loads(app.get_attribute('data-section_93094_3_*') ) #Lista de juegos ordenados por mas relevantes
    id2 = json.loads(app.get_attribute('data-section_80021_3_*'))
    id3 = json.loads(app.get_attribute('data-hubitems_3'))
    ids = id1['appids'] + id2['appids'] + id3['appids']
    return ids

def steam_id_name_finder(id):
    """
    Retrieves the name of a Steam game given its ID using the Steam Web API.

    Args:
        id (int): The Steam ID of the game.

    Returns:
        str: The name of the game, or an empty string if the game is not found.

    Raises:
        requests.exceptions.HTTPError: If an error occurs during the API request.
        ValueError: If the provided ID is not a valid integer.
    """
    try:
        URL = 'https://api.steampowered.com/ISteamApps/GetAppList/v0002/'
    except: 
        raise requests.exceptions.HTTPError
    else:
        page = json.loads(requests.get(URL).text) 
        game_list = page['applist']['apps']

   

        element = filter(lambda e:e['appid']==id, game_list)

        for e in element:
            return e['name']

        
def id_reviews_scrapper(id:int, n_rev_per_game:int,**kwargs):
    """
    Scrapes Steam game reviews for the given ID, up to a specified number.

    Args:
        id (int): The Steam ID of the game.
        n_rev_per_game (int): The maximum number of reviews to scrape.
        **kwargs (optional): Additional parameters to customize the request.
            Supported options include:
                - language (str): Filter reviews by language (default: 'english').
                - filter (str): Apply a review filter ('recent', 'updated', or 'all').
                - day_range (int): For 'all' filter, specify the number of days back to include (default: 360).
                - num_per_page (int): Number of reviews returned per page (default: 100, max: 100).
                - filter_offtopic_activity (int): Whether to filter off-topic activity (default: 1).
                - review_type (str): Filter reviews by type ('all', 'positive', or 'negative').

    Returns:
        dict: A dictionary containing:
            - review_list (list): A list of retrieved reviews, up to the specified number.
            - game_id (int): The Steam ID of the game.

    Raises:
        requests.exceptions.HTTPError: If an error occurs during the API request.
        ValueError: If invalid parameters are provided.

    Notes:
        - This function relies on the external library `requests`.
        - Consider ethical implications and potential website terms of service before scraping.
        - Be mindful of Steam API rate limits and responsible scraping practices.

    Examples:
        >>> reviews = id_reviews_scrapper(430, 10)  # 10 latest English reviews
        >>> print(reviews['review_list'][0]['review'])  # Print the first review

        >>> reviews = id_reviews_scrapper(430, 20, filter='positive')  # 20 positive reviews

        >>> reviews = id_reviews_scrapper(430, 50, **{'language': 'french'})  # 50 French reviews
    """
    
    
    
    params = {
        'json':1, #Return response in json
        'language':'english', #Language of reviews
        'filter':'all', #filter = {recent,updated,all}
        'day_range':'360', # for filter = all only , n days from now up to 365
        'num_per_page':'100', #Nomber reviews given per call (default 20) max 100
        'filter_offtopic_activity':1, #filter off topic activity
        'review_type':'all' #review_type = {all,positive,negative}
        }
    #Modify params
    for key,value in kwargs.items():
        params[key] = value
    
    
    game_URL = f'https://store.steampowered.com/appreviews/{id}?'
    review_list = []
    
    n_reviews = 0

    while n_reviews < n_rev_per_game:
        try:
            page = json.loads(requests.get(game_URL, params=params).text)
            
        except:
            raise requests.exceptions.HTTPError
        else:
            review_list = review_list + page['reviews']
            n_check = n_reviews + int(params['num_per_page'])
            n_reviews = n_reviews + len(page['reviews'])
            if(n_check != n_reviews and n_reviews > n_check): #Las reviews se repiten ciclicamente al alcanzar la ultima
                break
            params['cursor'] = page['cursor'] #agregamos un nuevo cursor a los parametros para mostrar luego 100 reviews mas
        
    

    
    
    reviews = {'review_list':review_list[:n_rev_per_game],'game_id':id}
    
        
    
    return reviews
    

def review_builder(reviews:dict):
    """
    Restructures scraped Steam game reviews into a more organized format.

    Args:
        reviews (dict): A dictionary containing the raw scraped reviews,
            with keys 'review_list' (list of reviews) and 'game_id' (int).

    Returns:
        dict: A restructured dictionary with the following keys:
            - game_id (list): A list of game IDs for each review.
            - game_name (list): A list of game names corresponding to each ID.
            - review (list): A list of the text content of each review.
            - voted_up (list): A list of voted_up values for each review.
            - votes_up (list): A list of votes_up values for each review.
            - votes_funny (list): A list of votes_funny values for each review.
            - weighted_vote_score (list): A list of weighted_vote_score values (float).

    Raises:
        TypeError: If the input `reviews` dictionary is not in the expected format.
        ValueError: If the retrieved game name is empty.

    Notes:
        - This function relies on the external function `steam_id_name_finder`.
        - Consider caching game names to avoid redundant API calls.
        - Explore alternative data structures (e.g., DataFrames) for more efficient handling.
    """
    
    reviews_list = reviews['review_list'] #LISTA CON LAS REVIEWS
    id = reviews['game_id']
    name = steam_id_name_finder(id)
    
    #Parametros:
    rev_list = []
    voted_up_list = []
    votes_up_list = []
    votes_funny_list = []
    weighted_vote_score_list = []
    id_list = []
    name_list = []
    
    for rev in reviews_list:
        rev_list.append(rev['review'])
        voted_up_list.append(rev['voted_up'])
        votes_up_list.append(rev['votes_up'])
        votes_funny_list.append(rev['votes_funny'])
        weighted_vote_score_list.append(float(rev['weighted_vote_score']))
        id_list.append(id)
        name_list.append(name)
     
        
    reviews_build = {
        'game_id':id_list,
        'game_name':name_list,
        'review':rev_list, 
        'voted_up':voted_up_list, 
        'votes_up':votes_up_list,
        'votes_funny':votes_funny_list, 
        'weighted_vote_score':weighted_vote_score_list
        }
    
    
    return reviews_build

def wikipedia_slang(exclude_words:list):
    """
    Scrapes gaming slang terms from a specific Wikipedia glossary page,
    excluding provided words.

    Args:
        exclude_words (list): A list of words to exclude from the results.

    Returns:
        list: A list of scraped slang terms, excluding duplicates and
            words in the exclude_words list.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the
            HTTP request.
        BeautifulSoup.FeatureNotFound: If the HTML parser is not found.

    Notes:
        - This function relies on the external libraries `requests` and
          `BeautifulSoup4`.
        - Be mindful of Wikipedia's terms of service and rate limits.
        - Handle potential errors gracefully, such as network issues or
          changes in page structure.
        - Consider using a more robust HTML parser or library for complex
          scraping tasks.
    """
    exclude_words = [x.lower() for x in exclude_words]
    W_URL = 'https://en.wikipedia.org/wiki/Glossary_of_video_game_terms'
    wikipedia_page = requests.get(W_URL)
    soup = BeautifulSoup(wikipedia_page.content, 'html.parser')
    results = soup.find_all('dfn', class_='glossary')
    slang = []
    for e in results:
        slang.append(e.text)
    slang = [x.lower() for x in slang]
    slang = list(set(slang).difference(set(exclude_words)))
    return slang

#AGREGAR CONDICION SI ID_REVIEWS SCRAPER ES VACIO
def game_slang_filter(reviews_df, slang:list=[], **kwargs):
    """
    Filters slang terms from game reviews using TF-IDF and potential Wikipedia scraping.

    Args:
        reviews_df (pd.DataFrame): A DataFrame containing a "review" column with textual content.
        slang (list, optional): A list of slang terms to explicitly include in the results.
        **kwargs (optional): Keyword arguments to adjust TF-IDF vectorization parameters.
            Supported options include:
                - min_df (int): Minimum document frequency for a term to be considered (default: 3).
                - stop_words (str or list): Stop words to remove (default: 'english').
                - lowercase (bool): Whether to convert text to lowercase (default: True).
                - strip_accents (str): How to handle accents (default: 'ascii').
                - ngram_range (tuple): Range of n-grams to consider (default: (1, 3)).

    Returns:
        list: A list of extracted slang terms present in the reviews or provided list.

    Raises:
        ValueError: If the reviews DataFrame does not have a "review" column.
        ImportError: If required libraries (`pandas`, `sklearn`) are not installed.

    Notes:
        - This function relies on the external libraries `pandas` and `sklearn`.
        - If `slang` is empty, it scrapes terms from Wikipedia (use responsibly).
        - Consider tuning TF-IDF parameters for optimal results.
        - Explore alternative slang extraction methods depending on your use case.

    Examples:
        >>> reviews_df = pd.DataFrame({'review': ['This game is lit AF!']})
        >>> slang = game_slang_filter(reviews_df, slang=['lit'])
        >>> print(slang)  # ['lit']

        >>> reviews_df = pd.read_csv('game_reviews.csv')
        >>> all_slang = game_slang_filter(reviews_df)
        >>> print(len(all_slang))  # Number of all extracted slang terms
    """
    

    params = {
        'min_df':3, 
        'stop_words':'english',
        'lowercase':True,
        'strip_accents':'ascii',
        'ngram_range':(1,3)
        }
    
    for key,value in kwargs.items():
        params[key] = value

    tfidf = TfidfVectorizer(**params)
    tfidf.fit(reviews_df['review'])
    names = tfidf.get_feature_names_out() #feature names

    if slang:
        slang = [x.lower() for x in slang]
        game_slang = set(names).intersection(set(slang)) # slang words present in the game reviews
    else:
        slang = wikipedia_slang(exclude_words=['life'])
        game_slang = set(names).intersection(set(slang)) # slang words present in the game reviews

    

    return list(game_slang)

def game_slang_analysis(words_df, game_slang):
    """
    Analyzes the distribution and sentiment of slang terms in game reviews.

    Args:
        words_df (pd.DataFrame): A DataFrame containing words and their properties.
            Must have columns named 'words' (slang terms) and 'index' (position in reviews).
        game_slang (list): A list of slang terms to analyze.

    Returns:
        dict: A dictionary mapping each slang term to its sentiment classification ('very bad', 'bad', 'more less bad', 'more less good', 'good', 'very good').

    Raises:
        ValueError: If required columns are missing in the words_df DataFrame.

    Notes:
        - This function relies on the presence of specific columns in the words_df DataFrame.
        - The sentiment classification is based on predefined intervals derived from word position indices.
        - Consider customizing the sentiment labels and classification logic based on your needs.
        - Explore alternative sentiment analysis techniques for more sophisticated predictions.

    Examples:
        >>> slang_df = pd.DataFrame({'words': ['lit', 'af', 'amazing'], 'index': [10, 50, 100]})
        >>> slang_analysis = game_slang_analysis(slang_df, ['lit', 'amazing'])
        >>> print(slang_analysis)  # {'lit': 'very good', 'amazing': 'good'}
    """
    
    indexes = []
    set1 = set(words_df['words'])
    set2 = set(game_slang)
    slang = list(set1.intersection(set2))
    for e in slang:
        try:
            ele = words_df[words_df['words'] == e]
            indexes.append(ele.index.values[0])
        except IndexError:
            print(f'ERROR IN WORD {e}')
    
    
    slang_list = list(np.array(game_slang)[np.array(indexes).argsort()])
    game_slang = dict(zip(slang_list,indexes))
    indexes = np.array(words_df['index'])

    
    indexes = np.array(words_df['index'])

    
    classification = {}
    
    ##Intervals:
    inter_a, inter_b = interval_calibrator(min=0, max=indexes.max(), porcentage=0.50)
    inter_a_1,inter_a_2 = interval_calibrator(min=inter_a[0],max=inter_a[1],porcentage=0.95)
    inter1,inter2 = interval_calibrator(min=inter_a_1[0],max=inter_a_1[1],porcentage=0.50)
    inter3,inter4 = interval_calibrator(min=inter_b[0],max=inter_b[1],porcentage=0.05)
    inter5,inter6 = interval_calibrator(min=inter4[0],max=inter4[1],porcentage=0.50)

    
    ####

    for key,value in game_slang.items():
        if value < inter1[1]:
            classification[key] = 'very bad'
        if value > inter2[0] and value < inter2[1]:
            classification[key] = 'bad'
        if value > inter_a_2[0] and value < inter_a_2[1]:
            classification[key] = 'more less bad'
        if value > inter3[0] and value < inter3[1]:
            classification[key] = 'more less good'
        if value > inter5[0] and value < inter5[1]:
            classification[key] = 'good'
        if value > inter6[0]:
            classification[key] = 'very good'


    
    return classification
    
def model_training(X,y,**kwargs):
    """
    Trains a Logistic Regression model for sentiment classification using TF-IDF features.

    Args:
        X (np.ndarray): A 2D array of textual data for training.
        y (np.ndarray): A 1D array of sentiment labels for training (e.g., 0/1 for negative/positive).

    Keyword Args:
        min_df (int, optional): Minimum document frequency for a term (default: 3).
        stop_words (str or list, optional): Stop words to remove (default: 'english').
        lowercase (bool, optional): Whether to convert text to lowercase (default: True).
        strip_accents (str, optional): How to handle accents (default: 'ascii').
        ngram_range (tuple, optional): Range of n-grams to consider (default: (1, 4)).
        C (float, optional): Inverse regularization strength for LogisticRegression (default: 10).

    Returns:
        dict: A dictionary containing:
            - model (LogisticRegression): The trained LogisticRegression model.
            - score (float): The best cross-validation accuracy score.
            - words (dict): A dictionary with three keys:
                - index (list): Sorted indices of features corresponding to model coefficients.
                - words (list): Feature names for the sorted indices.
                - coef (list): Sorted model coefficients corresponding to the features.

    Raises:
        ValueError: If input arrays are not 2D arrays or have different lengths.

    Notes:
        - This function relies on external libraries `numpy`, `sklearn.linear_model`,
          `sklearn.feature_extraction.text`, and `sklearn.model_selection`.
        - Consider optimizing hyperparameters with a wider grid search or different algorithms.
        - Explore feature engineering techniques for potentially better performance.

    Examples:
        >>> reviews = ['This game is awesome!', 'I hate this game so much.']
        >>> sentiment = [1, 0]  # 1: positive, 0: negative
        >>> model_results = model_training(reviews, sentiment)
        >>> print(model_results['score'])  # Best cross-validation accuracy
        >>> print(model_results['words']['words'][:5])  # Top 5 features (words)
    """

    
    X_train = X
    y_train = y
    
    params = {
        'min_df':3,
        'stop_words':'english',
        'lowercase':True,
        'strip_accents':'ascii',
        'ngram_range':(1,4)
    }
    for key,value in kwargs.items():
        params[key] = value
    tfidf = TfidfVectorizer(**params)

    pipe = make_pipeline(tfidf, LogisticRegression(max_iter=10000))
    param_grid = {'logisticregression__C':[10]}
    grid = GridSearchCV(pipe, cv=5, param_grid=param_grid)
    grid.fit(X_train,y_train)

    #to return
    best_model = grid.best_estimator_
    score = grid.best_score_
    coef = best_model.named_steps['logisticregression'].coef_
    feature_names = best_model.named_steps['tfidfvectorizer'].get_feature_names_out()
    words_dict = {'index':np.sort(coef[0].argsort()), 'words':feature_names[coef[0].argsort()], 'coef':coef[0][coef[0].argsort()]}


    return {'model':best_model, 'score':score, 'words':words_dict}


# def grpahs(slang_class:dict, n_words:int):
#     """
#     Visualizes the distribution of sentiment classifications for slang terms and
#     their coefficients in a 2-part plot.

#     Args:
#         slang_class (dict): A dictionary containing analysis results,
#             expected to have the following keys:
#                 - 'classification' (dict): Slang terms and their sentiment classifications.
#                 - 'score' (float): Overall positive score.
#                 - 'name' (str): Name of the analysis subject.
#                 - 'game_slang' (list): List of slang terms and their coefficients.
#         n_words (int): Number of top positive and negative slang terms to display.

#     Raises:
#         ValueError: If required keys are missing in the slang_class dictionary.

#     Notes:
#         - This function relies on external libraries matplotlib and seaborn.
#         - Consider adjusting plot sizes, colors, and annotations for better visualization.
#         - Explore alternative visualization techniques (e.g., word clouds, scatter plots).

#     Examples:
#         >>> slang_analysis = {'classification': {'lit': 'good', 'meh': 'bad'}, 'score': 0.75, 'name': 'Game X', 'game_slang': [('lit', 0.4), ('meh', -0.2)]}
#         >>> grpahs(slang_analysis, 1)
#     """

#     #Pie Chart
#     df_class = pd.DataFrame(slang_class['classification'], index=[0])
#     stk = df_class.stack().value_counts()
#     keys = stk.keys()
#     values = list(stk)

#     fig = plt.figure(figsize=(20,15))
#     ax1 = fig.add_subplot(2,1,1)


#     plt.subplots_adjust(wspace=0.7,hspace=0.7)
#     ax1.pie(x=values,labels=keys, autopct='%.0f%%')
    
#     ax1.annotate(f'Positive Score: {slang_class['score']*100}%',xy=(1,0.5),xytext=(1.5,0), fontsize=20)
#     ax1.annotate(f'Analysis for {slang_class['name']}', xy=(1,1), xytext=(1.5,1), fontsize=20)
#     ax1.set_title('Slang words count',fontsize=30)


#     #Barplots
#     ax2 = fig.add_subplot(2,1,2)
#     n = n_words
#     sc = pd.DataFrame(slang_class['game_slang']).sort_values(by=['coef'])
#     sc_negative = sc[sc['coef'] < 0]
#     sc_positive = sc[sc['coef'] > 0]
#     sns.barplot(x=sc_negative['words'][:n],y=sc['coef'][:n], color='blue')
#     sns.barplot(x=sc_positive['words'][-n:],y=sc['coef'][-n:], color='red')
#     xlabels = ax2.get_xticklabels()
#     ax2.set_xticklabels(xlabels,rotation=90)

#     plt.subplots_adjust(wspace=0.5,hspace=0.5)
#     plt.show()

def quality_review_finder(reviews:dict, model_results, n_score:int):
    reviews = pd.DataFrame(reviews)
    game_slang = game_slang_filter(reviews_df=reviews) #Get slang words present in reviews
    #Model
    words_df = pd.DataFrame(model_results['words'])
    game_slang_dict = words_df[words_df['words'].isin(game_slang)].reset_index(drop=True)


    
    analysis_df = dict(reviews[['review']])
    ##USE COUNTVECTORIZER
    params = {
        'stop_words':'english',
        'lowercase':True,
        'strip_accents':'ascii',
        'ngram_range':(1,3)
    }
    count = CountVectorizer(**params)

    new_rev = [] #list used to store the list of feature words obtained with countevectorizer
    new_rev_2 = [] #list used to store the list of slang words within the list elements in new_rev 
    nonzero = [] #stores the length of the list elements of new_rev_2
    for rev in analysis_df['review']:
        try:
            count.fit_transform([rev])
            new_rev.append(count.get_feature_names_out())
        except ValueError:
            new_rev.append([])
    
    for e in new_rev:
        new_rev_2.append(list(set(list(game_slang_dict['words'])).intersection(e)))
        nonzero.append(len(list(set(list(game_slang_dict['words'])).intersection(e))))
        

    analysis_df['review'] = new_rev_2
    analysis_df['nonzero'] = nonzero
    analysis_df = pd.DataFrame(analysis_df) #Dataframe that in the column 'reviews' hast as elements lists of slang words present in the reviews
    #A lot of them are empty
    df = analysis_df[analysis_df['nonzero']>0] #Clear the rows with empty lists
    df = df.sort_values(by=['nonzero'])
    indexes = list(df[df['nonzero']>n_score].index) #Indexes of the df that fullfill the condition
    quality_reviews = reviews.loc[indexes,['review']]
    ##
    word_list = set([])
    for e in df['review']:
        word_list = word_list.union(set(e))
    
    best_model = model_results['model']
    scores = best_model.predict(list(quality_reviews['review']))
    
    return {'reviews':list(quality_reviews['review']), 'slang_words':list(word_list), 'scores':scores}


def quality_analysis(quality_rev):
    df_dict = {'reviews':quality_rev['reviews'], 'scores':quality_rev['scores']}
    df_dict = pd.DataFrame(df_dict).replace(to_replace=0.0, value=-1.0)
    quality_word_scores = {}
    distribution_values = {}
    for word in quality_rev['slang_words']:
        quality_word_scores[f'{word}'] = 0
        distribution_values[f'{word}'] = 0
        

    for key,value in quality_word_scores.items():

        
        for i in list(df_dict.index):
            
            if (key in df_dict.loc[i,'reviews']):
            
                value = value + df_dict.loc[i,'scores']
                distribution_values[f'{key}'] = distribution_values[f'{key}'] + 1
        
        quality_word_scores[f'{key}'] = value 
    return {'quality_word_scores':quality_word_scores, 'distribution_values':distribution_values}


def quality_analysis_plotter(quality_word_scores, n_words):
    ws_df = pd.DataFrame(quality_word_scores, index=[0]).transpose().rename(columns={0:'score'})
    ws_df['words'] = ws_df.index
    ws_df = ws_df.sort_values(by=['score']) 
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    sns.barplot(x=ws_df[ws_df['score']<0]['words'][:n_words], y=ws_df[ws_df['score']<0]['score'][:n_words], color='blue')
    sns.barplot(x=ws_df[ws_df['score']>0]['words'][-n_words:], y=ws_df[ws_df['score']>0]['score'][-n_words:], color='red')

    xlabels = ax.get_xticklabels()
    ax.set_xticklabels(xlabels,rotation=90)
    plt.show()

def quality_distribution_plotter(quality_word_scores):
    
    ws_df = pd.DataFrame(quality_word_scores, index=[0]).transpose().rename(columns={0:'score'})
    ws_df['words'] = ws_df.index
    ws_df = ws_df.sort_values(by=['score']) 
    ws_df = ws_df[ws_df['score']>0]

    #PLOT
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    keys = ws_df['words']
    values = ws_df['score']
    sns.barplot(x=keys,y=values,color='green')
    xlabels = ax.get_xticklabels()
    ax.set_xticklabels(xlabels,rotation=90)
    plt.show()

def general_pie_graph(slang_class):

    #Pie Chart
    df_class = pd.DataFrame(slang_class, index=[0])
    stk = df_class.stack().value_counts()
    keys = stk.keys()
    values = list(stk)

    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(2,1,1)


    plt.subplots_adjust(wspace=0.7,hspace=0.7)
    ax1.pie(x=values,labels=keys, autopct='%.0f%%')
    ax1.set_title('Slang words count',fontsize=30)
    plt.show()