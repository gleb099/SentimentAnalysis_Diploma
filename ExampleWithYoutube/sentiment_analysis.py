!pip install selenium

!pip install wordcloud

import re
import time
from tqdm import tqdm
 
import numpy as np
import pandas as pd
 

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from selenium import webdriver

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from pymystem3 import Mystem
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import *

"""## Парсинг текстов комментариев"""

#Зададим путь к основной папке
path_main = r"C:\Users\IvanovNikita"\
                      "\OneDrive - ООО «АЛЬМА Сервисез Компани»"\
                        "\Рабочий стол\data-utilitarian"\
                        "\Sentiment analysis"

#Зададим путь к папке с chromedriver
path_chrome_driver = r"C:\Users\IvanovNikita"\
                      "\OneDrive - ООО «АЛЬМА Сервисез Компани»"\
                        "\Рабочий стол\data-utilitarian"\
                        "\Sentiment analysis\chromedriver_win32"\
                        "\chromedriver.exe"

"""### Парсинг текстов комментариев видео 1"""

webdriver.support.ui.WebDriverWait

webdriver.support.wait.WebDriverWait()

scrapped = []

with webdriver.Chrome(executable_path=path_chrome_driver) as driver:
    wait = webdriver.support.ui.WebDriverWait(driver,1)
    driver.get("https://www.youtube.com/watch?v=wCycCRk_Eak")

    for item in tqdm(range(200)): 
        wait.until(webdriver.support.expected_conditions.visibility_of_element_located(
            (By.TAG_NAME, "body"))).send_keys(webdriver.common.keys.Keys.END)
        time.sleep(2)

    for comment in wait.until(webdriver.support.expected_conditions.presence_of_all_elements_located(
        (By.CSS_SELECTOR, "#content"))):
        scrapped.append(comment.text)

len(scrapped)

comments = [x.split('\nОТВЕТИТЬ')[0].split('\n')[1] for x in scrapped[0].split('назад')][1:]

len(scrapped[1:])

comments_putin = comments + scrapped[1:]

for _ in range(10):

    print(comments_putin[np.random.randint(len(comments_putin))])

comments_putin_df = pd.DataFrame({'comment':comments_putin})

comments_putin_df.to_csv(path_main + '\\comments_putin.csv')

comments_putin_df

"""### Парсинг текстов комментариев видео 2"""

scrapped = []

with webdriver.Chrome(executable_path=path_chrome_driver) as driver:
    wait = webdriver.support.ui.WebDriverWait(driver,1)
    driver.get("https://www.youtube.com/watch?v=Uy4aUXevKEE")

    for item in tqdm(range(200)): 
        wait.until(webdriver.support.expected_conditions.visibility_of_element_located(
            (By.TAG_NAME, "body"))).send_keys(webdriver.common.keys.Keys.END)
        time.sleep(2.5)

    for comment in wait.until(webdriver.support.expected_conditions.presence_of_all_elements_located(
        (By.CSS_SELECTOR, "#content"))):
        scrapped.append(comment.text)

len(scrapped)

comments = [x.split('\nОТВЕТИТЬ')[0].split('\n') for x in scrapped[0].split('назад')]

comments_1 = [x[1] for x in comments if len(x)>1 ][2:]

len(scrapped[1:])

comments_shulman = comments_1 + scrapped[1:]

len(comments_shulman)

"""for _ in range(10):

    print(comments_shulman[np.random.randint(len(comments_shulman))])
"""

comments_shulman_df = pd.DataFrame({'comment':comments_shulman})

comments_shulman_df.to_csv(path_main + '\\comments_shulman.csv')

comments_shulman_df

"""## Загрузка размеченного датасета study.mokoron
Источник https://study.mokoron.com/
"""

positive = pd.read_csv(path_main + '\\russian_sentiment_tweet_automated_labeled\\positive.csv',
                       sep = ';',
                       header= None
                      )

negative = pd.read_csv(path_main + '\\russian_sentiment_tweet_automated_labeled\\negative.csv',
                       sep = ';',
                       header= None
                      )

positive_text = pd.DataFrame(positive.iloc[:, 3])
negative_text = pd.DataFrame(negative.iloc[:, 3])

positive_text['label'] = [1] * positive_text.shape[0]
negative_text['label'] = [0] * negative_text.shape[0]

labeled_tweets = pd.concat([positive_text, negative_text])

labeled_tweets.index = range(labeled_tweets.shape[0])

labeled_tweets.columns = ['text', 'label']
labeled_tweets

"""### Очистка размеченного датасета

Напишем фукнцию для очистки текстов от лишних символов
"""

# Оставим в тексте только кириллические символы
def clear_text(text):
    clear_text = re.sub(r'[^А-яЁё]+', ' ', text).lower()
    return " ".join(clear_text.split())
    

# напишем функцию удаляющую стоп-слова
def clean_stop_words(text, stopwords):
    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)

# загрузим список стоп-слов
stopwords = set(nltk_stopwords.words('russian'))
np.array(stopwords)

# Протестируем работу функции очистки текста
text = labeled_tweets['text'][np.random.randint(labeled_tweets.shape[0])]
print(text)
print('=======================================')
print(clean_stop_words((clear_text(text)), stopwords))

start_clean = time.time()

labeled_tweets['text_clear'] = labeled_tweets['text']\
     c                           .apply(lambda x: clean_stop_words(clear_text(str(x)), stopwords))

print('Обработка текстов заняла: '+str(round(time.time() - start_clean, 2))+' секунд')

labeled_tweets = labeled_tweets[['text_clear', 'label']]
labeled_tweets

labeled_tweets.to_csv(path_main + '\\labeled_tweets_clean.csv')

"""## Загрузка размеченного датасета  Linis Crowd
Источник http://www.linis-crowd.org/
"""

labeled_texts_1 = pd.read_excel(
    path_main + '\\linis_crowd_dataset\\doc_comment_summary.xlsx',
    sheet_name = 0,
    header=None
    )

labeled_texts_1

labeled_texts_1['label'] = pd.to_numeric(labeled_texts_1.iloc[:, 1], errors='coerce')

labeled_texts_1 = labeled_texts_1[[0, 'label']]

labeled_texts_1.columns = ['text', 'label']

labeled_texts_1.label.value_counts()

ind_drop = labeled_texts_1.query('label > 2 or label < - 2').index

ind_drop

labeled_texts_1 = labeled_texts_1.query('index not in @ind_drop')

for _ in range(4):
    
    sample = labeled_texts_1.sample(n = 1)
    
    print('label: ', sample.label.values[0])
    
    print(sample['text'].values[0][:200]) 
    
    print()

selected = labeled_texts_1.query('label != 0')

selected.label.value_counts()

selected.loc[:, 'label_binary'] = np.nan

selected.loc[((selected['label'] == -1) |
         (selected['label'] == -2)), 'label_binary'] = 0

selected.loc[((selected['label'] == 1) |
         (selected['label'] == 2)), 'label_binary'] = 1

selected.label_binary.value_counts()

for _ in range(3):
    
    sample_neg = selected.query('label_binary == 0').sample(n = 1)
    
    sample_pos = selected.query('label_binary == 1').sample(n = 1)
    
    print('label: ', sample_pos.label_binary.values[0])
    
    print(sample_pos['text'].values[0][:200]) 
    
    print('label: ', sample_neg.label_binary.values[0])
    
    print(sample_neg['text'].values[0][:200]) 
    
    print()

"""## Лемматизация текстов"""

start_clean = time.time()

selected['text_clear'] = selected['text']\
                               .apply(lambda x:
                                      clean_stop_words(
                                        clear_text(str(x)),
                                        stopwords))

print('Обработка текстов заняла: '+str(round(time.time() - start_clean, 2))+' секунд')

def lemmatize(df : (pd.Series, pd.DataFrame),
              text_column : (None, str),
              n_samples : int,
              break_str = 'br',
             ) -> pd.Series:
    """
    Принимает:
    df -- таблицу или столбец pandas содержащий тексты,
    text_column -- название столбца указываем если передаем таблицу,
    n_samples -- количество текстов для объединения,
    break_str -- символ разделения, нужен для ускорения,
    количество текстов записанное в n_samples объединяется 
    в одит большой текст с предварительной вставкой символа 
    записанного в break_str между фрагментами
    затем большой текст лемматизируется, после чего разбивается на
    фрагменты по символу break_str
    
    
    Возвращает:
    Столбец pd.Series с лемматизированными текстами
    в которых все слова приведены к изначальной форме:
    * для существительных — именительный падеж, единственное число;
    * для прилагательных — именительный падеж, единственное число,
    мужской род;
    * для глаголов, причастий, деепричастий — глагол в инфинитиве 
    (неопределённой форме) несовершенного вида.
    
    """
    
    result = []
    
    m = Mystem()    
    
    for i in tqdm(range((df.shape[0] // n_samples) + 1)) :
        
        start = i * n_samples
        
        stop = start + n_samples
        
        sample = break_str.join(df[text_column][start : stop].values)
        
        lemmas = m.lemmatize(sample)
        
        lemm_sample = ''.join(lemmas).split(break_str)
        
        result += lemm_sample
        
    return pd.Series(result, index = df.index)

selected['lemm_clean_tex'] = lemmatize(
    df = selected,
    text_column = 'text_clear',
    n_samples = 100,
    break_str = 'br',
    )

selected.head()

labeled_tweets['lemm_text_clear'] = lemmatize(
    df = labeled_tweets,
    text_column = 'text_clear',
    n_samples = 1000,
    break_str = 'br',
    )

labeled_tweets

"""## Объединение двух наборов текстов"""

sample_1 = labeled_tweets[['text_clear', 'lemm_text_clear', 'label']]

sample_2 = selected[['text_clear', 'lemm_clean_tex', 'label_binary']]

sample_2.columns = ['text_clear','lemm_text_clear', 'label']

sample_2.isna().sum()

sample_2 = sample_2.dropna()
sample_2.isna().sum()

joned_text = pd.concat([sample_1, sample_2])

joned_text

joned_text.label.value_counts()

joned_text.columns = ['text', 'lemm_text', 'label']

joned_text.index = range(joned_text.shape[0])

joned_text.isna().sum()

joned_text = joned_text.dropna()
joned_text.isna().sum()

"""## Получение TF-IDF векторных представлений размеченных текстов

Для обучения классификатора получим значения IDF (inference document frequency) для слов из тренировочного набора данных, значения IDF равны логарифму отношения количества документов к количеству документов содержащих искомое слово. Например для стандартных слов, которые встречаются практически в любом тексте IDF будет близок к единице, а для специфичных, которые встречаются в одном тексте из 100 это значение будет равно уже 2 (если мы берем основание логарифма 10).

Затем получив словарь со значениями IDF мы можем получить векторное представление каждого текста по следующему принципу, значения IDF слова умножаем на значения
"""

sample_2.columns = ['text', 'text_lemm', 'label']

sample_1.columns = ['text', 'text_lemm', 'label']

# предварительно разделим выборку на тестовую и обучающую
train, test = train_test_split(sample_1,
                        test_size = 0.2,
                        random_state = 12348,
                       )

print(train.shape)
print(test.shape)

train

# Сравним распределение целевого признака
for sample in [train, test]:    
    print(sample[sample['label'] == 1].shape[0] / sample.shape[0])

count_idf_positive = TfidfVectorizer(ngram_range = (1,1))
count_idf_negative = TfidfVectorizer(ngram_range = (1,1))

tf_idf_positive = count_idf_positive.fit_transform(train.query('label == 1')['text'])
tf_idf_negative = count_idf_negative.fit_transform(train.query('label == 0')['text'])

# Сохраним списки Idf для каждого класса
positive_importance = pd.DataFrame(
    {'word' : count_idf_positive.get_feature_names_out(),
     'idf' : count_idf_positive.idf_
    }).sort_values(by = 'idf', ascending = False)

negative_importance = pd.DataFrame(
    {'word' : count_idf_negative.get_feature_names_out(),
     'idf' : count_idf_negative.idf_
    }).sort_values(by = 'idf', ascending = False)

positive_importance.query('word not in @negative_importance.word and idf < 10.8')

negative_importance.query('word not in @positive_importance.word and idf < 10')

fig = plt.figure(figsize = (12,5))
positive_importance.idf.hist(bins = 100,
                             label = 'positive',
                             alpha = 0.5,
                             color = 'b',
                            )
negative_importance.idf.hist(bins = 100,
                             label = 'negative',
                             alpha = 0.5,
                             color = 'r',
                            )
plt.title('Распределение биграмм по значениям TF-IDF')
plt.xlabel('TF-IDF')
plt.ylabel('Количество слов')
plt.legend()
plt.show()

"""## Предварительное обучение моделей"""

# Получим векторные представления текстов
count_idf_1 = TfidfVectorizer(ngram_range=(1,1))

tf_idf_base_1 = count_idf_1.fit(sample_1['text'])
tf_idf_train_base_1 = count_idf_1.transform(train['text'])
tf_idf_test_base_1 = count_idf_1.transform(test['text'])

display(tf_idf_test_base_1.shape)
display(tf_idf_train_base_1.shape)

model_lr_base_1 = LogisticRegression(solver = 'lbfgs',
                                    random_state = 12345,
                                    max_iter= 10000,
                                    n_jobs= -1)

"""Получим прогноз и оценим качество модели"""

model_lr_base_1.fit(tf_idf_train_base_1, train['label'])

predict_lr_base_proba = model_lr_base_1.predict_proba(tf_idf_test_base_1)

"""### Сравнение качества классификации на лемматизированных текстах"""

# Получим векторные представления лемматизированных текстов
count_idf_lemm = TfidfVectorizer(ngram_range=(1,1))

tf_idf_lemm = count_idf_lemm.fit(sample_1['text_lemm'])
tf_idf_train_lemm = count_idf_lemm.transform(train['text_lemm'])
tf_idf_test_lemm = count_idf_lemm.transform(test['text_lemm'])

display(tf_idf_train_lemm.shape)
display(tf_idf_test_lemm.shape)

model_lr_lemm = LogisticRegression(solver = 'lbfgs',
                                    random_state = 12345,
                                    max_iter= 10000,
                                    n_jobs= -1)

"""Получим прогноз и оценим качество модели"""

model_lr_lemm.fit(tf_idf_train_lemm, train['label'])

predict_lr_lemm_proba = model_lr_lemm.predict_proba(tf_idf_test_lemm)

"""В качестве сравнения сделаем классификатор который в качестве прогноза выдает случайное число в интервале от 0 до 1 """

def coin_classifier(X:np.array) -> np.array:
    predict = np.random.uniform(0.0, 1.0, X.shape[0])
    return predict

coin_predict = coin_classifier(tf_idf_test_base_1)

fif = plt.figure(figsize = (8, 6))

pd.Series(coin_predict)\
                .hist(bins = 100,
                      alpha = 0.7,
                      color = 'r',
                      label = 'TF-IDF LogisticRegression'
                     )

pd.Series(predict_lr_base_proba[:, 1])\
                .hist(bins = 100,
                      alpha = 0.7,
                      color = 'b',
                      label = 'Coin'
                     )
plt.legend()   
plt.show()

"""### Визуализация ROC-кривых классификаторов"""

fpr_base, tpr_base, _ = roc_curve(test['label'], predict_lr_base_proba[:, 1])
roc_auc_base = auc(fpr_base, tpr_base)

fpr_lemm, tpr_lemm, _ = roc_curve(test['label'], predict_lr_lemm_proba[:, 1])
roc_auc_lemm = auc(fpr_lemm, tpr_lemm)

fpr_coin, tpr_coin, _ = roc_curve(test['label'], coin_predict)
roc_auc_coin = auc(fpr_base, tpr_base)

fig = make_subplots(1,1,
                    subplot_titles = ["Receiver operating characteristic"],
                    x_title="False Positive Rate",
                    y_title = "True Positive Rate"
                   )

fig.add_trace(go.Scatter(
    x = fpr_lemm,
    y = tpr_lemm,
    fill = 'tozeroy',
    name = "ROC lemm (area = %0.3f)" % roc_auc_lemm,
    ))

fig.add_trace(go.Scatter(
    x = fpr_base,
    y = tpr_base,
    #fill = 'tozeroy',
    name = "ROC base (area = %0.3f)" % roc_auc_base,
    ))

fig.add_trace(go.Scatter(
    x = fpr_coin,
    y = tpr_coin,
    mode = 'lines',
    line = dict(dash = 'dash'),
    name = 'Coin classifier (area = 0.5)'
    ))


fig.update_layout(
    height = 600,
    width = 800,
    xaxis_showgrid=False,
    xaxis_zeroline=False,
    template = 'plotly_dark',
    font_color = 'rgba(212, 210, 210, 1)'
    )

# Выведем матрицы ошибок
confusion_matrix(test['label'],
                 (predict_lr_base_proba[:, 1] > 0.5).astype('float'),
                 normalize='true',
                )

# Выведем матрицу ошибок
confusion_matrix(test['label'],
                 (predict_lr_lemm_proba[:, 1] > 0.5).astype('float'),
                 normalize='true',
                )

"""### Визуализация важности признаков"""

# Получим веса признаков, то есть множители 
# подобранные логистической регрессией 
# для каждого компонента вектора tf-idf

weights = pd.DataFrame({'words': count_idf_1.get_feature_names_out(),
                        'weights': model_lr_base_1.coef_.flatten()})
weights_min = weights.sort_values(by= 'weights')
weights_max = weights.sort_values(by= 'weights', ascending = False)

weights_min = weights_min[:100]
weights_min['weights'] = weights_min['weights'] * -1
weights_min

weights_max = weights_max[:100]
weights_max

# Воспользуемся библиотекой wordcloud для генерации картинок

wordcloud_positive = WordCloud(background_color="white",
                               colormap = 'Blues',
                               max_words=200,
                               mask=None, 
                                width=1600,
                               height=1600)\
                        .generate_from_frequencies(
                            dict(weights_max.values))

wordcloud_negative = WordCloud(background_color="black",
                               colormap = 'Reds',
                               max_words=200,
                               mask=None, 
                                width=1600,
                               height=1600)\
                        .generate_from_frequencies(
                            dict(weights_min.values))

# Выведем картинки сгенерированные вордклаудом
fig, ax = plt.subplots(1, 2, figsize = (20, 12))


ax[0].imshow(wordcloud_positive, interpolation='bilinear')
ax[1].imshow(wordcloud_negative, interpolation='bilinear')

ax[0].set_title('Топ ' +\
                str(weights_max.shape[0]) +\
                ' слов\n с наибольшим положительным весом',
               fontsize = 20
               )
ax[1].set_title('Топ ' +\
                str(weights_min.shape[0]) +\
                ' слов\n с наибольшим отрицательным весом',
               fontsize = 20
               )

ax[0].axis("off")
ax[1].axis("off")

plt.show()

"""### Снижение размерности признакового пространства модели"""

fig = make_subplots(1,1)

fig.add_trace(go.Histogram(
    x = weights.query('weights != 0')['weights'],
    #histnorm = 'probability',
    opacity = 0.5,
    showlegend = False
))

fig.add_trace(go.Histogram(
    x = weights.query('weights > 0.25 or weights < -0.25')['weights'],
    #histnorm = 'probability',
    opacity = 0.5,
    showlegend = False
))

fig.update_layout(
    height = 600,
    width = 800,
    xaxis_showgrid=False,
    xaxis_zeroline=False,
    template = 'plotly_dark',
    font_color = 'rgba(212, 210, 210, 1)'
    
)

vocab = weights.query('weights > 0.25 or weights < -0.25')['words']

vocab

"""Получим векторные представления текстов"""

count_idf = TfidfVectorizer(vocabulary=vocab,
                            ngram_range=(1,1))

tf_idf = count_idf.fit_transform(joned_text['text'])

tf_idf_train = count_idf.transform(train['text'])
tf_idf_test = count_idf.transform(test['text'])

display(tf_idf_test.shape)
display(tf_idf_train.shape)

model_lr_base = LogisticRegression(solver = 'lbfgs',
                                    random_state = 12345,
                                    max_iter= 10000,
                                    n_jobs= -1)

"""Получим прогноз и оценим качество модели"""

model_lr_base.fit(tf_idf_train, train['label'])

predict_lr_base_proba_1 = model_lr_base.predict_proba(tf_idf_test)

fpr_base_1, tpr_base_1, _ = roc_curve(test['label'], predict_lr_base_proba_1[:, 1])
roc_auc_base_1 = auc(fpr_base_1, tpr_base_1)

fig = make_subplots(1,1,
                    subplot_titles = ["Receiver operating characteristic"],
                    x_title="False Positive Rate",
                    y_title = "True Positive Rate"
                   )

fig.add_trace(go.Scatter(
    x = fpr_base,
    y = tpr_base,
    fill = 'tozeroy',
    name = "ROC curve (area = %0.3f)" % roc_auc_base,
    ))

fig.add_trace(go.Scatter(
    x = fpr_base_1,
    y = tpr_base_1,
    fill = 'tozeroy',
    name = "Less dimensity ROC curve (area = %0.3f)" % roc_auc_base_1,
    ))

fig.add_trace(go.Scatter(
    x = fpr_coin,
    y = tpr_coin,
    mode = 'lines',
    line = dict(dash = 'dash'),
    name = 'Coin classifier'
    ))


fig.update_layout(
    height = 600,
    width = 800,
    xaxis_showgrid=False,
    xaxis_zeroline=False,
    template = 'plotly_dark',
    font_color = 'rgba(212, 210, 210, 1)'
    )

"""Вывод:

Мы снизили размерность векторов tf-idf потеряв при этом 0.3% качества (площадь под ROC кривой при размерности > 170К 0.809, площадь под ROC кривой при размерности > 48K -- 0.806)

### Подбор оптимального порогового значения классификации
"""

scores = {}

weight = 0.55

for threshold in np.linspace(0, 1, 100):
    
    matrix = confusion_matrix(test['label'],
                 (predict_lr_base_proba[:, 0] <  threshold).astype('float'),
                 normalize='true',
                )

    score = matrix[0,0] * weight + matrix[1,1] * (1 - weight)
    
    scores[threshold] = score

pd.DataFrame({'true_score':scores.values(),
             'threshold':scores.keys()},
             ).sort_values(by = 'true_score', ascending = False).head()

matrix = confusion_matrix(test['label'],
                 (predict_lr_base_proba[:, 0] <  0.444444).astype('int'),
                 normalize='true',
                )
matrix

# Сделаем красивый график матрицы ошибок

fig = make_subplots(1,1)

fig.add_trace(go.Heatmap(
     y = [ 'positive', 'negative'],
     x = ['predicted_negative', 'predicted_positive'],
     z = [matrix[1, :], matrix[0, :]],
     colorscale = 'PuBu'
))

fig.add_trace(go.Heatmap(
     y = [ 'positive', 'negative'],
     x = ['predicted_negative', 'predicted_positive'],
     z = [matrix[1, :], matrix[0, :]],
     colorscale = 'PuBu'
))

fig.add_annotation(x=0, y=0,
            text = "false negative " + str(round(matrix[1,0], 2)),
            showarrow=False,
            font = dict(color = 'black'),
            yshift=10)

fig.add_annotation(x=1, y=1,
            text = "false positive " + str(round(matrix[0,1], 2)),
            showarrow=False,
            font = dict(color = 'black'),
            yshift=10)

fig.add_annotation(x=1, y=0,
            text = "true positive " + str(round(matrix[1,1], 2)),
            showarrow=False,
            font = dict(color = 'white'),
            yshift=10)

fig.add_annotation(x=0, y=1,
            text = "true negative " + str(round(matrix[0,0], 2)),
            showarrow=False,
            font = dict(color = 'white'),
            yshift=10)


fig.update_layout(height = 500,
                  width = 500,
                  template = 'plotly_dark',
                  font_color = 'rgba(212, 210, 210, 1)',
                 ).show()

"""## Классификация не размеченных комментариев

C помощью обученных tf-idf векторизатора и логистической регрессии получим оценки вероятности негатива в каждом из комментариев
"""

# очистим тексты комментариев под первым видео
start_clean = time.time()

comments_putin_df['text_clear'] = comments_putin_df['comment']\
                                .apply(lambda x: clean_stop_words(clear_text(str(x)), stopwords))

print('Обработка текстов заняла: '+str(round(time.time() - start_clean, 2))+' секунд')

# очистим тексты омментариев под вторым видео
start_clean = time.time()

comments_shulman_df['text_clear'] = comments_shulman_df['comment']\
                                .apply(lambda x: clean_stop_words(clear_text(str(x)), stopwords))

print('Обработка текстов заняла: '+str(round(time.time() - start_clean, 2))+' секунд')

"""### Визуализация ключевых слов"""

shulman_counter = CountVectorizer(ngram_range=(1, 1))
putin_counter = CountVectorizer(ngram_range=(1, 1))

shulman_count = shulman_counter.fit_transform(comments_shulman_df['text_clear'])
putin_count = putin_counter.fit_transform(comments_putin_df['text_clear'])

shulman_count.toarray().sum(axis = 0).shape

shulman_counter.get_feature_names_out().shape

# Сохраним списки Idf для каждого класса
shulman_frequence = pd.DataFrame(
    {'word' : shulman_counter.get_feature_names_out(),
     'frequency' : shulman_count.toarray().sum(axis = 0)
    }).sort_values(by = 'frequency', ascending = False)

putin_frequence = pd.DataFrame(
    {'word' : putin_counter.get_feature_names_out(),
     'frequency' : putin_count.toarray().sum(axis = 0)
    }).sort_values(by = 'frequency', ascending = False)

display(shulman_frequence.shape[0])
display(putin_frequence.shape[0])

putin_frequence = putin_frequence.query('word not in @shulman_importance.word')[:100]

shulman_frequence = shulman_frequence.query('word not in @putin_frequence.word')[:100]

# Воспользуемся библиотекой wordcloud для генерации картинок

wordcloud_shulman = WordCloud(background_color="black",
                               colormap = 'Blues',
                               max_words=200,
                               mask=None, 
                                width=1600,
                               height=1600)\
                        .generate_from_frequencies(
                            dict(putin_frequence.values))

wordcloud_putin = WordCloud(background_color="black",
                               colormap = 'Oranges',
                               max_words=200,
                               mask=None, 
                                width=1600,
                               height=1600)\
                        .generate_from_frequencies(
                            dict(shulman_frequence.values))

# Выведем картинки сгенерированные вордклаудом
fig, ax = plt.subplots(1, 2, figsize = (20, 12))


ax[0].imshow(wordcloud_shulman, interpolation='bilinear')
ax[1].imshow(wordcloud_putin, interpolation='bilinear')

ax[0].set_title('Топ ' +\
                str(shulman_frequence.shape[0]) +\
                ' слов наиболее частотных,\n ' +\
                ' уникальных слов в комментариях\n' +\
                ' под поздравлением В. Путина',
               fontsize = 20
               )
ax[1].set_title('Топ ' +\
                str(putin_frequence.shape[0]) +\
                ' слов наиболее частотных,\n ' +\
                ' уникальных слов в комментариях\n' +\
                'под поздравлением Е. Шульман',
               fontsize = 20
               )

ax[0].axis("off")
ax[1].axis("off")

plt.show()

"""## Получение оценки негативности комментария"""

# Выведем 5 случайных комментариев c оценкой негатива первого видео
for _ in range(5):

    source = comments_putin_df.sample(n=1)
    text_clear = source['text_clear'].values[0]
    text = source['comment'].values[0]

    print(text)

    tf_idf_text = count_idf.transform([text_clear])

    toxic_proba = model_lr_base.predict_proba(tf_idf_text)

    print('Вероятность негатива: ', toxic_proba[:, 0])
    print()

# Выведем 5 случайных комментариев c оценкой негатива второго видео
for _ in range(5):

    source = comments_shulman_df.sample(n=1)
    text_clear = source['text_clear'].values[0]
    text = source['comment'].values[0]

    print(text)

    tf_idf_text = count_idf.transform([text_clear])

    toxic_proba = model_lr_base.predict_proba(tf_idf_text)

    print('Вероятность негатива: ', toxic_proba[:, 0])
    print()

# Получим оценки негатива для всех комментариев первого и второго видео

putin_tf_idf = count_idf.transform(comments_putin_df['text_clear'])
shulman_tf_idf = count_idf.transform(comments_shulman_df['text_clear'])

putin_negative_proba = model_lr_base.predict_proba(putin_tf_idf)
shulman_negative_proba = model_lr_base.predict_proba(shulman_tf_idf)

comments_putin_df['negative_proba'] = putin_negative_proba[:, 0]
comments_shulman_df['negative_proba'] = shulman_negative_proba[:, 0]

"""Найдем доли негативных комментариев при оптимальном пороговом значении"""

putin_share_neg = (comments_putin_df['negative_proba'] > 0.44).sum() / comments_putin_df.shape[0]
putin_share_neg

shulman_share_neg = (comments_shulman_df['negative_proba'] > 0.44).sum() / comments_shulman_df.shape[0]
shulman_share_neg

# Commented out IPython magic to ensure Python compatibility.
fig = make_subplots(1,1,
                   subplot_titles=['Распределение комментариев по оценке негативности']     
                   )

fig.add_trace(go.Violin(
    x = comments_shulman_df['negative_proba'],
    meanline_visible = True,
    name = 'Shulman (N = %i)' % comments_putin_df.shape[0],
    side = 'positive',
    spanmode = 'hard'
))



fig.add_trace(go.Violin(
    x = comments_putin_df['negative_proba'],
    meanline_visible = True,
    name = 'Putin (N = %i)' % comments_shulman_df.shape[0],
    side = 'positive',
    spanmode = 'hard'
))


fig.add_annotation(x=0.8, y=1.5,
            text = "%0.2f — доля негативных комментариев (при p > 0.44)"\
#                    % putin_share_neg,
            showarrow=False,
            yshift=10)

fig.add_annotation(x=0.8, y=0.5,
            text = "%0.2f — доля негативных комментариев (при p > 0.44)"\
#                    % shulman_share_neg,
            showarrow=False,
            yshift=10)

fig.update_traces(orientation='h', 
                  width = 1.5,
                  points = False
                 )


fig.update_layout(height = 500,
                  #xaxis_showgrid=False,
                  xaxis_zeroline=False,
                  template = 'plotly_dark',
                  font_color = 'rgba(212, 210, 210, 1)',
                  legend=dict(
                    y=0.9,
                    x=-0.1,
                    yanchor='top',
                    ),
                 )
fig.update_yaxes(visible = False)
              


fig.show()

