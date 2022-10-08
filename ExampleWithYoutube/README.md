# Демонстрация базовых приемов при проведении сравнительного анализа тональности комментариев в YouTube 

<img src = 'https://github.com/NikitiusIvanov/russian_youtube_coments_sentimen_analysis/blob/main/visualization/kde.png' width = '900'>

<img src = 'https://github.com/NikitiusIvanov/russian_youtube_coments_sentimen_analysis/blob/main/visualization/download.png' width = '800'>

[Тетрадка jupyter-notebook с решением задачи](https://github.com/NikitiusIvanov/russian_youtube_coments_sentimen_analysis/blob/main/sentiment_analysis.ipynb)

Для решения задачи мы будем обучать бинарный классификатор LogisticRegression на векторных представлениях TF-IDF

В процессе построения нашео классификатора мы поучимся :

* Писать парсер комментариев

* Предобрабатывать тексты для их последующего анализа

* Получать частотность слов в наборах текстов

* Создавать красивые "облака слов"

* Находить размеченные датасеты и оценивать их пригодность для задачи

* Получать векторные представления текстов с помощью TF-IDF

* Разделять комментарии на положительные и отрицательные с помощью логистической регрессии 

* Оценивать качество классификации с помощью ROC кривых и матрицы ошибок

* Визуализировать наиболее важные для классификации слова

* Применять полученный классификатор для анализа тональности комментариев
