import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.matutils import cossim


def get_cluster(candidate_skills, job_titles):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(job_titles)

    num_clusters = 30
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    gh = ""

    for q in candidate_skills:
        gh = gh + q + " "

    new_title = gh
    new_title_vectorized = vectorizer.transform([new_title])
    new_title_cluster = kmeans.predict(new_title_vectorized)[0]
    return new_title_cluster


def get_recommendations(candidate_skills):
    df = pd.read_csv('df_edited.csv')

    job_titles = df['title']
    cluster = get_cluster(candidate_skills, job_titles)

    df['skills'] = df['skills'].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
    # создание датафрейма на данных с определенным значением кластера
    data = df.loc[df['cluster'] == cluster]
    data1 = data['skills']

    # Создание словаря и корпуса
    dictionary = Dictionary(data1)
    corpus = [dictionary.doc2bow(text) for text in data1]

    # Обучение LDA-модели
    num_topics = 20
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    candidate_bow = dictionary.doc2bow(candidate_skills)
    candidate_topic_distribution = lda_model.get_document_topics(candidate_bow)

    # Вычисление сходства с вакансиями
    similarity_scores = []

    for job_skills_bow in corpus:
        job_topic_distribution = lda_model.get_document_topics(job_skills_bow)
        similarity = cossim(candidate_topic_distribution, job_topic_distribution)
        similarity_scores.append(similarity)

    # Рекомендация топ-N вакансий на основе сходства
    N = 10
    top_N_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:N]
    recommended_jobs = [df['title'][i] for i in top_N_indices]

    # print(sorted(similarity_scores, reverse=True)[:N])

    new_df = pd.DataFrame(columns=data.columns)

    for index in top_N_indices:
        new_df = new_df.append(data.iloc[index])

    new_df['similarity'] = sorted(similarity_scores, reverse=True)[:N]
    
    new_df = new_df[['title', 'company', 'requirements', 'description', 'skills', 'cluster', 'similarity']]

    return new_df
