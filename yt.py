# Core Pkg
import streamlit as st 
import streamlit.components.v1 as components
from PIL import Image

# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

import requests
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from operator import itemgetter
from collections import Counter
import matplotlib
import squarify
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import ast
#import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import base64

st.set_page_config(page_title="Course Recommender system", layout="wide")

def add_bg_from_local(image_file):
	with open(image_file, "rb") as image_file:
		encoded_string = base64.b64encode(image_file.read())
	st.markdown(
	f"""
	<style>
	.stApp {{
		background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
		background-size: cover
	}}
	</style>
	""",
	unsafe_allow_html=True
	)

with open('style.css') as f:
	st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load Our Dataset
# def load_data(data):
# 	df = pd.read_csv(data)
# 	return df 

def combine_list(l):
	new_str=""
	for item in l:
		new_str=new_str+' '+str(item) 
	return new_str

def tokenize(text):
	stemmer=SnowballStemmer('english')
	return [stemmer.stem(word) for word in word_tokenize(text.lower())]

def tokenize_only(text):
	return [word for word in word_tokenize(text.lower())]

def vocab_stem(text):
	stemmer=SnowballStemmer('english')
	total_stemmed = []
	total_tokenized = []
	for i in text:
		obj_stemmed = tokenize(i) 
		total_stemmed.extend(obj_stemmed) 
		obj_tokenized = tokenize_only(i)
		total_tokenized.extend(obj_tokenized)
	vocab_frame = pd.DataFrame({'words': total_tokenized}, index = total_stemmed)
	#print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
	return vocab_frame 
  
def drop_words(vocab_frame):
	vocab_frame=vocab_frame.reset_index()
	vocab_frame.columns = ['index','words']
	vocab_frame=vocab_frame.drop_duplicates(subset='index', keep='first').set_index('index')
	#print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
	return vocab_frame

def get_label(index, vocab_frame, word_features):
	return vocab_frame.loc[word_features[index]].values.tolist()[0]

def get_common_words(model, count_words):
	count_words_new=count_words*(-1)-1
	common_words = model.cluster_centers_.argsort()[:,-1:count_words_new:-1]
	return common_words

def print_common_words(common_words, word_features, vocab_frame, print_list=True):
	dict_cluster={}
	for num, centroid in enumerate(common_words):
		dict_cluster[num]=[get_label(word, vocab_frame, word_features) for word in centroid]
		if print_list:
			print(str(num) + ' : ' + ', '.join(dict_cluster[num]))
	if print_list==False:
		return dict_cluster

def squarify_words(common_words, word_features, vocab_frame):
	colormaps=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds','Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
		   'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
	for num, centroid in enumerate(common_words):
		sizes=np.arange(10,10+len(centroid))
		cmap_name=colormaps[num]
		cmap = plt.get_cmap(cmap_name)
		labels=[get_label(word, vocab_frame, word_features) for word in centroid]
		mini=min(sizes)
		maxi=max(sizes)
		norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
		colors = [cmap(norm(value)) for value in sizes]
		squarify.plot(sizes=sizes, label=labels,alpha=0.6, color=colors)
		plt.title("Most frequent words in cluster "+str(num))
		plt.show()

def heatmap_categories_cluster(cluster_name, df_courses, cmap ):
	clusters = df_courses.groupby([cluster_name, 'primary_subcategory']).size()
	fig, ax = plt.subplots(figsize = (30, 15))
	sns.heatmap(clusters.unstack(level = 'primary_subcategory'), ax = ax, cmap = cmap)
	ax.set_xlabel('primary_subcategory', fontdict = {'weight': 'bold', 'size': 24})
	ax.set_ylabel(cluster_name, fontdict = {'weight': 'bold', 'size': 24})
	for label in ax.get_xticklabels():
		label.set_size(16)
		label.set_weight("bold")
	for label in ax.get_yticklabels():
		label.set_size(16)
		label.set_weight("bold")

def get_inertia(data, nClusterRange):
	inertias = np.zeros(len(nClusterRange))
	for i in range(len(nClusterRange)):
		model = KMeans(n_clusters=i+1, init='k-means++', random_state=1234).fit(data)
		inertias[i] = model.inertia_
	return inertias

def plot_inertia(kRange, inertia_Kmean):
	plt.figure(figsize=(10,8))
	plt.plot(kRange, inertia_Kmean, 'o-', color='seagreen', linewidth=3)
	#plt.plot([6], [testKmean[5]], 'o--', color='dimgray', linewidth=3)
	#plt.plot([1,6,11], [8520, 8170,7820], '--', color='k', linewidth=1)
	#plt.annotate("Let's try k=6", xy=(6, testKmean[5]), xytext=(6,7700),
			 #size=14, weight='bold', color='dimgray',
			 #arrowprops=dict(facecolor='dimgray', shrink=0.05))
	plt.xlabel('k [# of clusters]', size=18)
	#plt.ylabel('Inertia', size=14)
	#plt.title('Inertia vs KMean Parameter', size=14)

def print_titles_cluster(n_title, df_courses, cluster_name):
	for i in df_courses[cluster_name].unique():
		temp=df_courses[df_courses[cluster_name]==i]
		print(temp['published_title'].values[:n_title])

#functions for hierarchical clustering:
def get_linkage(X ):
	dist=pdist(X.todense(), metric='euclidean')
	z = linkage(dist, 'ward')
	return z

def plot_dendrogram(z, last_p_show, line_dist=None):
	# lastp is telling the algorithm to truncate using the number of clusters we set
	plt.figure(figsize=(20,10))
	plt.title('Dendrogram for attribute objectives')
	plt.xlabel('Data Index')
	plt.ylabel('Distance (ward)')
	dendrogram(z, orientation='top', leaf_rotation=90, p=last_p_show, truncate_mode='lastp', show_contracted=True);
	if line_dist!=None:
		plt.axhline(line_dist, color='k')

def plot_with_pca (X, labels, plot_n_sample):
	pca=PCA(n_components=2)
	X_2d=pca.fit_transform(X.todense())
	print('The explained variance through the first 2 principal comonent is {}.'
		  . format(round(pca.explained_variance_ratio_.sum(),4)))
	df = pd.DataFrame(dict(x=X_2d[:,0], y=X_2d[:,1], label=labels)) 
	df_sample=df.sample(plot_n_sample)
	groups = df_sample.groupby('label')
	cluster_colors=['b', 'g', 'y','r', 'k', 'grey', 'purple','orange', 'pink', 'brown']
	fig, ax = plt.subplots(figsize=(17, 9)) 
	for name in np.arange(len(df_sample['label'].unique())):
		temp=df_sample[df_sample['label']==name]
		ax.plot(temp.x, temp.y, marker='o', linestyle='', ms=12, 
			label='cluster '+str(name), 
			color=cluster_colors[name], 
			mec='none', alpha=0.6)
		ax.set_aspect('auto')
		ax.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
		ax.tick_params(axis= 'y', which='both', bottom='off', top='off', labelbottom='off')
	ax.legend(numpoints=1) 
	plt.title('Courses with PCA decompostion')

def plot_common_words(model, n_words, word_features, vocab_frame, df_courses, cluster_name):
	common_words=get_common_words(model, n_words)
	dict_cluster=print_common_words(common_words, word_features, vocab_frame, False)
	fig, ax=plt.subplots(figsize=(12,5))
	keys=df_courses[cluster_name].value_counts().sort_index().index
	values=df_courses[cluster_name].value_counts().sort_index().values
	colors=['b', 'g', 'y','r', 'k', 'grey', 'purple','orange', 'pink', 'brown']
	for j in range(len(keys)):
		ax.bar(keys[j], values[j], width=0.8, bottom=0.0, align='center', color=colors[j], alpha=0.4, label=dict_cluster[j]) 
	ax.set_xticks(np.arange(len(values)))
	ax.set_xticklabels(['cluster '+str(k) for k in keys])
	ax.set_ylabel('Number of courses')
	ax.set_title('Distribution of clusters with the top ' + str(n_words) + ' words')
	plt.legend(fontsize=13)

#functions for the recommender system
def normalize_features(df):
	df_norm = df.copy()
	for col in df_norm.columns:
		df_norm[col] = StandardScaler().fit_transform(df_norm[col].values.reshape(-1, 1))
	return df_norm

@st.cache
def recommend_courses(course_id, n_courses, df_courses, df_norm):
	# st.write(course_id)
	# st.write(n_courses)
	y_pred = df_courses['primary_subcategory']
	n_courses=n_courses+1
	id_=df_courses[df_courses['id']==course_id].index.values
	title=df_courses[df_courses['id']==course_id]['published_title']
	y_true = [df_courses.loc[df_courses['id']==course_id]['primary_subcategory'].item()] * (n_courses-1)
	X = df_norm.values
	Y = df_norm.loc[id_].values.reshape(1, -1)
	cos_sim = cosine_similarity(X, Y)
	df_sorted=df_courses.copy()
	df_sorted['cosine_similarity'] = cos_sim
	df_sorted=df_sorted.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
	
	return title, y_pred, y_true, df_sorted.iloc[1:n_courses][['id','published_title', 'cosine_similarity', 'primary_subcategory', 'price', 'objectives']]

@st.cache
def recommend_courses_search(search_text, n_courses, df_courses, df_norm, df):

	flag_sum = df['title'].str.lower().str.contains(str(search_text).lower()).sum()
	if flag_sum > 1:
		#print("Lots of it")
		return flag_sum, False, False, False, df[df['title'].str.lower().str.contains(str(search_text).lower())]

	elif flag_sum == 1:
		#print(df_courses['published_title'].str.contains(search_text).sum())
		#print(df_courses[df_courses['published_title'].str.contains(search_text)]['published_title'])
		y_pred = df_courses['primary_subcategory']
		n_courses=n_courses+1
		id_=df_courses[df_courses['id'].isin(df[df['title'].str.lower().str.contains(str(search_text).lower())]['id'])].index.values
		title=df_courses[df_courses['id'].isin(df[df['title'].str.lower().str.contains(str(search_text).lower())]['id'])]['published_title']
		y_true = [df_courses.loc[df_courses['id'].isin(df[df['title'].str.lower().str.contains(str(search_text).lower())]['id'])]['primary_subcategory'].item()] * (n_courses-1)
		X = df_norm.values
		Y = df_norm.loc[id_].values.reshape(1, -1)
		cos_sim = cosine_similarity(X, Y)
		df_sorted=df_courses.copy()
		df_sorted['cosine_similarity'] = cos_sim
		df_sorted=df_sorted.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)

		return flag_sum, title, y_pred, y_true, df_sorted.iloc[1:n_courses][['id','published_title', 'cosine_similarity', 'primary_subcategory', 'price', 'objectives']]

	else:
		return flag_sum, False, False, False, False


def recommend_for_user(user_name, n_courses, df_reviews, df_courses, df_norm):
	list_courses=df_reviews[df_reviews['user_name']==user_name]['course_id'].values
	len_courses=len(list_courses)
	index_courses=df_courses[df_courses['id'].isin(list_courses)].index
	for course_id in list_courses:
		title, df_recommend= recommend_courses(course_id, n_courses, df_courses, df_norm)
		print('The following courses are recommended after taking the course {} with the id {}:'
		  .format(title.values[0],course_id))
		print(df_recommend)
		print()
	if len_courses>1:
		n_courses=n_courses+1
		df_temp=df_courses.copy()
		for i, course_id in enumerate(list_courses):
			id_=df_courses[df_courses['id']==course_id].index.values
			X = df_norm.values
			Y = df_norm.loc[id_].values.reshape(1, -1)
			cos_sim = cosine_similarity(X, Y)
			df_temp[i] = cos_sim
		temp_avg=df_temp.iloc[:,-len_courses:].mean(axis=1).values
		df_temp['avg_cos_sim']=temp_avg
		df_temp.drop(index=index_courses, inplace=True)
		df_temp=df_temp.sort_values('avg_cos_sim', ascending=False).reset_index(drop=True)
		print('The following courses are recommended after all taken courses:')
		print(df_temp.iloc[1:n_courses][['published_title', 'avg_cos_sim']])

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">????Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">????</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">????Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">?????????????????????????? Students:</span>{}</p>
</div>
"""

# Search For Course 
@st.cache
def search_term_if_not_found(term,df):
	result_df = df[df['id'].str.contains(term)]
	return result_df

def initialize_course_widget(no):
	course_cols = st.columns(no)
	for c in course_cols:
		with c:
			st.empty()

	return course_cols

def fetch_course_details(result, df):
	title_list = []
	image = []
	url = []
	inst = []
	ratings = []
	for i in range(len(result)):
		image.append(df[df['id']==result['id'].values[i]]['image_200_H'].values[0])
		title_list.append(df[df['id']==result['id'].values[i]]['title'].values[0])
		url.append('https://www.udemy.com' + df[df['id']==result['id'].values[i]]['url'].values[0])
		inst.append(df[df['id']==result['id'].values[i]]['instructional_level_simple'].values[0])
		ratings.append(df[df['id']==result['id'].values[i]]['rating'].values[0])

	return title_list, image, url, inst, ratings

def main():

	st.title("Analytical Study on Course Recommendation System for Online Courses")

	menu = ["Home", "Recommend by CourseID", "Recommend by Title", "About"]
	choice = st.sidebar.selectbox("Menu",menu)

	df = pd.read_csv("df_courses3.csv")
	df = df.drop_duplicates(subset='id', keep="first")

	df_courses=pd.read_csv('df_courses_recomm.csv', index_col=0)
	df_reviews=pd.read_csv('df_reviews_cleaned.csv', index_col=0)
	df_norm = pd.read_csv('df_normalised.csv', index_col=0)

	if choice == "Home":
		st.subheader("Home")
		st.header('Description')
		st.markdown('Framework for online course business providers that can be integrated with their platforms to recommend significant courses to customers. This system can also be used as a standalone application by users as well as providers to visualize and perform analytics based on the results of the recommendation models. The providers can gain insight into the relationship between the recommendation systems and related courses. Using cosine similarity and content based recommendation and machine learning techniques to build models and recommendation engines. With the necessary results and statistics, the providers can have a competitive edge in the eLearning sphere')

		st.dataframe(df)
	
	# if choice =="EDA":
	# 	st.subheader('Udemy Dataset')
	# 	st.markdown('Data about Udemy was also gathered from Kaggle, which concentrates on four different topics pertaining to business finance, graphic design, musical instruments, and web design offered on the platform. Course Difficulty and the number of reviews for the course are the key elements in this analysis, which will aid in the development of the recommendation model.')

	# 	image = Image.open('Udemy_Summary1.png')
	# 	st.image(image, caption='Summary and Statistics for Udemy Dataset')


	# 	image = Image.open('Udemy6.png')
	# 	st.image(image, caption='Course Ratiings')
	# 	st.markdown('There are around 900 courses with no reviews/ratings, but most of the ratings are between rating 4 and 4.5.')

	# 	image = Image.open('Udemy1.png')
	# 	st.image(image, caption='Price range of the courses')
	# 	st.markdown('The bar chart shows the price range of the courses. The price ranges between 0 and 199 EUR. Most courses cost eiter 19.99 or 199.99 $.')

	# 	image = Image.open('Udemy2.png')
	# 	st.image(image, caption='Courses with high number of subscribers')
	# 	st.markdown('The plot shows courses that are most visited. The most popular course is Java with more than 300.000 subscribers. Courses with the top 10 most subscribers can be seen')

	# 	image = Image.open('Udemy3.png')
	# 	st.image(image, caption='Number of reviews per course')
	# 	st.markdown('Number of reviews per course: most courses have very few reviews. The limit of the available reviews from the API is 10.000, hence there are couple courses around 10.000 ')

	# 	image = Image.open('Udemy4.png')
	# 	st.image(image, caption='Number of reviews per user')
	# 	st.markdown('The first 20 values of the most common number of reviews per user most users were plotted. More than 600000 users have only 1 review')

	# 	image = Image.open('Udemy5.png')
	# 	st.image(image, caption='Usernames with the most reviews')
	# 	st.markdown('Plotted the usernames with the most reviews - Username is not unique')
	
	elif choice == "Recommend by CourseID":
		st.subheader("Recommend Courses by CourseID")
		
		search_course_id = st.number_input("Search by course id", min_value=0)
		n_courses = st.number_input("Number (upto 50)",max_value=50)
		if st.button("Recommend"):
			if search_course_id is not None:
				try:
					#title, y_true, result =recommend_courses(3051582, 10, df_courses, df_norm)
					#title, y_true, result =recommend_courses(2280568, 10, df_courses, df_norm)
					#title, y_true, result =recommend_courses(1438222, 10, df_courses, df_norm)
					# title, y_true, result =recommend_courses(566284, 10, df_courses, df_norm)
					# title, y_true, result =recommend_courses(996228, 10, df_courses, df_norm)
					#title, y_true, result =recommend_courses(354176, 10, df_courses, df_norm)
					#title, y_true, result =recommend_courses(526104, 10, df_courses, df_norm) #100%
					# title, y_true, result =recommend_courses(1906852, 10, df_courses, df_norm) #70%
					# title, y_true, result =recommend_courses(1259404, 10, df_courses, df_norm) #30% but the courses are relevant to the 
					#title, y_pred, y_true, result = recommend_courses(2234122, 10, df_courses, df_norm) # 90%
					title, y_pred, y_true, result = recommend_courses(search_course_id, n_courses, df_courses, df_norm)
					st.write('Course: ', str(search_course_id), '-', title.values[0], '(', y_true[0], ')')
					cols = initialize_course_widget(len(result))
					ids = result['id']
					categorys = result['primary_subcategory']
					price = result['price']
					titles, poster, url, inst, ratings = fetch_course_details(result, df)
					#counter = 0
					for c, d, t, cat, p, price, u, i, r in zip(cols, ids, titles, categorys, poster, price, url, inst, ratings):
						with c:
							st.image(p)
							st.markdown(f"<a style='display: block; text-align: center;' href={u}>{t}</a>", unsafe_allow_html=True)
							st.write("Course ID: ", d)
							st.write("Category: ", cat)
							# st.write(price)
							st.write(i)
							st.write("Rating: ", r)
					
				except:
					result = "Not Found"
					st.warning(result)
					st.info("Suggested Options include")
					#result_df = search_term_if_not_found(search_course_id,df)
					#st.dataframe(result_df)

			
	elif choice == "Recommend by Title":
		st.subheader("Recommend Courses by Title")

		options = st.multiselect('Search for a course title (select any one)', df["title"].unique())
		#options = st.selectbox('Search for a course title', df["title"].unique())
		n_courses = st.number_input("Number (upto 50)", min_value=1, max_value=50)
		show_courses_recomm_btn = st.button("Recommend")

		if show_courses_recomm_btn:
			selected_title_id = df[df['title'] == options[0]]['id'].values[0]
			print(selected_title_id)
			title, y_pred, y_true, result = recommend_courses(selected_title_id, n_courses, df_courses, df_norm)
			st.write('Course: ', selected_title_id, '-', title.values[0], '(', y_true[0], ')')
			cols = initialize_course_widget(len(result))
			ids = result['id']
			categorys = result['primary_subcategory']
			price = result['price']
			titles, poster, url, inst, ratings = fetch_course_details(result, df)
			#counter = 0
			for c, d, t, cat, p, price, u, i, r in zip(cols, ids, titles, categorys, poster, price, url, inst, ratings):
				with c:
					st.image(p)
					st.markdown(f"<a style='display: block; text-align: center;' href={u}>{t}</a>", unsafe_allow_html=True)
					st.write("Course ID: ", d)
					st.write("Category: ", cat)
					# st.write(price)
					st.write(i)
					st.write("Rating: ", r)
					

	else:
		st.subheader("About")
		st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
	main()