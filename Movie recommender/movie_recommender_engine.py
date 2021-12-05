import pandas as pd


##Step 1: Read CSV File

df = pd.read_csv('movie_dataset.csv')
print (df.columns)

##Step 2: Select Features

selected_features = ['keywords', 'cast', 'genres', 'director']

for f in selected_features:
  df[f] = df[f].fillna("")

##Step 3: Create a column in DF which combines all selected features

def combine_features(row):
  try:
    return row['keywords'] + row['cast'] +  row['genres'] +  row['director']
  except:
    print("Error:", row)

df['combined_to_use'] = df.apply(combine_features, axis = 1)
#df
print ("Combiner Features", df['combined_to_use'].head())

##Step 4: Create count matrix from this new combined column

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_to_use']) 


##Step 5: Compute the Cosine Similarity based on the count_matrix

cosine_sim = cosine_similarity(count_matrix) 
movie_that_user_likes = 'Troy'

## Step 6: Get index of this movie from its title

###### Helper functions #######
def get_title_from_index(index):
   return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
   return df[df.title == title]["index"].values[0]

movie_index = get_index_from_title(movie_that_user_likes) 

similar_movies = list(enumerate(cosine_sim[movie_index]))


## Step 7: Get a list of similar movies in descending order of similarity score

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1], reverse=True)
#sorted_similar_movies

## Step 8: Print titles of first 50 movies

i = 0
for element in sorted_similar_movies:
  print(get_title_from_index(element[0]))
  i = i + 1
  if i>50:
    break