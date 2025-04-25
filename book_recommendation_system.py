#%% md
# # Book Recommendation System
#%% md
# ## Import Library
#%%
!pip install kaggle pandas scikit-learn tensorflow matplotlib unzip
#%%
# Basic utility libraries
import os          # for interacting with the operating system (e.g., paths)
import shutil      # for file operations like copying
import zipfile     # for extracting zip files
import re          # for regex
from collections import defaultdict
from math import floor
from IPython.display import display, Image 
import random

# Data handling and analysis
import pandas as pd       # for working with tabular data (DataFrames)
import numpy as np        # for numerical operations and array handling

# Data visualization
import matplotlib.pyplot as plt  # for plotting graphs and charts
import seaborn as sns            # for statistical visualizations and styling

# Content-Based Filtering (CBF)  tools
from sklearn.feature_extraction.text import TfidfVectorizer  #TF-IDF 
from sklearn.metrics.pairwise import cosine_similarity # cosine similarity

# Deep Learning (for Collaborative Filtering)
from sklearn.preprocessing import LabelEncoder # Encode data
from sklearn.model_selection import train_test_split
import tensorflow as tf                      # core TensorFlow library
import keras                                 # high-level API for building models
from keras import layers                     # used to define neural network layers
from keras import ops                        # Provides mathematical operations for custom model building
#%% md
# ## Data Loading 
#%%
!kaggle datasets download -d arashnic/book-recommendation-dataset # Download dataset from Kaggle
#%%
!unzip book-recommendation-dataset.zip -d dataset #unzip dataset
#%%
books = pd.read_csv("dataset/Books.csv")
users = pd.read_csv("dataset/Users.csv")
ratings = pd.read_csv("dataset/Ratings.csv")

print(f"Books: {books.shape[0]} data | Unique ISBN: {books['ISBN'].nunique()}")
print(f"Users: {users.shape[0]} data | Unique User-ID: {users['User-ID'].nunique()}")
print(f"Ratings: {ratings.shape[0]} data | Unique User-ID: {ratings['User-ID'].nunique()}")
#%% md
# ## Data Understanding - Univariate Exploratory Data Analysis (EDA) 
#%% md
# ### Books Variable
#%%
books.info() # show structure and types
#%%
books.describe() 
#%%
# Number of unique values in key columns
print(f"\nUnique ISBNs: {books['ISBN'].nunique()}")
print(f"Unique Titles: {books['Book-Title'].nunique()}")
print(f"Unique Authors: {books['Book-Author'].nunique()}")
print(f"Unique Publishers: {books['Publisher'].nunique()}")
#%%
print("Missing values in Books:")
print(books.isnull().sum(), "\n") # Check missing value
#%%
# Top 15 Publishers
top_publishers = books['Publisher'].value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_publishers.values, y=top_publishers.index, hue=top_publishers.index, legend=False, palette="coolwarm")
plt.title("Top 15 Most Frequent Publishers")
plt.xlabel("Count")
plt.ylabel("Publisher")
plt.show()
#%%
# Top 15 Authors
top_authors = books['Book-Author'].value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_authors.values, y=top_authors.index, hue=top_authors.index, legend=False, palette="rocket")
plt.title("Top 15 Most Frequent Authors")
plt.xlabel("Count")
plt.ylabel("Author")
plt.show()
#%%
# Check outlier 
temp_books = books.copy()
temp_books['Year-Of-Publication'] = pd.to_numeric(temp_books['Year-Of-Publication'], errors='coerce')


plt.figure(figsize=(8, 4))
sns.boxplot(x=temp_books['Year-Of-Publication'].dropna() )
plt.title("Box Plot of Year of Publication")
plt.xlabel("Year")
plt.show()
#%% md
# ### Users Variable 
#%%
users.info() # Preview structure and types
#%%
users.describe() # Summary statistics
#%%
print(f"\nUnique Users: {users['User-ID'].nunique()}")
print(f"Unique Locations: {users['Location'].nunique()}")
#%%
print("Missing values in Users:")
print(users.isnull().sum(), "\n") # Check missing value
#%% md
# ####  User Age Group Distribution
# 
# To provide a clearer view of user demographics, the users are grouped users  four age categories:
# 
# - **Child**: 5–12 years old
# - **Teenager**: 13–20 years old
# - **Adult**: 21–59 years old
# - **Senior**: 60+ years old
#%%
# Filter users with a realistic age range (5 to 100 years)
filtered_users = users[(users['Age'] >= 5) & (users['Age'] <= 100)].copy()

# Define age group categories
def get_age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 20:
        return 'Teenager'
    elif age <= 59:
        return 'Adult'
    else:
        return 'Senior'

# Apply age group categorization
filtered_users['AgeGroup'] = filtered_users['Age'].apply(get_age_group)


plt.figure(figsize=(8, 5))
ax = sns.countplot(data=filtered_users, x='AgeGroup', order=['Child', 'Teenager', 'Adult', 'Senior'], palette='Set3', hue='AgeGroup', legend=False)

# Add data labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    
plt.title("Distribution of Users by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Users")
plt.show()
#%%
# Checking outliers on user based on age
plt.figure(figsize=(8, 4))
sns.boxplot(x=users['Age'])
plt.title("Box Plot of User Age")
plt.xlabel("Age")
plt.show()
#%% md
# # Ratings Variable
#%%
ratings.info() # Preview structure and types
#%%
ratings.describe() # show summary
#%%
ratings.head()
#%%
print(f"\nUnique User-ID: {ratings['User-ID'].nunique()}")
print(f"Unique ISBN: {ratings['ISBN'].nunique()}")
print(f"Most common rating: \n{ratings['Book-Rating'].value_counts().sort_index()}")
#%% md
# #### Distribution of Book Ratings per User 
#%%
plt.figure(figsize=(10, 5))
sns.countplot(x='Book-Rating', hue='Book-Rating', legend=False, data=ratings, palette='coolwarm')
plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

# Add exact number above bars
ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

plt.tight_layout()
plt.show()
#%%
# Print the distribution of ratings from 0 to 10
rating_distribution = ratings['Book-Rating'].value_counts().sort_index()

# Display the distribution
print("Distribution of Ratings (0-10):")
print(rating_distribution)
#%% md
# ## Data Preprocessing
# 
#%% md
# ### Books Variable Cleaning
#%% md
# #### Handling missing values
#%%
print("Missing values in books before handling:")
print(books.isnull().sum())
#%%
# Fill the missing values in Book-Author and Publisher with Unknown
books['Book-Author'] = books['Book-Author'].fillna('Unknown')
books['Publisher'] = books['Publisher'].fillna('Unknown')
#%%
# Drop missing values on Image-URL-L
books = books[books['Image-URL-L'].notnull()]
#%%
print("Missing values in books after handling:")
print(books.isnull().sum())
print(f"Final shape: {books.shape}")
#%% md
# ### 
#%% md
# #### Year Of Publication Outlier Handling
#%%
# Copy and convert year column to numeric
temp_books = books.copy()
temp_books['Year-Of-Publication'] = pd.to_numeric(temp_books['Year-Of-Publication'], errors='coerce')
#%%
# Drop NaN for analysis
year_data = temp_books['Year-Of-Publication'].dropna()
#%%
# Handling the Outlier with IQR
Q1 = year_data.quantile(0.25)
Q3 = year_data.quantile(0.75)
IQR = Q3 - Q1

# Define acceptable range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR: {IQR}")
print(f"Acceptable Year Range: {int(lower_bound)} to {int(upper_bound)}")
#%%
# Apply only years within the IQR range
books = temp_books[(temp_books['Year-Of-Publication'] >= lower_bound) & 
                   (temp_books['Year-Of-Publication'] <= upper_bound)]

print(f"Remaining data shape after outlier removal: {books.shape}")
#%%
plt.figure(figsize=(8, 4))
sns.boxplot(x=books['Year-Of-Publication'])
plt.title("Box Plot of Year of Publication")
plt.xlabel("Year")
plt.show()
#%% md
# ### Users Variable
#%% md
# #### Handling Outlier and Missing Values in Age
#%%
# Copy and filter the valid age ranges
users_cleaned = users.copy()
users_cleaned = users_cleaned[(users_cleaned['Age'] >= 5) & (users_cleaned['Age'] <= 100)]

#%%
# Fill remaining missing values with median age (from within the range)
median_age = users_cleaned['Age'].median()
users_cleaned['Age'].fillna(median_age)

print(f"Users cleaned shape: {users_cleaned.shape}")
#%%
# Checking outliers on user based on age
plt.figure(figsize=(8, 4))
sns.boxplot(x=users_cleaned['Age'])
plt.title("Box Plot of User Age")
plt.xlabel("Age")
plt.show()
#%% md
# ### Ratings Variable
#%% md
# ### Drop Rating 0
# Users have read the book, but they don't give the feedback. So, rating 0 is dropped
#%%
ratings_cleaned = ratings[ratings['Book-Rating'] > 0].copy()
print(f"Ratings after removing 0s: {ratings_cleaned.shape}")
#%%
plt.figure(figsize=(10, 5))
sns.countplot(x='Book-Rating', hue='Book-Rating', legend=False, data=ratings_cleaned, palette='Blues')
plt.title("Distribution of Explicit Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")

# Add labels on bars
ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

plt.tight_layout()
plt.show()
#%% md
# ## Data Preparation
# All variable data is merged to be used in both Collaborative Filtering (CF) and Content-Based Filtering (CBF). It can be flexibly used for both models.
#%%
# Merger Ratings and Books
ratings_books = ratings_cleaned.merge(books, on='ISBN', how='inner')
print(f"Merged (ratings + books): {ratings_books.shape}")
#%%
# Merger with users
full_data = ratings_books.merge(users_cleaned, on='User-ID', how='inner')
print(f"Final merged shape (ratings + books + users): {full_data.shape}")
#%%
full_data.head()
#%% md
# ## Modelling
#%% md
# ### Model Development with Content Based Filtering
#%%
# Create a copy of full_data for CBF modeling and using used columns
cbf_data = full_data[['Book-Title', 'Book-Author', 'Publisher', 'Image-URL-L']].drop_duplicates()
cbf_data = cbf_data.drop_duplicates(subset=['Book-Title', 'Book-Author'], keep='first').reset_index(drop=True)
#%%
#Combine book title and author for TF-IDF
cbf_data['combined_features'] = cbf_data['Book-Title'] + ' ' + cbf_data['Book-Author']
cbf_data['combined_features'] = cbf_data['combined_features'].fillna('')
#%%
# Using TF-IDF
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(cbf_data['combined_features'])

vocab = tfidf.get_feature_names_out()

print(f"Unique features (token): {len(vocab)}")
print(vocab[:20])
#%%
tfidf_matrix.shape
#%%
# Create lowercase index mapping
cbf_data['Book-Title-Lower'] = cbf_data['Book-Title'].str.lower()
indices = pd.Series(cbf_data.index, index=cbf_data['Book-Title-Lower']).drop_duplicates()
#%%
def normalize_title(title):
    title = title.lower()
    title = re.sub(r'[^a-z\s]', '', title)  # remove digits/punctuation
    title = re.sub(r'\s+', ' ', title).strip()
    return title
#%%
def safe_display_image(url, width=100):
    if isinstance(url, str) and url.startswith('http'):
        try:
            display(Image(url=url, width=width))
        except:
            print("[Image not available]")
    else:
        print("[No image URL]")
#%%
def get_book_recommendations(title, k=5, max_author_ratio=0.3):
    """
    Return top-k diverse books based on TF-IDF cosine similarity.
    Limits max percentage of books from the same author.
    Excludes semantic duplicates based on normalized title + author.
    """
    title = title.lower()

    if title not in indices:
        print(f"Book title '{title}' not found in the dataset.")
        return

    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    if idx >= tfidf_matrix.shape[0]:
        print(f"Index {idx} is out of range for the TF-IDF matrix.")
        return

    # Normalize input title and author
    input_title_norm = normalize_title(cbf_data.loc[idx, 'Book-Title'])
    input_author = cbf_data.loc[idx, 'Book-Author'].lower()

    # Compute cosine similarity
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1]

    # Author diversity filtering
    max_per_author = max(1, floor(k * max_author_ratio))
    author_counts = defaultdict(int)
    filtered = []

    # Filtering book to not same with the input
    for i in top_indices:
        if i == idx:
            continue

        candidate = cbf_data.iloc[i]
        candidate_title_norm = normalize_title(candidate['Book-Title'])
        candidate_author = candidate['Book-Author'].lower()

        # Skip duplicates
        if candidate_title_norm == input_title_norm and candidate_author == input_author:
            continue

        # Author diversity filter
        if author_counts[candidate_author] >= max_per_author:
            continue

        filtered.append(i)
        author_counts[candidate_author] += 1

        if len(filtered) == k:
            break

    if not filtered:
        print("No diverse recommendations found.")
        return

    # Display input book
    print("\nInput Book:")
    input_book = cbf_data.loc[idx]
    print(f"Title     : {input_book['Book-Title']}")
    print(f"Author    : {input_book['Book-Author']}")
    print(f"Publisher : {input_book['Publisher']}")
    safe_display_image(input_book['Image-URL-L'])

    print(f"\nTop {k} Recommendations:\n")
    
    for i, index in enumerate(filtered, start=1):
        rec = cbf_data.iloc[index]
        print(f"#{i}")
        print(f"Title     : {rec['Book-Title']}")
        print(f"Author    : {rec['Book-Author']}")
        print(f"Publisher : {rec['Publisher']}")
        safe_display_image(rec['Image-URL-L'])
        print("-" * 40)
#%%
get_book_recommendations("Harry Potter and the Prisoner of Azkaban")
#%% md
# ### Model Development with Collaborative Filtering (CF)
#%% md
# #### Data Understanding
#%%
# Get data for cf_data 
cf_data = full_data[['User-ID', 'ISBN', 'Book-Rating']].copy()
#%%
cf_data.info()
#%% md
# #### Data Preparation 
# In data preparation, there are two steps:
# 1. Encode User-ID and ISBN to Integer 
# 2. Rename column with easy naming
# 3. Convert Book-Rating to float
#%%
# Encode User-ID and ISBN to Integer
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

cf_data['user'] = user_encoder.fit_transform(cf_data['User-ID'])
cf_data['item'] = item_encoder.fit_transform(cf_data['ISBN'])
#%%
# Cek result encode
cf_data.head()
#%%
# Convert rating to float
cf_data['rating'] = cf_data['Book-Rating'].astype(float)
#%%
# Get only used columns
cf_data = cf_data[['user', 'item', 'rating']]
cf_data.head()
#%%
# Count the number of unique users
num_users = cf_data['user'].nunique()

# Count the number of unique items (books)
num_items = cf_data['item'].nunique()

# Find the minimum and maximum rating values
min_rating = cf_data['rating'].min()
max_rating = cf_data['rating'].max()

# Display the results
print(f"Number of Users       : {num_users}")
print(f"Number of Items       : {num_items}")
print(f"Minimum Rating Value  : {min_rating}")
print(f"Maximum Rating Value  : {max_rating}")
#%% md
# #### Split Train and Testing Data
# Split 80% Training, 20% Testing
#%%
# Create the input array X (pairs of user and item)
X = cf_data[['user', 'item']].values

# Normalize ratings to range [0, 1]
y = cf_data['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
#%%
# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,           # Input: user-item pairs
    y,           # Target: normalized ratings
    test_size=0.2,
    random_state=42
)
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")
#%% md
# #### Training
#%%
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        
        # User embedding
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)

        # Item embedding
        self.item_embedding = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.item_bias = layers.Embedding(input_dim=num_items, output_dim=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0]) # Call layer embedding 1
        user_bias = self.user_bias(inputs[:, 0])        # Call layer embedding 2
        item_vector = self.item_embedding(inputs[:, 1]) # Call layer embedding 3
        item_bias = self.item_bias(inputs[:, 1])        # Call layer embedding 4

        dot_user_item = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=True)
        result = dot_user_item + user_bias + item_bias

        return tf.nn.sigmoid(result)
#%%
# Get user and item dimensions
num_users = cf_data['user'].nunique()
num_items = cf_data['item'].nunique()

# Build model
model = RecommenderNet(num_users, num_items)
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError()
    ]
)
#%%
# Training
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=10,
    verbose=1,
    validation_data=(X_test, y_test)
)
#%% md
# #### Visualize metrics
#%%
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Loss (MSE)
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # RMSE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
    plt.title('RMSE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # MAE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
    plt.title('MAE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
#%%
plot_training_history(history)
#%% md
# #### Get Books Recommendation
#%%
def show_high_rated_book_and_recommendation(original_user_id, top_k=5, min_rating=8.0, top_k_rating=3):
    # Encode user ID
    try:
        encoded_user_id = user_encoder.transform([original_user_id])[0]
    except ValueError:
        print(f"User-ID {original_user_id} not found in training data.")
        return pd.DataFrame()

    # Filter books rated ≥ min_rating by the user
    high_rated_df = cf_data[(cf_data['user'] == encoded_user_id) & (cf_data['rating'] >= min_rating)]
    
    # Sort by actual rating value (descending) and get top k__rating
    top_rated_df = high_rated_df.sort_values(by='rating', ascending=False).head(top_k_rating)
    
    # Decode item IDs to ISBNs
    high_rated_isbns = item_encoder.inverse_transform(top_rated_df['item'])
    
    # Lookup book details
    high_rated_books = books[books['ISBN'].isin(high_rated_isbns)][['Book-Title', 'Book-Author', 'Publisher', 'Image-URL-L']].drop_duplicates().reset_index(drop=True)
    
    print(f"\nTop {top_k_rating} Books rated ≥ {min_rating} by user {original_user_id}:\n")
    if high_rated_books.empty:
        print("No high-rated books found.")
    else:
        for i, row in high_rated_books.iterrows():
            print(f"#{i+1}: {row['Book-Title']} by {row['Book-Author']} ({row['Publisher']})")
            safe_display_image(row['Image-URL-L'])
            print("-" * 40)

    # Not rated Books → Recommendation
    user_all_rated_df = cf_data[cf_data['user'] == encoded_user_id]
    rated_items = user_all_rated_df['item'].values
    
    all_items = np.arange(num_items) # Create array
    unrated_items = np.setdiff1d(all_items, rated_items) # Get only unrated items from all items

    user_input = np.array([[encoded_user_id, item] for item in unrated_items]) # Create user - item to every unrated items
    
    #Predict using the model
    predicted_ratings = model.predict(user_input, verbose=0).flatten()

    top_indices = predicted_ratings.argsort()[-top_k:][::-1] # Get high prediction 
    
    recommended_item_ids = unrated_items[top_indices]
    recommended_isbns = item_encoder.inverse_transform(recommended_item_ids)
    
    # Get all recommendation book from books
    recommended_books = books[books['ISBN'].isin(recommended_isbns)][['Book-Title', 'Book-Author', 'Publisher', 'Image-URL-M']].drop_duplicates().reset_index(drop=True)

    print(f"\nTop {top_k} Book Recommendations for User {original_user_id}:\n")
    if recommended_books.empty:
        print("No recommendations available.")
    else:
        for i, row in recommended_books.iterrows():
            print(f"#{i+1}: {row['Book-Title']} by {row['Book-Author']} ({row['Publisher']})")
            safe_display_image(row['Image-URL-M'], width=100)
            print("-" * 40)
#%%
# Random users
unique_users = np.unique(X_train[:, 0])
original_user_ids = user_encoder.inverse_transform(unique_users)
random_user_id = random.choice(original_user_ids)
print(f"Get high rated books and recommendation for user ID: {random_user_id}")
show_high_rated_book_and_recommendation(random_user_id)
#%%
show_high_rated_book_and_recommendation(110887)