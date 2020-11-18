# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
# %%
genres_encoded = pd.read_csv("data/posters-and-genres.csv").drop(columns=['Id', 'Genre'])
genres = genres_encoded.columns.tolist()
genres_encoded.shape


# %%
genres_encoded = genres_encoded.values
genres_encoded


# %%
genres_count = np.sum(genres_encoded, axis=0)
genres_count


# %%
plt.style.use('ggplot')


# %%
data = {'Genre': genres, 'Count': genres_count}
df_genre_count = pd.DataFrame(data).set_index('Genre').sort_values('Count', ascending=False)


# %%
fig, ax = plt.subplots(figsize=(10, 8))
df_genre_count.plot(kind='barh', legend=False, ax=ax)
ax.set_xlabel("Count")
ax.set_ylabel("Genre")
ax.set_title("Distribution of Genres in Dataset")
# plt.savefig("distribution-of-genres.png")


# %%
df_genre_percent = df_genre_count.copy()
df_genre_percent['Count'] = 100*df_genre_percent['Count'] / len(genres_encoded)
df_genre_percent.rename({'Count': 'Percentage'}, inplace=True)


# %%
fig, ax = plt.subplots(figsize=(10, 8))
df_genre_percent.plot(kind='barh', color='green', legend=False, ax=ax)
ax.set_xlabel("Percentage (%)")
ax.set_ylabel("Genre")
ax.set_title("Distribution of Genres in Dataset")
# plt.savefig("percentage-of-genres.png")

# %% [markdown]
# So our data is heavily favorite toward Dramas and Comedies
# %% [markdown]
# ## Clean it up

# %%
movies_metadata = pd.read_csv("data/movies-metadata.csv", thousands=",").drop(columns=['Unnamed: 0', '_id'])
movies_metadata[movies_metadata['imdbID'] == 'tt2070791'] # bad data sample because of a missing


# %%
def create_year(series_object):
    # print(series_object)
    if isinstance(series_object['Released'], str):
        year = int(series_object['Released'].split()[-1])
    else:
        year = 0
    return year


# %%
movies_metadata['release_year'] = movies_metadata.apply(create_year, axis=1)


# %%
movie_years = movies_metadata.drop(movies_metadata[movies_metadata['release_year'] == 0].index)
min_year = movie_years['release_year'].min()
max_year = movie_years['release_year'].max()
print(min_year, max_year)


# %%
count_by_year = np.zeros(max_year-min_year+1, dtype='int32')
# print("YEAR| COUNT MOVIES")
for year in range(min_year, max_year+1):
    count_by_year[year-2010] = movie_years.loc[movie_years['release_year'] == year, 'release_year'].count()
    # print(f"{year}:", count_by_year[year-2010])


# %%
import matplotlib.pyplot as plt
years = np.arange(min_year, max_year+1)
labels = np.arange(len(years))
plt.figure(figsize=(10, 8))
plt.barh(labels, count_by_year, align='center')
plt.yticks(labels, years)
plt.ylabel("Year Released")
plt.xlabel("Number of films")
plt.title("Number of films from each year")
# plt.savefig("distribution-by-year.png")


# %%
genres = pd.read_csv("data/posters-and-genres.csv").rename(columns={'Id': 'imdbID'})
movies_metadata = movies_metadata.merge(genres, on='imdbID')


# %%
duplicates = movies_metadata['imdbID'].duplicated(keep=False)
duplicates.sum()


# %%
movies_metadata = movies_metadata.drop_duplicates()
movies_metadata.shape


# %%
missing_years = movies_metadata.loc[movies_metadata['release_year'] == 0][['Language', 'Box_office', 'Country', 'Rated', 'imdbID', 'Awards',
       'Poster', 'Director', 'Released', 'Writer', 'imdbVotes', 'Runtime', 'Type',
       'Response', 'imdbRating', 'Title', 'Genre_x']]


# %%
from PIL import Image

def plot_posters(movie_list):
    img_dir = "data/Images/"
    fig = plt.figure(figsize=(16, 12))
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
    columns = 8
    rows = len(movie_list) // columns + 1
    # fig, ax = plt.subplots(rows, columns, figsize=(40, 12))
    for i in range(len(movie_list)):
        img = Image.open(os.path.join(img_dir, movie_list[i][0] + ".jpg")).resize((300, 450))
        # print(i, i // columns, i%columns, rows)
        ax = fig.add_subplot(rows, columns, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(movie_list[i][1], fontsize=8)
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# %%
movie_list = missing_years[['imdbID', 'Title']].values
plot_posters(movie_list)


# %%
missing_years[['imdbID', 'Director', 'Title', 'Type']]

# %% [markdown]
# We need to do a few things: eliminate "series", fix unmatching movie posters if they exist, and deal with missing values
# %% [markdown]
# ### Eliminate Series

# %%
movies_metadata.loc[movies_metadata['Type'] == 'series']

# %% [markdown]
# So we have 14 series in this dataset. Series are not movies, so they will be eliminated.

# %%
print(movies_metadata.shape)
movies_metadata_cleaned = movies_metadata.drop(movies_metadata.loc[movies_metadata['Type'] == 'series'].index)
print(movies_metadata_cleaned.shape)

# %% [markdown]
# Looks good. We dropped 14 values.
# %% [markdown]
# ### Eliminate Bad Posters
# %% [markdown]
# Now after some additional evaluation, the poster that appeared in previous analysis was removed because it was a series. Therfore, we can make the assumption that the movie posters are likely to be the accurate poster, considering the data processing step outlined by the team that created this dataset (scraping IMDb data). If the posters are wrong, then they are wrong on IMDb, which we cannot do much to change confidently.
# %% [markdown]
# ### Missing Values

# %%
movies_metadata_cleaned.isnull().sum()

# %% [markdown]
# So there are a few issues here: missing categorical values do not make much sense to replace with another category, except potentially rated, language, and country. But at the same time, rating could mean the movie is in fact, unrated. Other "non-categorical" categorical variables are director, writer, actors, awards, etc., meaning these values are not derived from any fixed set. That means we can kind of ignore them.
# 
# However, since we are missing 1785 values for box_office, which will be out `y`, there could be difficulty there with estimating the value. Mode is a value that doesn't really make sense, but mean and median are not off the table. However, because there is so much data, we shoudl look at those values that are missing.

# %%
movies_metadata_cleaned.loc[movies_metadata_cleaned['Box_office'].isnull()]

# %% [markdown]
# First, let's start with the easier task of dealing with numerical data missing values.
# %% [markdown]
# For now, I will jsut drop movies without a budget.

# %%
movies_metadata_cleaned.dropna(subset=['Box_office'], inplace=True)

# %% [markdown]
# ## Categeroize Budgets

# %%
print(movies_metadata_cleaned['Box_office'].mean())
print(movies_metadata_cleaned['Box_office'].median())

# %% [markdown]
# This difference implies there is substantial differences between the greatest values and the rest of the data. Let's plot the box_office

# %%
plt.figure()
plt.scatter(np.arange(len(movies_metadata_cleaned['Box_office'].values)), movies_metadata_cleaned['Box_office'].values)
plt.xticks([])
plt.ylabel("Box Office (in 100 Million Dollars)")

# %% [markdown]
# This graph doesn't really tell us much, so lets plot a probability density.

# %%
box_office_values = np.sort(movies_metadata_cleaned['Box_office'].values)
range_data = (box_office_values[-1] - box_office_values[0])
print(range_data)
ax = plt.figure(figsize=(10, 7))
for i, binwidth in enumerate([100000, 1000000]): # this will take some time to run...
    ax = plt.subplot(2, 2, i + 1)
    ax.hist(box_office_values, color='red', edgecolor='black', bins=int(range_data/binwidth))
    ax.set_title(f"Histogram with binwidth={binwidth}")
    ax.set_xlabel("Box Office")
    ax.set_ylabel("Number of Movies within that range")
plt.tight_layout()
plt.show()

# %% [markdown]
# Yeah so there is quite a bit of data in the low range of box_office... Let's see what the some percentiles are...

# %%
for percentile in np.arange(10, 100, 10):
    print(percentile, f"| {np.percentile(box_office_values, percentile):.2f}")
# print(twentieth_percentile)

# %% [markdown]
# So, as expected, the mean of 24880168.703288626 does NOT reflect the data very well, considering it only really represents the top 20-30% of the data. 
# %% [markdown]
# The scales from the paper were as follows:
# 
#     class 1: profit less than or equal to 0.5 million
# 
#     class 2: profit between 0.5M and 1M
# 
#     class 3: profit between 1M and 40M
# 
#     class 4: profit between 40M and 150M
#     
#     class 5: profit greater than 150M
# 
# Now we currently do not have budget information, so we cannot calculate profit right now. But we can categorize based on box office.
# %% [markdown]
# So we first need to know how many to put into each group. If we want equally distributed data, we can just put 20% into each category, and let the thresholds define themselves, or we could preedefine the ranges. For now, we will evenly distribute the data.

# %%
class_thresholds = np.zeros(5, dtype='int32')
for split_point in np.arange(20, 100, 20):
    class_thresholds[split_point//20-1] = round(np.percentile(box_office_values, split_point))
    print(f"Class {split_point//20}: values <", f"{class_thresholds[split_point//20-1]}")

# %% [markdown]
# So our classes will be set up as follows:
#     
#     class 1: box office less than $214,749
# 
#     class 2: box office between $214,749 and $1,750,232
# 
#     class 3: box office between $1,750,232 and $9,592,706
#     
#     class 4: box office between $9,592,706 and $34,957,764
#     
#     class 5: box office greater than or equal to $34,957,764
#     

# %%
def create_category(series_object, class_thresholds):
    '''
    reflects the categories similar to those specified by paper
    '''
    box_office = series_object['Box_office']
    if box_office < class_thresholds[0]:
        return 1
    elif box_office < class_thresholds[1]:
        return 2
    elif box_office < class_thresholds[2]:
        return 3
    elif box_office < class_thresholds[3]:
        return 4
    else:
        return 5

# %% [markdown]
# Now let's create this new row in the metadata so we can better predict data.

# %%
movies_metadata_cleaned['box_office_class'] = movies_metadata_cleaned.apply(lambda x: create_category(x, class_thresholds), axis=1)
movies_metadata_cleaned.head()

# %% [markdown]
# ## Curious about how movies panned based on their release month? I am.
# %% [markdown]
# Let's see what movies are missing a release date.

# %%
movies_metadata_cleaned.loc[movies_metadata_cleaned['Released'].isnull()]

# %% [markdown]
# We will get back to that, but let's make a column for release month first, then assign to those missing values the maximum across the missing values.

# %%
month_map = {
    'Jan': 0,
    'Feb': 1,
    'Mar': 2,
    'Apr': 3,
    'May': 4,
    'Jun': 5,
    'Jul': 6,
    'Aug': 7,
    'Sep': 8,
    'Oct': 9,
    'Nov': 10,
    'Dec': 11,
}
def create_month(series_object, month_map=month_map):
    if isinstance(series_object['Released'], str):
        month = month_map[series_object['Released'].split()[1]]
    else:
        month = np.NaN
    return month


# %%
movies_metadata_cleaned['release_month'] = movies_metadata_cleaned.apply(create_month, axis=1)


# %%
print(movies_metadata_cleaned.loc[movies_metadata_cleaned['release_month'].isnull()].shape)

# %% [markdown]
# Now we can fill these values with the mode in the released_month column, which is:

# %%
movies_metadata_cleaned['release_month'].mode()[0]


# %%
movies_metadata_cleaned['release_month'].fillna(movies_metadata_cleaned['release_month'].mode()[0], inplace=True)
print(movies_metadata_cleaned.loc[movies_metadata_cleaned['release_month'].isnull()].shape)

# %% [markdown]
# A shape of 0 rows means we did it! Now lets see the distributtion for each class per month

# %%
movies_metadata_cleaned = movies_metadata_cleaned.astype({'release_month': 'int32'}).sort_values(['release_month'])


# %%
months = movies_metadata_cleaned['release_month'].value_counts().sort_index().index.tolist()
months


# %%
class1_month_counts = movies_metadata_cleaned.loc[movies_metadata_cleaned['box_office_class'] == 1]['release_month'].value_counts().sort_index().values
class2_month_counts = movies_metadata_cleaned.loc[movies_metadata_cleaned['box_office_class'] == 2]['release_month'].value_counts().sort_index().values
class3_month_counts = movies_metadata_cleaned.loc[movies_metadata_cleaned['box_office_class'] == 3]['release_month'].value_counts().sort_index().values
class4_month_counts = movies_metadata_cleaned.loc[movies_metadata_cleaned['box_office_class'] == 4]['release_month'].value_counts().sort_index().values
class5_month_counts = movies_metadata_cleaned.loc[movies_metadata_cleaned['box_office_class'] == 5]['release_month'].value_counts().sort_index().values


# %%
months = np.array(months)
month_names = month_map.keys() # the label locations
width = 0.15  # the width of the bars
# print(month_names)
plt.figure(figsize=(12, 8))
rects1 = plt.bar(months - 2*width, class1_month_counts, width, label='Class 1', align='center')
rects1 = plt.bar(months - width, class2_month_counts, width, label='Class 2', align='center')
rects1 = plt.bar(months, class3_month_counts, width, label='Class 3', align='center')
rects1 = plt.bar(months + width, class4_month_counts, width, label='Class 4', align='center')
rects1 = plt.bar(months + 2*width, class5_month_counts, width, label='Class 5', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Number of Movies')
# ax.set_title('Number of movies from each category per month')
# ax.set_xlabel("Month")
# ax.set_xticks(x)
# ax.set_xticklabels(months)
# ax.legend()
plt.ylabel('Number of Movies')
plt.title('Number of movies from each category per month')
plt.xlabel("Month")
plt.xticks(months, labels=month_names)
# plt.xticklabels(months)
# plt.facecolor('white')
plt.legend(loc=(0.1, 0.8), shadow=True, facecolor='white')
plt.savefig("figures/month-distribution-by-category.png")


# %%



