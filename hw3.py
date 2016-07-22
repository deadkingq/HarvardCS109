import json

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

# set some nicer defaults for matplotlib
from matplotlib import rcParams

#these colors come from colorbrewer2.org. Each is an RGB triplet
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecessary plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


api_key = ''
movie_id = '770672122'  # toy story 3
url = 'http://api.rottentomatoes.com/api/public/v1.0/movies/%s/reviews.json' % movie_id

#these are "get parameters"
options = {'review_type': 'top_critic', 'page_limit': 20, 'page': 1, 'apikey': api_key}
data = requests.get(url, params=options).text
data = json.loads(data)  # load a json string into a collection of lists and dicts


from io import StringIO
movie_txt = requests.get('https://raw.github.com/cs109/cs109_data/master/movies.dat').text
movie_file = StringIO(movie_txt) # treat a string like a file
movies = pd.read_csv(movie_file, delimiter='\t')

def rt_id_by_imdb(imdb):
    """
    Queries the RT movie_alias API. Returns the RT id associated with an IMDB ID,
    or raises a KeyError if no match was found
    """
    url = 'http://api.rottentomatoes.com/api/public/v1.0/'+ 'movie_alias.json'

    imdb = "%7.7i" % imdb
    params = dict(id=imdb, type='imdb', apikey=api_key)

    r = requests.get(url, params=params).text
    r = json.loads(r)

    return r['id']

def _imdb_review(imdb):
    """
    Query the RT reviews API, to return the first page of reviews
    for a movie specified by its IMDB ID

    Returns a list of dicts
    """
    rtid = rt_id_by_imdb(imdb)
    url = 'http://api.rottentomatoes.com/api/public/v1.0/' + 'movies/{0}/reviews.json'.format(rtid)

    params = dict(review_type='top_critic',
                  page_limit=20,
                  page=1,
                  country='us',
                  apikey=api_key)
    data = json.loads(requests.get(url, params=params).text)
    data = data['reviews']
    data = [dict(fresh=r['freshness'],
                 quote=r['quote'],
                 critic=r['critic'],
                 publication=r['publication'],
                 review_date=r['date'],
                 imdb=imdb, rtid=rtid
                 ) for r in data]
    return data


def fetch_reviews(movies, row):
        m = movies.irow(row)
    try:
        result = pd.DataFrame(_imdb_review(m['imdbID']))
        result['title'] = m['title']
        return result
    except KeyError:
        return None



def build_table(movies, rows):
    dfs = [fetch_reviews(movies, r) for r in range(rows)]
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True)

critics = build_table(movies, 3000)
critics.to_csv('critics.csv', index=False)
critics = critics[~critics.quote.isnull()]

n_reviews = len(critics)
n_movies = critics.rtid.unique().size
n_critics = critics.critic.unique().size

print "Number of reviews: %i" % n_reviews
print "Number of critics: %i" % n_critics
print "Number of movies:  %i" % n_movies

critics.groupby('critic').rtid.count().hist(log=True, bins=range(20), edgecolor='white')
plt.xlabel("Number of reviews per critic")
plt.ylabel("N")

counts = critics.groupby(['critic', 'publication']).critic.count()
counts.sort()
counts[-1:-6:-1]

df = critics.copy()
df['fresh'] = df.fresh == 'fresh'
grp = df.groupby('critic')
counts = grp.critic.count()  # number of reviews by each critic
means = grp.fresh.mean()     # average freshness for each critic

means[counts > 100].hist(bins=10, edgecolor='w', lw=1)
plt.xlabel("Average rating per critic")
plt.ylabel("N")
plt.yticks([0, 2, 4, 6, 8, 10])

data = movies[['year', 'rtTopCriticsRating']]
data = data.convert_objects(convert_numeric=True)
data = data[(data['rtTopCriticsRating'] > 0)]
means = data.groupby('year').mean().dropna()

plt.plot(data['year'], data['rtTopCriticsRating'], 'o', mec='none', alpha=.2, label='Data')
plt.plot(means.index, means['rtTopCriticsRating'], '-', label='Yearly Average')
plt.legend(loc='lower left', frameon=False)
plt.xlabel("Year")
plt.ylabel("Average Score")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

def make_xy(critics, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(critics.quote)
    X = X.tocsc()  # some versions of sklearn return COO format
    Y = (critics.fresh == 'fresh').values.astype(np.int)
    return X, Y

X, Y = make_xy(critics)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
clf = MultinomialNB().fit(xtrain, ytrain)

print("Accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest)))
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

print("Accuracy on training data: %0.2f" % (training_accuracy))
print("Accuracy on test data:     %0.2f" % (test_accuracy))

def calibration_plot(clf, xtest, ytest):
    prob = clf.predict_proba(xtest)[:, 1]
    outcome = ytest
    data = pd.DataFrame(dict(prob=prob, outcome=outcome))

    #group outcomes into bins of similar probability
    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)
    binwidth = bins[1] - bins[0]

    #freshness ratio and number of examples in each bin
    cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
    cal['pmid'] = (bins[:-1] + bins[1:]) / 2
    cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])

    #the calibration plot
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
    plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
    plt.ylabel("Empirical P(Fresh)")
    remove_border(ax)

    #the distribution of P(fresh)
    ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)

    plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'],
            width=.95 * (bins[1] - bins[0]),
            fc=p[0].get_color())

    plt.xlabel("Predicted P(Fresh)")
    remove_border()
    plt.ylabel("Number")

calibration_plot(clf, xtest, ytest)

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()

from sklearn.cross_validation import KFold

def cv_score(clf, x, y, score_func):
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(clf, x[test], y[test]) # evaluate score function on held-out data
    return result / nfold # average

#the grid of parameters to search over
alphas = [0, .1, 1, 5, 10, 50]
min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
max_loglike = -np.inf

for alpha in alphas:
    for min_df in min_dfs:
        vectorizer = CountVectorizer(min_df = min_df)
        X, Y = make_xy(critics, vectorizer)

        #your code here
        clf = MultinomialNB(alpha=alpha)
        loglike = cv_score(clf, X, Y, log_likelihood)

        if loglike > max_loglike:
            max_loglike = loglike
            best_alpha, best_min_df = alpha, min_df

vectorizer = CountVectorizer(min_df=best_min_df)
X, Y = make_xy(critics, vectorizer)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

calibration_plot(clf, xtest, ytest)

# Your code here. Print the accuracy on the test and training dataset
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

words = np.array(vectorizer.get_feature_names())

x = np.eye(xtest.shape[1])
probs = clf.predict_log_proba(x)[:, 0]
ind = np.argsort(probs)

good_words = words[ind[:10]]
bad_words = words[ind[-10:]]

good_prob = probs[ind[:10]]
bad_prob = probs[ind[-10:]]

print "Good words\t     P(fresh | word)"
for w, p in zip(good_words, good_prob):
    print "%20s" % w, "%0.2f" % (1 - np.exp(p))

print "Bad words\t     P(fresh | word)"
for w, p in zip(bad_words, bad_prob):
    print "%20s" % w, "%0.2f" % (1 - np.exp(p))

x, y = make_xy(critics, vectorizer)

prob = clf.predict_proba(x)[:, 0]
predict = clf.predict(x)

bad_rotten = np.argsort(prob[y == 0])[:5]
bad_fresh = np.argsort(prob[y == 1])[-5:]

print "Mis-predicted Rotten quotes"
print '---------------------------'
for row in bad_rotten:
    print critics[y == 0].quote.irow(row)

print "Mis-predicted Fresh quotes"
print '--------------------------'
for row in bad_fresh:
    print critics[y == 1].quote.irow(row)

