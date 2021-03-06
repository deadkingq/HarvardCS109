from collections import defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

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

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

#this mapping between states and abbreviations will come in handy later
states_abbrev = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


#adapted from  https://github.com/dataiap/dataiap/blob/master/resources/util/map_util.py

#load in state geometry
state2poly = defaultdict(list)

with open('data/us-states.json') as data_file:
    data = json.load(data_file)
for f in data['features']:
    state = states_abbrev[f['id']]
    geo = f['geometry']
    if geo['type'] == 'Polygon':
        for coords in geo['coordinates']:
            state2poly[state].append(coords)
    elif geo['type'] == 'MultiPolygon':
        for polygon in geo['coordinates']:
            state2poly[state].extend(polygon)


def draw_state(plot, stateid, **kwargs):
    """
    draw_state(plot, stateid, color=..., **kwargs)

    Automatically draws a filled shape representing the state in
    subplot.
    The color keyword argument specifies the fill color.  It accepts keyword
    arguments that plot() accepts
    """
    for polygon in state2poly[stateid]:
        xs, ys = zip(*polygon)
        plot.fill(xs, ys, **kwargs)


def make_map(states, label):
    """
    Draw a cloropleth map, that maps data onto the United States

    Inputs
    -------
    states : Column of a DataFrame
        The value for each state, to display on a map
    label : str
        Label of the color bar

    Returns
    --------
    The map
    """
    fig = plt.figure(figsize=(12, 9))
    ax = plt.gca()

    if states.max() < 2: # colormap for election probabilities
        cmap = cm.RdBu
        vmin, vmax = 0, 1
    else:  # colormap for electoral votes
        cmap = cm.binary
        vmin, vmax = 0, states.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    skip = set(['National', 'District of Columbia', 'Guam', 'Puerto Rico',
                'Virgin Islands', 'American Samoa', 'Northern Mariana Islands'])
    for state in states_abbrev.values():
        if state in skip:
            continue
        color = cmap(norm(states.ix[state]))
        draw_state(ax, state, color = color, ec='k')

    #add an inset colorbar
    ax1 = fig.add_axes([0.45, 0.70, 0.4, 0.02])
    cb1=mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                  norm=norm,
                                  orientation='horizontal')
    ax1.set_title(label)
    remove_border(ax, left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-180, -60)
    ax.set_ylim(15, 75)
    return ax

import datetime
today = datetime.datetime(2012, 10, 2)

electoral_votes = pd.read_csv("data/electoral_votes.csv").set_index('State')

predictwise = pd.read_csv('data/predictwise.csv').set_index('States')
# print(electoral_votes.head())
# print(predictwise.head())
# make_map(predictwise.Obama, "Obama Votes")

def simulate_election(model,n_sim):
    simulations = np.random.uniform(size=(51, n_sim))
    obama_votes = (simulations < model.Obama.values.reshape(-1, 1)) * model.Votes.values.reshape(-1, 1)
    return obama_votes.sum(axis=0)

result=simulate_election(predictwise, 10000)

def plot_simulation(simulation):
    plt.hist(simulation,bins=np.arange(240, 380, 1),label='simulations')
    plt.axvline(332, 0, .5, color='r', label='Actual Outcome')
    plt.axvline(269, 0, .5, color='k', label='Victory Threshold')
    p05 = np.percentile(simulation, 5.)
    p95 = np.percentile(simulation, 95.)
    iq = int(p95 - p05)
    pro= ((simulation >= 269).mean() * 100)
    plt.title('chance of Obama: %.2f' %(pro))
    plt.xlabel("Obama Electoral College Votes")
    plt.ylabel("Probability")
    remove_border()
    plt.show()

gallup_2012=pd.read_csv("data/g12.csv").set_index('State')
gallup_2012["Unknown"] = 100 - gallup_2012.Democrat - gallup_2012.Republican

def simple_gallup_model(gallup):
    return pd.DataFrame(dict(Obama=(gallup.Dem_Adv > 0).astype(float)))

# model = simple_gallup_model(gallup_2012)
# model = model.join(electoral_votes)
# prediction = simulate_election(model, 10000)
# plot_simulation(prediction)
# make_map(model.Obama, "P(Obama): Simple Model")

from scipy.special import erf
def uncertain_gallup_model(gallup):
    sigma = 3
    a=gallup.Dem_Adv / np.sqrt(2 * sigma**2)
    prob = .5 * (1 + erf(a))
    return pd.DataFrame(dict(Obama=prob), index=gallup.index)

# model = uncertain_gallup_model(gallup_2012)
# model = model.join(electoral_votes)
# prediction = simulate_election(model, 10000)
# plot_simulation(prediction)
# make_map(model.Obama, "P(Obama): Gallup + Uncertainty")
# plt.show()

def biased_gallup(gallup,bias):
    gallupC=gallup.copy()
    gallupC.Dem_Adv -= bias
    return uncertain_gallup_model(gallupC)


gallup_08 = pd.read_csv("data/g08.csv").set_index('State')
results_08 = pd.read_csv('data/2008results.csv').set_index('State')
#
prediction_08 = gallup_08[['Dem_Adv']]
prediction_08['Dem_Win']=results_08["Obama Pct"] - results_08["McCain Pct"]
#
#
# plt.plot(prediction_08.Dem_Adv, prediction_08.Dem_Win, 'o')
# plt.xlabel("2008 Gallup Democrat Advantage")
# plt.ylabel("2008 Election Democrat Win")
# fit = np.polyfit(prediction_08.Dem_Adv, prediction_08.Dem_Win, 1)
# x = np.linspace(-40, 80, 10)
# y = np.polyval(fit, x)
# plt.plot(x, y)
# print(fit)
#
# prediction_08[(prediction_08.Dem_Win < 0) & (prediction_08.Dem_Adv > 0)]
# print((prediction_08.Dem_Adv - prediction_08.Dem_Win).mean())

national_results=pd.read_csv("data/nat.csv")
national_results.set_index('Year',inplace=True)
polls04=pd.read_csv("data/p04.csv")
polls04.State=polls04.State.replace(states_abbrev)
polls04.set_index("State", inplace=True);
pvi08=polls04.Dem - polls04.Rep - (national_results.xs(2004)['Dem'] - national_results.xs(2004)['Rep'])

e2008=pd.DataFrame(dict(pvi=pvi08, Dem_Win = prediction_08.Dem_Win, Dem_Adv=prediction_08.Dem_Adv-prediction_08.Dem_Adv.mean()))
e2008['obama_win']=1*(prediction_08.Dem_Win > 0)
e2008 = e2008.sort_index()

pvi12 = e2008.Dem_Win - (national_results.xs(2008)['Dem'] - national_results.xs(2008)['Rep'])
e2012 = pd.DataFrame(dict(pvi=pvi12, Dem_Adv=gallup_2012.Dem_Adv - gallup_2012.Dem_Adv.mean()))
e2012 = e2012.sort_index()

results2012 = pd.read_csv("data/2012results.csv")
results2012.set_index("State", inplace=True)
results2012 = results2012.sort_index()

plt.plot(e2008.pvi, e2012.pvi, 'o', label='Data')
fit = np.polyfit(e2008.pvi, e2012.pvi, 1)
x = np.linspace(-40, 80, 10)
y = np.polyval(fit, x)
plt.plot(x, y, label='Linear fit')
plt.xlabel("2004 PVI")
plt.ylabel("2008 PVI")
plt.legend(loc='upper left')

plt.xlabel("Gallup Democrat Advantage (from mean)")
plt.ylabel("pvi")
colors=["red","blue"]
ax=plt.gca()
for label in [0, 1]:
    color = colors[label]
    mask = e2008.obama_win == label
    l = '2008 McCain States' if label == 0 else '2008 Obama States'
    ax.scatter(e2008[mask]['Dem_Adv'], e2008[mask]['pvi'], c=color, s=60, label=l)

ax.scatter(e2012['Dem_Adv'], e2012['pvi'], c='gray', s=60,
           marker="s", label='2012 States', alpha=.3)
plt.legend(frameon=False, scatterpoints=1, loc='upper left')
remove_border()

from sklearn.linear_model import LogisticRegression

def prepare_features(frame2008, featureslist):
    y= frame2008.obama_win.values
    X = frame2008[featureslist].values
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return y, X

def fit_logistic(frame2008, frame2012, featureslist, reg=0.0001):
    y, X = prepare_features(frame2008, featureslist)
    clf2 = LogisticRegression(C=reg)
    clf2.fit(X, y)
    X_new = frame2012[featureslist]
    obama_probs = clf2.predict_proba(X_new)[:, 1]

    df = pd.DataFrame(index=frame2012.index)
    df['Obama'] = obama_probs
    return df, clf2

from sklearn.grid_search import GridSearchCV

def cv_optimize(frame2008, featureslist, n_folds=10, num_p=100):
    y, X = prepare_features(frame2008, featureslist)
    clf = LogisticRegression()
    parameters = {"C": np.logspace(-4, 3, num=num_p)}
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_

def cv_and_fit(frame2008, frame2012, featureslist, n_folds=5):
    bp, bs = cv_optimize(frame2008, featureslist, n_folds=n_folds)
    predict, clf = fit_logistic(frame2008, frame2012, featureslist, reg=bp['C'])
    return predict, clf

res, clf = cv_and_fit(e2008, e2012, ['Dem_Adv', 'pvi'])
predict2012_logistic = res.join(electoral_votes)

prediction = simulate_election(predict2012_logistic, 10000)
plot_simulation(prediction)
make_map(predict2012_logistic.Obama, "P(Obama): Logistic")

from matplotlib.colors import ListedColormap
def points_plot(e2008, e2012, clf):
    """
    e2008: The e2008 data
    e2012: The e2012 data
    clf: classifier
    """
    Xtrain = e2008[['Dem_Adv', 'pvi']].values
    Xtest = e2012[['Dem_Adv', 'pvi']].values
    ytrain = e2008['obama_win'].values == 1

    X=np.concatenate((Xtrain, Xtest))

    # evenly sampled points
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    #plot background colors
    ax = plt.gca()
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=.5)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=.5)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)

    # Plot the 2008 points
    ax.plot(Xtrain[ytrain == 0, 0], Xtrain[ytrain == 0, 1], 'ro', label='2008 McCain')
    ax.plot(Xtrain[ytrain == 1, 0], Xtrain[ytrain == 1, 1], 'bo', label='2008 Obama')

    # and the 2012 points
    ax.scatter(Xtest[:, 0], Xtest[:, 1], c='k', marker="s", s=50, facecolors="k", alpha=.5, label='2012')
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)

    return ax

points_plot(e2008, e2012, clf)
plt.xlabel("Dem_Adv (from mean)")
plt.ylabel("PVI")

multipoll = pd.read_csv('data/cleaned-state_data2012.csv', index_col=0)
multipoll.State.replace(states_abbrev, inplace=True)
multipoll.start_date = multipoll.start_date.apply(pd.datetools.parse)
multipoll.end_date = multipoll.end_date.apply(pd.datetools.parse)
multipoll['poll_date'] = multipoll.start_date + (multipoll.end_date - multipoll.start_date).values / 2
multipoll['age_days'] = (today - multipoll['poll_date']).values / np.timedelta64(1, 'D')
multipoll = multipoll[multipoll.age_days > 0]
multipoll = multipoll.drop(['Date', 'start_date', 'end_date', 'Spread'], axis=1)
multipoll = multipoll.join(electoral_votes, on='State')
multipoll.dropna()


def state_average(multipoll):
    groups = multipoll.groupby('State')
    n = groups.size()
    mean = groups.obama_spread.mean()
    std = groups.obama_spread.std()
    std[std.isnull()] = .05 * mean[std.isnull()]
    return pd.DataFrame(dict(N=n, poll_mean=mean, poll_std=std))

avg = state_average(multipoll).join(electoral_votes, how='outer')


def default_missing(results):
    red_states = ["Alabama", "Alaska", "Arkansas", "Idaho", "Wyoming"]
    blue_states = ["Delaware", "District of Columbia", "Hawaii"]
    results.ix[red_states, ["poll_mean"]] = -100.0
    results.ix[red_states, ["poll_std"]] = 0.1
    results.ix[blue_states, ["poll_mean"]] = 100.0
    results.ix[blue_states, ["poll_std"]] = 0.1
default_missing(avg)

def aggregated_poll_model(polls):
    sigma = polls.poll_std
    prob =  .5 * (1 + erf(polls.poll_mean / np.sqrt(2 * sigma ** 2)))
    return pd.DataFrame(dict(Obama=prob, Votes=polls.Votes))

model = aggregated_poll_model(avg)
sims = simulate_election(model, 10000)
plot_simulation(sims)
plt.xlim(250, 400)
make_map(model.Obama, "P(Obama): Poll Aggregation")

def weights(df):
    lam_age = .5 ** (df.age_days / 30.)
    w = lam_age / df.MoE ** 2
    return w

def wmean(df):
    w = weights(df)
    result = (df.obama_spread * w).sum() / w.sum()
    return result

def wsig(df):
    return df.obama_spread.std()

def weighted_state_average(multipoll):

    groups = multipoll.groupby('State')
    poll_mean = groups.apply(wmean)
    poll_std = groups.apply(wsig)
    poll_std[poll_std.isnull()] = poll_mean[poll_std.isnull()] * .05

    return pd.DataFrame(dict(poll_mean = poll_mean, poll_std = poll_std))

average = weighted_state_average(multipoll)
average = average.join(electoral_votes, how='outer')
default_missing(average)
model = aggregated_poll_model(average)
sims = simulate_election(model, 10000)
plot_simulation(sims)
plt.xlim(250, 400)
make_map(model.Obama, "P(Obama): Weighted Polls")