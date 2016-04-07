from fnmatch import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pattern import web
import re
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
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


def get_poll_xml(poll_ID):
    XML=requests.get("http://charts.realclearpolitics.com/charts/%d.xml"%int(poll_ID)).text
    return XML


def _strip(s):
    """This function removes non-letter characters from a word

    for example _strip('Hi there!') == 'Hi there'
    """
    return re.sub(r'[\W_]+', '', s)


def plot_colors(xml):
    """
    Given an XML document like the link above, returns a python dictionary
    that maps a graph title to a graph color.

    Both the title and color are parsed from attributes of the <graph> tag:
    <graph title="the title", color="#ff0000"> -> {'the title': '#ff0000'}

    These colors are in "hex string" format. This page explains them:
    http://coding.smashingmagazine.com/2012/10/04/the-code-side-of-color/

    """
    dom = web.Element(xml)
    result = {}
    for graph in dom.by_tag('graph'):
        title = _strip(graph.attributes['title'])
        result[title] = graph.attributes['color']
    return result


def rcp_poll_data(xml):
    dom = web.Element(xml)
    result = {}

    dates = dom.by_tag('series')[0]
    dates = {n.attributes['xid']: str(n.content) for n in dates.by_tag('value')}

    keys = dates.keys()

    result['date'] = pd.to_datetime([dates[k] for k in keys])

    for graph in dom.by_tag('graph'):
        name = graph.attributes['title']
        data = {n.attributes['xid']: float(n.content)
                if n.content else np.nan for n in graph.by_tag('value')}
        result[name] = [data[k] for k in keys]

    result = pd.DataFrame(result)
    result = result.sort(columns=['date'])

    return result


def poll_plot(poll_id):
    """
    Make a plot of an RCP Poll over time

    Parameters
    ----------
    poll_id : int
        An RCP poll identifier
    """
    xml = get_poll_xml(poll_id)
    data = rcp_poll_data(xml)
    colors = plot_colors(xml)

    #remove characters like apostrophes
    data = data.rename(columns = {c: _strip(c) for c in data.columns})

    #normalize poll numbers so they add to 100%
    norm = data[colors.keys()].sum(axis=1) / 100
    for c in colors.keys():
        data[c] /= norm

    for label, color in colors.items():
        plt.plot(data.date, data[label], color=color, label=label)

    plt.xticks(rotation=70)
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Normalized Poll Percentage")


def is_gov_race(l):
    """return True if a URL refers to a Governor race"""
    pattern = 'http://www.realclearpolitics.com/epolls/????/governor/??/*-*.html'
    return fnmatch(l, pattern)


def find_governor_races(html):
    dom = web.Element(html)
    links = [a.attributes.get('href', '') for a in dom.by_tag('a')]
    links = [l for l in links if is_gov_race(l)]
    #eliminate duplicates!
    links = list(set(links))
    return links

def race_result(url):
    dom = web.Element(requests.get(url).text)

    table = dom.by_tag('div#polling-data-rcp')[0]
    result_data = table.by_tag('tr.final')[0]
    td = result_data.by_tag('td')

    results = [float(t.content) for t in td[3:-1]]
    tot = sum(results) / 100

    #get table headers
    headers = table.by_tag('th')
    labels = [str(t.content).split('(')[0].strip() for t in headers[3:-1]]

    return {l:r / tot for l, r in zip(labels, results)}

def id_from_url(url):
    """Given a URL, look up the RCP identifier number"""
    return url.split('-')[-1].split('.html')[0]


def plot_race(url):
    """Make a plot summarizing a senate race

    Overplots the actual race results as dashed horizontal lines
    """
    #hey, thanks again for these functions!
    id = id_from_url(url)
    xml = get_poll_xml(id)
    colors = plot_colors(xml)

    if len(colors) == 0:
        return

    #really, you shouldn't have
    result = race_result(url)

    poll_plot(id)
    plt.xlabel("Date")
    plt.ylabel("Polling Percentage")
    for r in result:
        plt.axhline(result[r], color=colors[_strip(r)], alpha=0.6, ls='--')
        
def party_from_color(color):
    if color in ['#0000CC', '#3B5998']:
        return 'democrat'
    if color in ['#FF0000', '#D30015']:
        return 'republican'
    return 'other'

def error_data(url):
    """
    Given a Governor race URL, download the poll data and race result,
    and construct a DataFrame with the following columns:

    candidate: Name of the candidate
    forecast_length: Number of days before the election
    percentage: The percent of poll votes a candidate has.
                Normalized to that the canddidate percentages add to 100%
    error: Difference between percentage and actual race reulst
    party: Political party of the candidate

    The data are resampled as necessary, to provide one data point per day
    """

    id = id_from_url(url)
    xml = get_poll_xml(id)

    colors = plot_colors(xml)
    if len(colors) == 0:
        return pd.DataFrame()

    df = rcp_poll_data(xml)
    result = race_result(url)

    #remove non-letter characters from columns
    df = df.rename(columns={c: _strip(c) for c in df.columns})
    for k, v in result.items():
        result[_strip(k)] = v

    candidates = [c for c in df.columns if c is not 'date']

    #turn into a timeseries...
    df.index = df.date

    #...so that we can resample at regular, daily intervals
    df = df.resample('D')
    df = df.dropna()

    #compute forecast length in days
    #(assuming that last forecast happens on the day of the election, for simplicity)
    forecast_length = (df.date.max() - df.date).values
    forecast_length = forecast_length / np.timedelta64(1, 'D')  # convert to number of days

    #compute forecast error
    errors = {}
    normalized = {}
    poll_lead = {}

    for c in candidates:
        #turn raw percentage into percentage of poll votes
        corr = df[c].values / df[candidates].sum(axis=1).values * 100.
        err = corr - result[_strip(c)]

        normalized[c] = corr
        errors[c] = err

    n = forecast_length.size

    result = {}
    result['percentage'] = np.hstack(normalized[c] for c in candidates)
    result['error'] = np.hstack(errors[c] for c in candidates)
    result['candidate'] = np.hstack(np.repeat(c, n) for c in candidates)
    result['party'] = np.hstack(np.repeat(party_from_color(colors[_strip(c)]), n) for c in candidates)
    result['forecast_length'] = np.hstack(forecast_length for _ in candidates)

    result = pd.DataFrame(result)
    return result

page = requests.get('http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html').text.encode('ascii', 'ignore')


def all_error_data():
    data = [error_data(race_page) for race_page in find_governor_races(page)]
    return pd.concat(data, ignore_index=True)

errors = all_error_data()
errors.error.hist(bins=50)
plt.xlabel("Polling Error")
plt.ylabel('N')


def bootstrap_result(c1, c2, errors, nsample=1000):
    """
    Given the current polling data for 2 candidates, return the
    bootstrap-estimate for the win probability of each candidate

    Parameters
    ----------
    c1 : float
       The current proportion of poll votes for candidate 1
    c2 : float
       The current proportio of poll votes for candidate 2
    errors : DataFrame
       The errors DataFrame
    nsample : int
       The number of bootstrap iteraionts. Default=1000

    Returns
    -------
    p1, p2
    The probability that each candidate will win, based on the bootstrap simulations
    """
    #first, normalize votes to 100
    tot = (c1 + c2)
    c1 = 100. * c1 / tot
    c2 = 100. * c2 / tot

    indices = np.random.randint(0, errors.shape[0], nsample)
    errors = errors.error.irow(indices).values

    #errors are symmetrical -- an overestimate for candidate 1
    #is an underestimate for candidate 2
    c1_actual = c1 - errors
    c2_actual = c2 + errors

    p1 = (c1_actual > c2_actual).mean()
    p2 = 1 - p1
    return p1, p2


#Look up the data as of 9/24/2013
#virginia
nsample = 10000
mcauliffe, cuccinelli = 43.0, 39.0

pm, pc = bootstrap_result(mcauliffe, cuccinelli, errors, nsample=nsample)
print "Virginia Race"
print "-------------------------"
print "P(McAuliffe wins)  = %0.2f" % pm
print "P(Cuccinelli wins) = %0.2f" % pc

#new jersey
print "\n\n"
print "New Jersey Race"
print "-----------------------"
christie, buono = 55.4, 31.8
pc, pb = bootstrap_result(christie, buono, errors, nsample=nsample)
print "P(Christie wins) = %0.2f" % pc
print "P(Buono wins)    = %0.2f" % pb



