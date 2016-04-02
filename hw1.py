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
    XML=requests.get("http://charts.realclearpolitics.com/charts/%d.xml"%poll_ID).text
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

