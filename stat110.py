
# coding: utf-8

# In[24]:

get_ipython().magic(u'matplotlib')
import seaborn as sns

import pandas as pd
import numpy as np

import seaborn

from matplotlib import pylab as plt

from collections import defaultdict

from __future__ import with_statement


# In[4]:

gradefile = "stat110.csv"


# In[5]:

grades = pd.read_csv(gradefile)


# In[6]:

# drop top two rows (:points possible, NaN)
grades = grades.drop([0,1],axis=0)


# In[7]:

# we're writing out student names so we can fill in gender data manually 
sfile = "students.csv"

try:
    rstudents = pd.read_csv(sfile)
    grades['Gender'] = list(rstudents['Gender'])
except EnvironmentError:
    grades['Gender'] = None


# In[8]:

google_doc = "google_doc.csv"
google = pd.read_csv(google_doc)
names = pd.read_csv("names.csv")


# In[9]:

grades


# In[10]:

names.columns, google.columns, grades.columns


# In[11]:

# Join the names to the midterms, so then we can join the result to the canvas data
googleNames = pd.merge(google, names, left_on="Name", right_on="Doc")

# For some reason, 'HW2 electronic?' column comes in as 'object' type rather than float.
# This is likely due to the fact that someone input Score(...) as a value for the column.
# We do this for all the columns in place!
keys = ['HW2', 'HW3', 'HW4', 'HW5', 'HW6', 'HW7']
for key in keys:
    googleNames[key + ' electronic?'] = googleNames[key + ' electronic?'].convert_objects(convert_numeric=True)
print "Finished converting all of the values!"


# In[12]:

googleNames.columns


# In[13]:

# Now join to the canvas
joinedResults = pd.merge(googleNames, grades, left_on="Canvas", right_on="Student")
joinedResults.columns


# In[14]:

joinedResults['Homework 7 (53575)']


# In[15]:

# Keep the important columns
toKeep = ['Doc', 'M1', 'M2', 'M3', 'M4', 'M', 'Homework 1 (42101)', 'Homework 2 (46145)', 'Homework 3 (47478)', 'Homework 4 (48482)']
toKeep += ['HW4 electronic?', 'HW3 electronic?', 'HW2 electronic?', 'Student', 'Gender' ]
toKeep += ['Homework 7 (53575)', 'HW7 electronic?', 'Homework 6 (52035)', 'HW6 electronic?', 'Homework 5 (50651)', 'HW5 electronic?']


# In[16]:

dataAll = joinedResults[toKeep]


# In[17]:

def getpset(pset, source=grades):
    subset = source
    # print len(grades)
    # make prettry
    import re
    subset = subset.filter(regex=pset)
    subset.columns = [re.sub(r'\([^)]*\)', '', name) for name in subset.columns]
    
    subset = subset.convert_objects(convert_numeric=True)
    # print len(subset)
    
    # we remove user who received a score of zero if from extension
    subset = subset[subset[subset.columns[-1]] > 0]
    # print len(subset)
    
    # we add in the Student and Gender Columns
    subset['Student'] = source.Student
    subset['Gender'] = source.Gender
        
    # reorder columns
    subset = subset.ix[:,[subset.columns[-2],subset.columns[-1]] + list(subset.columns[:len(subset.columns) - 2])]
    # print len(subset)
    return subset


# In[18]:

def make_corr_plot(d, title="plot"):
    f, ax = plt.subplots(figsize=(9, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.corrplot(d, annot=False, sig_stars=False,
                 diag_names=False, cmap=cmap, ax=ax)
    f.tight_layout()
    plt.title(title)
    f.savefig(title)


# In[19]:

def make_histogram(d, title="histogram",xlabel="Score (ouf of 42)",step=5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = d.dropna()
    nd = []
    for el in d:
        try:
            if el != 0:
                nd.append(float(el))
            else:
                print "Removed grade"
        except ValueError:
            pass
    ax.hist(nd,bins=range(0,int(max(d))+step,step),alpha=0.8,align='mid')
    plt.title(title + " Histogram")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    fig.savefig(title + ".png")


# In[20]:

# on campus mean
def stats(pset, percent=True):
    if percent:
        print "Mean: {:.2f}% (standard deviation {:.2f}%)\n Median {:.2f}%\n Lower Quartile: {:.2f}%\n Upper Quartile: {:.2f}%\n Min: {:.2f}%\n Max: {:.2f}%".format(
            np.mean(pset), np.std(pset),np.median(pset), np.percentile(pset,25), np.percentile(pset,75), min(pset),max(pset))
    else:
        print "Mean: {:.2f} (standard deviation {:.2f})\n Median {:.2f}\n Lower Quartile: {:.2f}\n Upper Quartile: {:.2f}\n Min: {:.2f}\n Max: {:.2f}".format(
            np.mean(pset), np.std(pset),np.median(pset), np.percentile(pset,25), np.percentile(pset,75), min(pset),max(pset))


# In[21]:

def filterPset(pset, t=None, source=dataAll):
    # Filters based on the type of the grade for `pset'. If t parameter is None, then we keep all the grades.
    # If t is True, we keep only the electronic copies.
    # If t is False, we keep only the paper copies.
    # pset should really be in type 'HWx electronic?'
    
    # Drop NAs
    source.dropna(subset=[pset])
    
    # Which do we keep?
    keepType = [1, 0] if t is None else [1] if t else [0]
    
    # Filter on data type we've decided to keep
    return source[source[pset].isin(keepType)]


# In[22]:

len(filterPset("HW7 electronic?")), len(filterPset("HW7 electronic?", t=True)), len(filterPset("HW7 electronic?", t=False))


# In[46]:


hws = ['2', '3', '4', '5', '6', '7']
types = [('All', None), ('Electronic', True), ('Paper', False)]
statistics = {} # We have a key: (Mean, Std, Median, LQ, UQ, Min, Max)

timeSeries = defaultdict(list) # We have statistic: [] a list of items which contains the statistic for each homework in hws.
# Note that the order follows the order specified in hws!

for hw in hws:
    for text, t in types:
        res = getpset("Homework {}".format(hw), source=filterPset("HW{} electronic?".format(hw), t=t))
        
        # Calculate the statistics and print the results out
        print "Results for Homework {} ({})".format(hw, text)
        print "Number of samples is {}".format(len(res))
        column = res["Homework {} ".format(hw)]
        stats(column , percent=False)
        print ""
        
        # We're collecting some statistics so we can plot them later. 
        statistics["HW{} ({})".format(hw, text)] = (
            hw, t, 
            np.mean(column),
            np.std(column),
            np.median(column),
            np.percentile(column, 25),
            np.percentile(column, 75),
            min(column), max(column))
        
        timeSeries['{}_mean'.format(text)].append(np.mean(column))
        # timeSeries['{}_std_dev'.format(text)].append(np.std(column))
        timeSeries['{}_median'.format(text)].append(np.median(column))
        timeSeries['{}_count'.format(text)].append(len(column))
        #timeSeries['lower_quartile'].append(np.percentile(column, 25))
        #timeSeries['upper_quartile'].append(np.percentile(column, 75))
        #timeSeries['min'].append(min(column))
        #timeSeries['max'].append(max(column))


# In[52]:

for statistic in timeSeries:
    if 'std_dev' not in statistic and 'count' not in statistic:
        plt.plot(range(2,8), timeSeries[statistic], label=statistic)
plt.legend()
plt.title("Statistics Over Time for Stat 110 Homework Assignments")
plt.xlabel("Homework Number")
plt.ylabel("Statistic (out of 42)")
plt.ylim((37,42))
plt.show()


# In[53]:

for statistic in timeSeries:
    if 'count' in statistic:
        plt.plot(range(2,8), timeSeries[statistic], label=statistic)
plt.legend()
plt.title("Homework Submissions for Statistics 110")
plt.ylabel("#Submissions")
plt.xlabel("Homework Number")
plt.show()


# In[23]:

hw_means = []
hw_std = []
for hw in hws:
    hw_means.append([ (x[2]) for k,x in sorted(statistics.items()) if x[0] == hw])
    hw_std.append([ x[3] for k,x in sorted(statistics.items()) if x[0] == hw])
hw_means = pd.DataFrame(hw_means)
hw_std = pd.DataFrame(hw_std)


# In[24]:

allT, allTStd = hw_means.ix[:, 0], hw_std.ix[:, 0]
electronic, electronicStd = hw_means.ix[:, 1], hw_std.ix[:, 0]
paper, paperStd = hw_means.ix[:, 2], hw_std.ix[:, 0]


# In[25]:

hw_std


# In[26]:

N = len(hws)
ind = np.arange(N)
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(ind, allT, width, color='r', yerr=allTStd)
rects2 = ax.bar(ind+width, electronic, width, color='y', yerr=electronicStd)
rects3 = ax.bar(ind+2*width, paper, width, color='b', yerr=paperStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean Score')
ax.set_title('Mean Homework Scores by Type')
ax.set_xticks(ind+1.5*width)
ax.set_xticklabels( ('HW2', 'HW3', 'HW4', 'HW5') )

ax.legend( (rects1[0], rects2[0], rects3[0]), ('All', 'Electronic', 'Paper') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%f'%height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)


# In[27]:

hw_name = 'Homework 7 '
hw = getpset(hw_name)
make_histogram(hw[hw_name], hw_name, "Score (of of 42, step=3)", step=3)


# In[28]:

stats(hw[hw_name], percent=False)


# In[ ]:

# We reuse the homeworks to plot the average score on a pset broken by type compared with the average score
# on the midterm.
statistics_midterm = {} # We have a key: (Mean, Std, Median, LQ, UQ, Min, Max)
for hw in hws:
    for text, t in types:
        res = getpset("M", source=filterPset("HW{} electronic?".format(hw), t=t))
        midterm = res["M"]
        
        print "Results for Homework {} ({})".format(hw, text)
        print "Number of samples is {}".format(len(res))
        stats(midterm , percent=False)
        print ""
        
        # Calculate some statistics
        statistics_midterm["HW{} ({})".format(hw, text)] = (
            hw, t, 
            len(midterm),
            np.mean(midterm),
            np.std(midterm),
            np.median(midterm),
            np.percentile(midterm, 25),
            np.percentile(midterm, 75),
            min(midterm), max(midterm))


# In[ ]:

midterm_means = []
midterm_std = []
midterm_meadians = []
for hw in hws:
    midterm_means.append([ (x[3]) for k,x in sorted(statistics_midterm.items()) if x[0] == hw])
    midterm_std.append([ x[4] for k,x in sorted(statistics_midterm.items()) if x[0] == hw])
    midterm_meadians.append([ x[5] for k,x in sorted(statistics_midterm.items()) if x[0] == hw])
midterm_means = pd.DataFrame(midterm_means)
midterm_std = pd.DataFrame(midterm_std)
midterm_meadians = pd.DataFrame(midterm_meadians)


# In[ ]:

allT, allTStd = midterm_means.ix[:, 0], midterm_std.ix[:, 0]
electronic, electronicStd = midterm_means.ix[:, 1], midterm_std.ix[:, 0]
paper, paperStd = midterm_means.ix[:, 2], midterm_std.ix[:, 0]


# In[ ]:

N = len(hws)
ind = np.arange(N)
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(ind, allT, width, color='r', yerr=allTStd)
rects2 = ax.bar(ind+width, electronic, width, color='y', yerr=electronicStd)
rects3 = ax.bar(ind+2*width, paper, width, color='b', yerr=paperStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean Score')
ax.set_title('Mean Midterm Scores by Homework Type Submissions')
ax.set_xticks(ind+1.5*width)
ax.set_xticklabels( ('HW2', 'HW3', 'HW4', 'HW5') )

ax.legend( (rects1[0], rects2[0], rects3[0]), ('All', 'Electronic', 'Paper') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%f'%height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)


# In[ ]:

make_histogram(midtermClean.M, "Midterm Points", "Score (out of 50)", step=5)


# In[ ]:

stats(midtermClean.M, percent=False)


# In[ ]:

midterm2 = getpset("Midterm 2")
midterm2e = getpset("Midterm 2", True)


# In[ ]:

midterm2.ix[:,:8].head(10)


# In[ ]:

m2 = midterm2.ix[:,:6].astype(float)
m2e = midterm2e.ix[:,:6].astype(float)

totals = [8.,10.,11.,8.,10.,8.]
for i,total in enumerate(totals):
    m2.ix[:,i] = m2.ix[:,i] / total
    m2e.ix[:,i] = m2e.ix[:,i] / total

corrm2 = m2.corr()
corrm2e = m2e.corr()


# In[ ]:

m2.max(axis=0), midterm2.max(axis=0)


# In[ ]:

m2.head(10)


# In[ ]:

make_corr_plot(m2, "Midterm 2 Correlation")
make_corr_plot(m2e, "Midterm 2 Correlation Extention")


# In[ ]:

corrm2.head(10)


# In[ ]:

stats(midterm2['Midterm 2 Current Score'])
stats(midterm2e['Midterm 2 Current Score'])


# In[ ]:

for col, cole in zip(exam.columns[2:],exame.columns[2:]):
    print "Statistics for {}.".format(col)
    stats(exam[col],False)
    stats(exame[cole],False)


# In[ ]:

make_histogram(midterm2['Midterm 2 Current Score'], 'Midterm 2',numBins=10)
make_histogram(midterm2e['Midterm 2 Current Score'], 'Midterm 2',numBins=5)


# In[ ]:

ungraded = ['Extra Credit Final Score', 'Midterm Private Final Score']
percentiles = getpset("Final Score").drop(ungraded,axis=1)
percentilese = getpset("Final Score",True).drop(ungraded,axis=1)


# In[ ]:

percentiles.head(10)


# In[ ]:

# normalize the percentiles
corrp = percentiles.corr()
corrpe = percentilese.corr()


# In[ ]:

corrpe


# In[ ]:

make_corr_plot(percentiles,title="Grade Correlation")
make_corr_plot(percentilese,title="Grade Correlation Extension")


# In[ ]:

corrm2 = m2.div(pd.Series([8.,10.,11.,8.,10.,8.]).iloc[0],axis='index')
corrm2e = m2e.div(pd.Series([8.,10.,11.,8.,10.,8.]).iloc[0],axis='index')


# In[ ]:

c2 = ((corrm2 - corrm2.mean(axis=0)) / corrm2.std(axis=0)).corr()
c2e = ((corrm2e - corrm2e.mean(axis=0)) / corrm2e.std(axis=0)).corr()


# In[ ]:

make_corr_plot(c2,title="Grade Correlation")
make_corr_plot(c2e,title="Grade Correlation Extension")


# In[ ]:

def make_scatter(xaxis, yaxis = "Midterm 2 Final Score"):
    fig = plt.figure()
    plt.scatter(percentiles[xaxis], percentiles[yaxis])
    plt.title("{} vs {}".format(yaxis,x))


# In[ ]:

psets = ["Problem Set {} Final Score".format(x) for x in xrange(1,7)]
psets


# In[ ]:

for pset in psets:
    make_scatter(pset)


# In[ ]:

def make_scatter_index(yaxis = "Midterm 2 Final Score"):
    fig = plt.figure()
    plt.scatter(percentiles.index, percentiles[yaxis])
    plt.title("{} vs index".format(yaxis))


# In[ ]:

for pset in psets:
    make_scatter_index(pset)


# In[ ]:

# now we analyze gender data
male = percentiles[percentiles.Gender == "M"]
female = percentiles[percentiles.Gender == "F"]
malee =percentilese[percentilese.Gender == "M"]
femalee =percentilese[percentilese.Gender == "F"]


# In[ ]:

male.mean(), len(male)


# In[ ]:

female.mean(), len(female)


# In[ ]:

malee.mean(), len(malee)


# In[ ]:

femalee.mean(), len(femalee)


# In[ ]:

make_histogram(male['Problem Set 6 Final Score'], 'pset6_hist_male', 10)
make_histogram(female['Problem Set 6 Final Score'], 'pset6_hist_female',10)


# In[ ]:

make_histogram(male['Midterm 2 Final Score'], 'pset6_hist_male', 10)
make_histogram(female['Midterm 2 Final Score'], 'pset6_hist_female',10)


# In[ ]:

allgrades = getpset("Problem Set [0-9] Final|Midterm [0-2] Final")
allgradese = getpset("Problem Set [0-9] Final|Midterm [0-2] Final", True)


# In[ ]:

make_corr_plot(allgrades, "All Grades")


# In[ ]:

allproblems = getpset("Pset [0-9]|Midterm [0-2] Final")
allproblemse = getpset("Pset [0-9]|Midterm [0-2] Final", True)


# In[ ]:

make_corr_plot(allproblems, "All Problems")


# In[ ]:



