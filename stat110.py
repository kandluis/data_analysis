
# coding: utf-8

# In[52]:

get_ipython().magic(u'matplotlib')
import seaborn as sns

import pandas as pd
import numpy as np

import seaborn

from matplotlib import pylab as plt

from __future__ import with_statement


# In[53]:

gradefile = "stat110.csv"
grades = pd.read_csv(gradefile)

# drop top two rows (:points possible, NaN)
grades = grades.drop([0,1],axis=0)


# In[54]:

# we're writing out student names so we can fill in gender data manually 
sfile = "students.csv"

try:
    rstudents = pd.read_csv(sfile)
    grades['Gender'] = list(rstudents['Gender'])
except EnvironmentError:
    grades['Gender'] = None


# In[55]:

grades.columns


# In[66]:

def getpset(pset):
    subset = grades
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
    subset['Student'] = grades.Student
    subset['Gender'] = grades.Gender
        
    # reorder columns
    subset = subset.ix[:,[subset.columns[-2],subset.columns[-1]] + list(subset.columns[:len(subset.columns) - 2])]
    # print len(subset)
    return subset


# In[67]:

def make_corr_plot(d, title="plot"):
    f, ax = plt.subplots(figsize=(9, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.corrplot(d, annot=False, sig_stars=False,
                 diag_names=False, cmap=cmap, ax=ax)
    f.tight_layout()
    plt.title(title)
    f.savefig(title)


# In[68]:

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
    ax.hist(nd,bins=range(0,42+step,step),alpha=0.8,align='mid')
    plt.title(title + " Histogram")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    fig.savefig(title + ".png")


# In[69]:

# on campus mean
def stats(pset, percent=True):
    if percent:
        print "mean: {:.2f}% (standard deviation {:.2f}%) and median {:.2f}%. Lower Quartile: {:.2f}%. Upper Quartile: {:.2f}%. Min: {:.2f}%, Max: {:.2f}%".format(
            np.mean(pset), np.std(pset),np.median(pset), np.percentile(pset,25), np.percentile(pset,75), min(pset),max(pset))
    else:
        print "mean: {:.2f} (standard deviation {:.2f}) and median {:.2f}. Lower Quartile: {:.2f}. Upper Quartile: {:.2f}. Min: {:.2f}, Max: {:.2f}".format(
            np.mean(pset), np.std(pset),np.median(pset), np.percentile(pset,25), np.percentile(pset,75), min(pset),max(pset))


# In[70]:

hw1 = getpset("Homework 1")
hw2 = getpset("Homework 2")
# print len(grades[grades["Homework 2 (46145)"] != 0]["Homework 2 (46145)"])


# In[71]:

len(hw2['Homework 2 '])


# In[72]:

make_histogram(hw2['Homework 2 '], 'Problem Set 2', step=5)


# In[73]:

stats(hw2['Homework 2 '], percent=False)


# In[14]:

final = getpset("^Final Score")
finale = getpset("^Final Score", True)
len(finale)


# In[15]:

finale


# In[16]:

finale = finale[[student in list(pset8e.Student)for student in finale.Student]]


# In[31]:

make_histogram(final['Final Score'], 'Cumulative Grade', step=5)
make_histogram(finale['Final Score'], 'Cumulative Grade (extension)', step=15)


# In[35]:

stats(final['Final Score'])
stats(finale['Final Score'])


# In[24]:

grades.head(10)


# In[21]:

midterm2 = getpset("Midterm 2")
midterm2e = getpset("Midterm 2", True)


# In[22]:

midterm2.ix[:,:8].head(10)


# In[24]:

m2 = midterm2.ix[:,:6].astype(float)
m2e = midterm2e.ix[:,:6].astype(float)

totals = [8.,10.,11.,8.,10.,8.]
for i,total in enumerate(totals):
    m2.ix[:,i] = m2.ix[:,i] / total
    m2e.ix[:,i] = m2e.ix[:,i] / total

corrm2 = m2.corr()
corrm2e = m2e.corr()


# In[51]:

m2.max(axis=0), midterm2.max(axis=0)


# In[98]:

m2.head(10)


# In[65]:

make_corr_plot(m2, "Midterm 2 Correlation")
make_corr_plot(m2e, "Midterm 2 Correlation Extention")


# In[30]:

corrm2.head(10)


# In[66]:

stats(midterm2['Midterm 2 Current Score'])
stats(midterm2e['Midterm 2 Current Score'])


# In[32]:

for col, cole in zip(exam.columns[2:],exame.columns[2:]):
    print "Statistics for {}.".format(col)
    stats(exam[col],False)
    stats(exame[cole],False)


# In[91]:

make_histogram(midterm2['Midterm 2 Current Score'], 'Midterm 2',numBins=10)
make_histogram(midterm2e['Midterm 2 Current Score'], 'Midterm 2',numBins=5)


# In[24]:

ungraded = ['Extra Credit Final Score', 'Midterm Private Final Score']
percentiles = getpset("Final Score").drop(ungraded,axis=1)
percentilese = getpset("Final Score",True).drop(ungraded,axis=1)


# In[25]:

percentiles.head(10)


# In[26]:

# normalize the percentiles
corrp = percentiles.corr()
corrpe = percentilese.corr()


# In[27]:

corrpe


# In[28]:

make_corr_plot(percentiles,title="Grade Correlation")
make_corr_plot(percentilese,title="Grade Correlation Extension")


# In[106]:

corrm2 = m2.div(pd.Series([8.,10.,11.,8.,10.,8.]).iloc[0],axis='index')
corrm2e = m2e.div(pd.Series([8.,10.,11.,8.,10.,8.]).iloc[0],axis='index')


# In[113]:

c2 = ((corrm2 - corrm2.mean(axis=0)) / corrm2.std(axis=0)).corr()
c2e = ((corrm2e - corrm2e.mean(axis=0)) / corrm2e.std(axis=0)).corr()


# In[ ]:

make_corr_plot(c2,title="Grade Correlation")
make_corr_plot(c2e,title="Grade Correlation Extension")


# In[131]:

def make_scatter(xaxis, yaxis = "Midterm 2 Final Score"):
    fig = plt.figure()
    plt.scatter(percentiles[xaxis], percentiles[yaxis])
    plt.title("{} vs {}".format(yaxis,x))


# In[132]:

psets = ["Problem Set {} Final Score".format(x) for x in xrange(1,7)]
psets


# In[ ]:

for pset in psets:
    make_scatter(pset)


# In[139]:

def make_scatter_index(yaxis = "Midterm 2 Final Score"):
    fig = plt.figure()
    plt.scatter(percentiles.index, percentiles[yaxis])
    plt.title("{} vs index".format(yaxis))


# In[140]:

for pset in psets:
    make_scatter_index(pset)


# In[29]:

# now we analyze gender data
male = percentiles[percentiles.Gender == "M"]
female = percentiles[percentiles.Gender == "F"]
malee =percentilese[percentilese.Gender == "M"]
femalee =percentilese[percentilese.Gender == "F"]


# In[33]:

male.mean(), len(male)


# In[32]:

female.mean(), len(female)


# In[34]:

malee.mean(), len(malee)


# In[35]:

femalee.mean(), len(femalee)


# In[38]:

make_histogram(male['Problem Set 6 Final Score'], 'pset6_hist_male', 10)
make_histogram(female['Problem Set 6 Final Score'], 'pset6_hist_female',10)


# In[40]:

make_histogram(male['Midterm 2 Final Score'], 'pset6_hist_male', 10)
make_histogram(female['Midterm 2 Final Score'], 'pset6_hist_female',10)


# In[33]:

allgrades = getpset("Problem Set [0-9] Final|Midterm [0-2] Final")
allgradese = getpset("Problem Set [0-9] Final|Midterm [0-2] Final", True)


# In[34]:

make_corr_plot(allgrades, "All Grades")


# In[27]:

allproblems = getpset("Pset [0-9]|Midterm [0-2] Final")
allproblemse = getpset("Pset [0-9]|Midterm [0-2] Final", True)


# In[28]:

make_corr_plot(allproblems, "All Problems")


# In[ ]:



