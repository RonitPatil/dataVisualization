import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the data 
column_names = ['age', 'work_class', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data = pd.read_csv('adult_data.txt', names=column_names, delimiter=',', skipinitialspace=True)
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

##########################BOX PLOT FOR AGES########################
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data.dropna(subset=['age'], inplace=True)

plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='age', data=data)
plt.title('Age Distribution by Income Levels')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()

##########################PIE CHART FOR WORK CLASS#####################################
def misc(series, cutoff_percent=3):
    small_categories = series[series / series.sum() * 100 < cutoff_percent]
    if not small_categories.empty:
        series = series[series / series.sum() * 100 >= cutoff_percent]
        series['Misc(Category below 3%)'] = small_categories.sum()
    return series


less_income = data[data['income'] == '<=50K']
more_income = data[data['income'] == '>50K']

less_count = misc(less_income['work_class'].value_counts())
more_count = misc(more_income['work_class'].value_counts())

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].pie(less_count, labels=less_count.index, autopct='%1.1f%%', startangle=90)
ax[0].set_title('Education Distribution for Income <=50K')
ax[1].pie(more_count, labels=more_count.index, autopct='%1.1f%%', startangle=90)
ax[1].set_title('Education Distribution for Income >50K')

plt.tight_layout()
plt.show()

############################STACKED BAR CHART FOR EDUCATION ECCUPATION MARITAL STATUS######################
grouped = data.groupby(['education', 'occupation', 'marital_status', 'income']).size().unstack('income', fill_value=0)

grouped['total'] = grouped.sum(axis=1)
main_cat = grouped.sort_values(by='total', ascending=False).head(10)

main_cat = main_cat.drop(columns='total')

ax = main_cat.plot(kind='bar', stacked=True, figsize=(14, 10), width=1)
ax.set_xlabel('Education, Occupation, Marital Status')
ax.set_ylabel('Count')
ax.set_title('Top 10 Combinations: Income by Education, Occupation, and Marital Status')
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
ax.legend(title='Income Bracket')
plt.tight_layout()

plt.show()

#######################HEATMAP FOR INDIVIDUALS BY RACE AND SEX################
data['income_binary'] = data['income'].apply(lambda x: 1 if '>50K' in x.strip() else 0)
counts = data.groupby(['race', 'sex', 'income_binary']).size().unstack(fill_value=0)

counts.columns = ['<=50K', '>50K']

plt.figure(figsize=(12, 10))
sns.heatmap(counts, annot=True, cmap='coolwarm', fmt="d")  # Use "d" for integer format
plt.title('Heatmap of Count of Individuals by Race and Sex for Different Income Brackets')
plt.ylabel('Race, Sex')
plt.tight_layout()
plt.xlabel('Income Brackets')

plt.show()

###############################SCATTERPLOT FOR AGE VS HOURS PER WEEK###########
data['income'] = data['income'].astype('category')
g = sns.FacetGrid(data, col='income', hue='income', col_wrap=2, height=5)
g.map(plt.scatter, 'hours_per_week', 'age', alpha=0.6)

g.add_legend()
g.fig.suptitle('Scatter Plot of Hours per Week vs Age, Faceted by Income Level')
g.set_axis_labels('Hours per Week', 'Age')
plt.tight_layout()
plt.show()