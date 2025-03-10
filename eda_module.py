#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


class EDA:
    def __init__(self):
        self.data = None
        self.cleaned_data = None

    def load_data(self, file_path="people.data"):
        custom_headers = [
            ("age", int),
            ("workclass", object),
            ("fnlwgt", int),
            ("education", object),
            ("education_num", int),
            ("marital_status", object),
            ("occupation", object),
            ("relationship", object),
            ("race", object),
            ("sex", object),
            ("capital-gain", int),
            ("capital_loss", int),
            ("hours_per_week", int),
            ("native_country", object),
            ("income", object)
        ]

        column_dtype_map = {col: dtype for col, dtype in custom_headers}
        data = pd.read_csv(file_path, header=None, names=[col for col, _ in custom_headers], dtype=column_dtype_map)
        
        return data
    
    def show_feature_class_distribution(self, data, feature, ax=None):
        class_counts = data[feature].value_counts()
        if ax is None:
            plt.bar(class_counts.index, class_counts.values, color='yellow')
            plt.xticks(rotation=30)
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(f'Class Distribution ({feature})')
        else:
            ax.bar(class_counts.index, class_counts.values, color='yellow')
            ax.set_xticklabels(class_counts.index, rotation=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('count')
            ax.set_title(f'Class Distribution ({feature})')

    def clean_data(self, data):
        # Check if any column contains '?'
        has_question_mark = data.apply(lambda col: col.astype(str).str.contains('\?').any())

        # Print the columns containing '?'
        columns_with_question_mark = has_question_mark[has_question_mark].index.tolist()

        # Creating new DataFrame
        data_cleaned = data.copy()

        for feature in columns_with_question_mark:
            data_cleaned = data_cleaned[data_cleaned[feature] != ' ?']

        # Resetting the index
        data_cleaned.reset_index(drop=True, inplace=True)

        return data_cleaned
    
    def box_plot(self, data_cleaned, x_feature, y_feature, ax=None):    
        if ax is None:
            sns.boxplot(x=x_feature, y=y_feature, data=data_cleaned)
            plt.xticks(rotation = 30)
            plt.title(f'Box Plot: {x_feature} vs {y_feature}')
        else:  
            sns.boxplot(x=x_feature, y=y_feature, data=data_cleaned)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_title(f'Box Plot: {x_feature} vs {y_feature}')
    
    def heatmap(self, data_cleaned, x_feature, y_feature, ax=None):
        if ax is None:
            crosstab = pd.crosstab(data_cleaned[y_feature], data_cleaned[x_feature])
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Heatmap: {x_feature} vs {y_feature}')
        else:
            crosstab = pd.crosstab(data_cleaned[y_feature], data_cleaned[x_feature])
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Heatmap: {x_feature} vs {y_feature}')
    
    def class_distribution_features_needed(self, data_cleaned, feature, ax=None):
        class_counts = data_cleaned[feature].value_counts()
        if ax is None:
            plt.bar(class_counts.index, class_counts.values, color='green')
            plt.xticks(rotation = 30)
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(f'Class Distribution ({feature})')
        else:
            ax.bar(class_counts.index, class_counts.values, color='green')
            ax.set_xticklabels(class_counts.index, rotation=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('count')
            ax.set_title(f'Class Distribution ({feature})')
    
    def corr_heatmap(self, data_encoded, ax=None):
        if ax is None:
            sns.heatmap(data_encoded.corr(), annot=True, cmap='copper', fmt=".2f")
            plt.title('Correlation Heatmap', color='Black', fontsize=23)
        else:
            sns.heatmap(data_encoded.corr(), annot=True, cmap='copper', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap')

