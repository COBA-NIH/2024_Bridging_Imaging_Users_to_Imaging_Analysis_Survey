import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from pywaffle import Waffle
import squarify
import plotly.graph_objects as go
import textwrap 
from collections import Counter

def normalized_percent_graphs(df, columns, plot_filename, include_null=False,
                              prints=False,xlabel='Source',cmap=False):
    """
    Takes dataframes and columns (either a list or a dict
    wherein the key is the real column name and the values are
    the name to use in the figure, if different) and returns
    a graph of per-question normalized percents. Defaults to not 
    including null results but can. Makes a couple useful prints
    along the way
    """
    if type(columns)==list:
        columns = {x:x for x in columns}

    if include_null:
        fill = 'Not answered'
    else:
        fill = ''
    df.fillna(fill,inplace=True)
    if prints:
        print(f"Original shape: {df.shape}")

    #subset to rows which have an answer in any row
    rows_with_answers = '(`'+'`  != "") | (`'.join(columns)+'`  != "")'
    df = df.query(rows_with_answers)
    if prints:
        print(f"Shape after filtering: {df.shape}")

    df_list = []
    for eachcol in columns.keys():
        if not include_null:
            sub_df=df.query(f'`{eachcol}` != ""')
        else:
            sub_df=df
        if prints:
            print(sub_df.shape,sub_df[eachcol].value_counts())
        normed_df = sub_df[eachcol].value_counts(normalize=True)
        normed_df = normed_df.rename(columns[eachcol])# have to do it here, newlines in the query are sad :(
        df_list.append(normed_df)
    normed_df = pd.concat(df_list,axis=1)
    if not include_null:
        normed_df.fillna(0,inplace=True)

    melted = normed_df.melt(ignore_index=False)
    melted = melted.reset_index(names='percent')
    melted = melted.sort_values(by=['percent']).reset_index(drop=True)
    if prints:
        print("Melted results",melted)
    melted

    if not cmap:
        cmap="deep"

    p = (so.Plot(melted,x='variable',y='value',color='percent')
        .add(so.Bar(), so.Stack())
        .layout(size=(8,4))
        .scale(color=cmap)
        .label(x=xlabel,y="Fraction of answers",legend='Budget fraction')
        )
    if plot_filename[-4]!='.': #if we haven't given an extension, assume you want both png and svg
        p.save(loc=plot_filename+'.svg', bbox_inches="tight")
        p.save(loc=plot_filename+'.png', bbox_inches="tight")
    else:
        p.save(loc=plot_filename, bbox_inches="tight")

def select_all_that_apply_hist_facet(df,question_col,plot_filename,options_dict=False,
                                     facet_col=False, drop_empty=True,how='facet',delim=", ",
                                     ylabel='Options',create_other=False,**kwargs):
    """
    Make a faceted (or not) graph from a "select all that apply" column
    You can drop just empties in the facet col ('facet'), question('question'),
    neither(False) or all (True). You can choose to show facet as colors or columns,
    and optionally rename the column names with shortnames. 
    """
    df.fillna('',inplace=True)
    if drop_empty==False:
        facet_drop = False
        q_drop = False
    elif drop_empty=='facet':
        facet_drop=True
        q_drop=False
    elif drop_empty=='question':
        facet_drop=False
        q_drop=True
    else:
        facet_drop=True
        q_drop=True
    df_list = []
    if facet_col:
        facet_vals = list(set(df[facet_col]))
        if facet_drop:
            facet_vals.remove('')
    else:
        facet_vals = [0]
    for facet in facet_vals:
        if facet_col:
            sub_df = df.query(f'`{facet_col}` == "{facet}"')
        else:
            sub_df=df
        if q_drop:
            sub_df = sub_df.query(f'`{question_col}` != ""')
        flat_list = []
        for x in sub_df[question_col]:
            #messy to handle delims that are fancy, like parentheses
            if delim[0]!= ",":
                to_append = delim.split(",")[0]
                split_x = x.split(delim)
                sub_split = [x+to_append for x in split_x[:-1]]
                flat_list+= sub_split+[split_x[-1]]
            else:
                flat_list+=(x.split(delim))
        if options_dict:
            flat_list = [options_dict[x] for x in flat_list]
        if create_other:
            kept_flat_list = [x for x in flat_list if x in create_other]
            flat_list = kept_flat_list + ['Other']*(len(flat_list)-len(kept_flat_list))
        flat_series = pd.Series(flat_list,name=facet)
        value_counts = flat_series.value_counts(normalize=True)
        value_counts.rename(facet,inplace=True)
        df_list.append(value_counts)
    normed_df = pd.concat(df_list,axis=1)

    melted = normed_df.melt(ignore_index=False)
    melted = melted.reset_index(names='percent')
    melted = melted.sort_values(by=['variable','percent']).reset_index(drop=True)
    if how=='facet':
        g = sns.catplot(melted,y='percent',x='value',col='variable',kind='bar',**kwargs)
        g.set_axis_labels("Fraction of answers",ylabel)
        g.set_titles(col_template=f"{facet_col} = "+"{col_name}")
    elif how=='color':
        g = sns.catplot(melted,y='percent',x='value',hue='variable',kind='bar', **kwargs)
        g.legend.set_title(facet_col)
        plt.ylabel(ylabel)
        plt.xlabel("Fraction of answers")
    elif how=='single':
        g = sns.catplot(melted,y='percent',x='value',col='variable',kind='bar',**kwargs)
        g.set_titles(col_template=question_col)
        plt.ylabel('')
    if plot_filename[-4]!='.': #if we haven't given an extension, assume you want both png and svg
        plt.savefig(plot_filename+'.png')
        plt.savefig(plot_filename+'.svg')
    else:
        plt.savefig(plot_filename)

def wordcloud_func(col_name,new_stop_list,plot_filename,df,exclude_col_name=True,**kwargs):
    """wrapper around https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
    See that documentation for more kwargs to pass (colors, max words, etc)
    """
    if exclude_col_name:
        all_stopwords = list(STOPWORDS)+new_stop_list+col_name.split(' ')
    else:
        all_stopwords = list(STOPWORDS)+new_stop_list
    wordcloud_words = ' '.join(list(df[col_name].dropna())).lower()

    wc = WordCloud(background_color='white',min_word_length=4,min_font_size=34,
                   stopwords=all_stopwords,regexp=r"\w[\w'\/]+",**kwargs
                  ).generate(wordcloud_words)
    plt.title(col_name+"\n",wrap=True)
    plt.axis('off')
    plt.imshow(wc)
    plt.savefig(plot_filename, bbox_inches='tight')

def single_select(df, column, plot_filename,
                  type='waffle',sns_palette='Spectral',hue=None, rotate_x=False, **kwargs):
    """
    Makes plots of type 'histogram','waffle', 'square', or 'pie' for a single-select column.
    """
    colors = sns.color_palette(sns_palette, len(df[column].value_counts()))
    data = df[column].value_counts()
    if type=='waffle':
        fig = plt.figure(
            FigureClass=Waffle,
            rows=10,
            columns=10,
            values=data,
            colors=colors,
            legend={'loc': 'upper right', 'bbox_to_anchor': (1.75, 1)},
            labels=[f"{loc} ({y})\n{y/len(df)*100:.1f}%" for loc, y in data.items()],
        )
    elif type=='square':
        color=sns.color_palette(sns_palette, len(data))
        squarify.plot(sizes=data.values.tolist(),
              color=colors,
              pad=True,
              label=[f"{x} ({y})\n{y/len(df)*100:.1f}%" for x, y in data.items()],
                       )
        plt.axis("off")
    elif type=='pie':
        plt.pie(data,
                labels=[f"{x} ({y})" for x,y in data.items()],
                colors=colors, autopct='%1.1f%%', **kwargs)
    elif type=='histogram':
        if hue:    
            ax = sns.histplot(data=df, x=column,hue=hue, **kwargs)
            sns.move_legend(ax, "upper right", bbox_to_anchor=(1.5, 1))
        else:
            ax = sns.histplot(data=df, x=column, **kwargs)
        if rotate_x:
            ax.tick_params(axis='x', rotation=90)
    plt.title(column+"\n",wrap=True)
    if plot_filename[-4]!='.': #if we haven't given an extension, assume you want both png and svg
        plt.savefig(plot_filename+'.png', bbox_inches="tight")
        plt.savefig(plot_filename+'.svg', bbox_inches="tight")
    else:
        plt.savefig(plot_filename, bbox_inches="tight")

def make_counts_per_multiselect_group(data,single_select_col=False,multi_select_col=False):
    # input whole df as columns of multiselects
    if not single_select_col and not multi_select_col:
        all_vals = []
        for col in data.columns:
            vals = list(set([i for si in [x.split(', ') for x in data[col].dropna().to_list()] for i in si]))
            all_vals+=vals
        all_vals = list(set(all_vals))
        df_grouped = pd.DataFrame(columns=data.columns,index=all_vals)
        for col in data.columns:
            countdic = Counter([i for si in [x.split(', ') for x in data[col].dropna().to_list()] for i in si])
            for i in countdic:
                df_grouped.loc[i, col] = countdic[i]
        df_grouped.reset_index(inplace=True)
    # if single_select_col and multi_select_col are passed in will group by single_select_col
    # input df of any columns, filters to single_select_col and multi_select_col
    else:        
        # get possible values from multiselect category
        possible_values = list(set([i for si in [x.split(', ') for x in data[multi_select_col].to_list()] for i in si]))
        # create dataframe with counts per category
        df_sliced = data[[single_select_col, multi_select_col]].dropna()
        df_sliced[multi_select_col] = df_sliced[multi_select_col].str.split(', ')
        df_grouped = df_sliced.groupby(single_select_col).sum().reset_index()

        for val in possible_values:
            for y in df_grouped[single_select_col].unique():
                df_grouped.loc[df_grouped[single_select_col] == y, val] = df_grouped.loc[df_grouped[single_select_col] == y][multi_select_col].values.tolist()[0].count(val)
                    
        df_grouped.drop(columns=[multi_select_col], inplace=True)
    return df_grouped

def plot_single_vs_multi(df_grouped,
                        legend_title,
                        plot_filename,
                        plottitle,
                        groupbyfirstcol=True,
                        colsort=False,
                        colorlist=False):
    df = df_grouped.melt(id_vars=df_grouped.columns[0])
    if colsort:
        df = df.sort_values(by='variable', key=lambda x: x.map({k: i for i, k in enumerate(colsort)}))
    if not colorlist:
        colorlist=sns.color_palette("Paired")

    fig, ax = plt.subplots()
    ax.xaxis.set_tick_params(rotation=90)

    if groupbyfirstcol:
        df = df.rename(columns={df.columns[1]:legend_title})
        p = (so.Plot(df,x=df.columns[0],y='value', color=df.columns[1])
        .add(so.Bar(), so.Agg(),so.Stack())
        .label(title=plottitle)
        .scale(color=colorlist)
        .on(ax)
        )
    else:
        df = df.rename(columns={df.columns[0]:legend_title})
        p = (so.Plot(df,x=df.columns[1],y='value', color=df.columns[0])
        .add(so.Bar(), so.Agg(),so.Stack())
        .label(title=plottitle)
        .scale(color=colorlist)
        .on(ax)
        )
    
    if plot_filename[-4]!='.': #if we haven't given an extension, assume you want both png and svg
        p.save(loc=plot_filename+'.svg', bbox_inches="tight")
        p.save(loc=plot_filename+'.png', bbox_inches="tight")
    else:
        p.save(loc=plot_filename, bbox_inches="tight")

def df_for_percentage_stackedcharts(data):
    """
    Creation of dataframe for percentage_stacked charts
    Parameters:
    data: pd.DataFrame
    """
    df = pd.DataFrame()
    for colnames, items in data.items():
        dataframe = items.value_counts().to_frame() # counting the number of occurences of each unique values
        df = pd.concat([df,dataframe], axis=1)  # Appending to the values to the dataframe

    df.columns = data.columns
    df = df.reset_index()           #resetting the index of the dataframe such that it is easier for making graphs
    df = df.rename(columns={'index':'interest'})
    df = df.set_index('interest').transpose()

    #creating a dataframe for the percentage values
    per_df = pd.DataFrame()         #creating a separate dataframe for percentage values
    for col in df.columns:
        per_df[col] = [((i/df.iloc[0, 0:4].sum())*100) for i in df[col]]  #calculating the percentage values
        per_df[col] = per_df[col].round(decimals=1)
    per_df.index =df.index  #defining the index of the dataframe to be similar as the previous dataframe
    per_df.reset_index(inplace=True)  #resetting the index of the dataframe
    return per_df