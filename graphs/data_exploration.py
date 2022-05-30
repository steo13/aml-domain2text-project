import json
from matplotlib import markers
import matplotlib.pyplot as plt
from numpy import size

from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px

AVERAGE_SIMILARITIES_FILE = './average_similarities.txt'
PACS_EMBEDDING_FOLDER = './aml-domain2text-project/graphs/'

'''
Barplot target domains vs average distances from sources
@ return plytplot barplot
'''
def average_similarities ():
    plt.figure(figsize=(10, 8))
    bar_width = 0.25
    labels = []
    bars = {}
    DOMAINS = ['Cartoon', 'Photo', 'Sketch', 'ArtPainting']
    COLORS = ['g', 'yellow', 'b', 'r']

    file = open(AVERAGE_SIMILARITIES_FILE, 'r')
    for line in file:
        dict = json.loads(line)
        key = list(dict.keys())[0]
        labels.append(key)
        for domain in dict[key]:
            if domain in bars:
                bars[domain].append(dict[key][domain])
            else:
                bars[domain] = []
                bars[domain].append(dict[key][domain])

    rows = [[0, 2, 3], [0 + bar_width, 1, 2 + bar_width], [0 + 2*bar_width, 1 + bar_width, 3 + bar_width], [1 + 2*bar_width, 2 + 2*bar_width, 3 + 2*bar_width]]

    for (index, row) in enumerate(rows):
        plt.bar(row, bars[DOMAINS[index]], width=bar_width, edgecolor='white', color=COLORS[index], label=DOMAINS[index])

    for index, value in enumerate(bars['Cartoon']):
        plt.text(rows[0][index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    for index, value in enumerate(bars['Photo']):
        plt.text(rows[1][index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    for index, value in enumerate(bars['Sketch']):
        plt.text(rows[2][index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    for index, value in enumerate(bars['ArtPainting']):
        plt.text(rows[3][index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    
    plt.xticks([r + bar_width for r in range(4)], ['ArtPainting', 'Cartoon', 'Sketch', 'Photo'])
    plt.xlabel('Target domains', fontweight='bold')
    plt.ylabel('Average similarities from sources', fontweight='bold')
    plt.grid(color='black', linestyle='--', linewidth=0.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, title="Sources")
    plt.show()

def PCA_embedding():
    df = []
    final_df = []
    DOMAINS = ['ArtPainting', 'Cartoon', 'Sketch', 'Photo']
    COLORS = ['r', 'g', 'b', 'y']

    for domain in DOMAINS:
        local_df = pd.read_csv(PACS_EMBEDDING_FOLDER+domain+'.csv')
        # Separating target
        local_df.drop('Target', inplace=True, axis=1)
        # Separating index
        local_df.drop('Index', inplace=True, axis=1)
        df.append(local_df)

        x = local_df.T

        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
        target = pd.Series([domain for _ in range(0, len(x))])
        local_final_df = pd.concat([principal_df, target], axis=1)
        local_final_df = local_final_df.rename(columns={0: "Target"})
        final_df.append(local_final_df)
    
    # PCA-3 aggregated and to csv
    PCA_df = pd.concat(final_df, ignore_index=True)
    # export as csv file more analysis
    PCA_df.to_csv(PACS_EMBEDDING_FOLDER+'PACS_PCA.csv')
    print("Navigate through 3-PCA PACS embedding!\n")
    # express plot
    color_discrete_map = {'ArtPainting': 'rgb(255,0,0)', 'Cartoon': 'rgb(0,128,0)', 'Sketch': 'rgb(0,0,255)', 'Photo': 'rgb(255,255,51)'}
    fig = px.scatter_3d(
      PCA_df,
      x='PC1', 
      y='PC2', 
      z='PC3', 
      color='Target',
      opacity=0.7,
      color_discrete_map=color_discrete_map,
      width=800, height=600
    )
    fig.update_traces(marker_size=2)
    fig.show()

    # matplotlib plot
    print("\nStatic view of 3-PCA PACS embedding!\n")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_zlabel('PC3', fontsize=15)
    ax.set_title('3 components PCA', fontsize=15)
    for (index, f_df) in enumerate(final_df):
        ax.scatter(f_df.loc[:, 'PC1'], f_df.loc[:, 'PC2'], f_df.loc[:, 'PC3'], c=COLORS[index], s=2)
    ax.legend(DOMAINS)
    ax.grid()
    plt.show()

if __name__ == '__main__':
    # Plot the average distances between domains
    print("Average similarities from Source to Target domains\n")
    average_similarities()
    # plot PCA
    print("\nPCA applied on the embeddings of PACS dataset\n")
    PCA_embedding()
