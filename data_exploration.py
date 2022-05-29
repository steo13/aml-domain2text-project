import json
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pandas as pd

AVERAGE_DISTANCES_FILE = './average_distances.txt'

'''
Barplot target domains vs average distances from sources
@ return plytplot barplot
'''
def average_distances ():
    plt.figure(figsize=(10, 8))
    bar_width = 0.25
    labels = []
    bars = {}
    file = open(AVERAGE_DISTANCES_FILE, 'r')
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

    r1 = [0, 2, 3]
    r2 = [0 + bar_width, 1, 2 + bar_width]
    r3 = [0 + 2*bar_width, 1 + bar_width, 3 + bar_width]
    r4 = [1 + 2*bar_width, 2 + 2*bar_width, 3 + 2*bar_width]

    plt.bar(r1, bars['Cartoon'], width=bar_width, edgecolor='white', color='green', label='Cartoon')
    plt.bar(r2, bars['Photo'], width=bar_width, edgecolor='white', color='yellow', label='Photo')
    plt.bar(r3, bars['Sketch'], width=bar_width, edgecolor='white', color='blue', label='Sketch')
    plt.bar(r4, bars['ArtPainting'], width=bar_width, edgecolor='white', color='red', label='ArtPainting')
    
    for index, value in enumerate(bars['Cartoon']):
        plt.text(r1[index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    for index, value in enumerate(bars['Photo']):
        plt.text(r2[index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    for index, value in enumerate(bars['Sketch']):
        plt.text(r3[index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')
    for index, value in enumerate(bars['ArtPainting']):
        plt.text(r4[index] - 0.12,  value + 0.01, float("{0:.2f}".format(value)), fontsize='small')

    plt.xticks([r + bar_width for r in range(4)], ['ArtPainting', 'Cartoon', 'Sketch', 'Photo'])
    plt.xlabel('Target domains', fontweight='bold')
    plt.ylabel('Average similarities from sources', fontweight='bold')
    plt.grid(color='black', linestyle='--', linewidth=0.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, title="Sources")
    plt.show()



def PCA_embedding():
    df = pd.read_csv('./PACS_emb_csv/ArtPainting.csv')
    # Separating target
    df.drop('Target', inplace=True, axis=1)
    # Separating index
    df.drop('Index', inplace=True, axis=1)
    x = df.T

    df1 = pd.read_csv('./PACS_emb_csv/Cartoon.csv')
    # Separating target
    df1.drop('Target', inplace=True, axis=1)
    # Separating index
    df1.drop('Index', inplace=True, axis=1)
    x1 = df1.T

    df2 = pd.read_csv('./PACS_emb_csv/Sketch.csv')
    # Separating target
    df2.drop('Target', inplace=True, axis=1)
    # Separating index
    df2.drop('Index', inplace=True, axis=1)
    x2 = df2.T

    df3 = pd.read_csv('./PACS_emb_csv/Photo.csv')
    # Separating target
    df3.drop('Target', inplace=True, axis=1)
    # Separating index
    df3.drop('Index', inplace=True, axis=1)
    x3 = df3.T

    '''
    # Separating out the features
    # features = [str(it) for it in range(0, 256)]
    # x = df.loc[:, features].values

    # Standardizing the features
    # x = StandardScaler().fit_transform(x)
    df1 = pd.read_csv('./Cartoon.csv')
    # Separating out the features
    x1 = df1.loc[:, features].values
    # Standardizing the features
    # x1 = StandardScaler().fit_transform(x1)

    df2 = pd.read_csv('./Sketch.csv')
    # Separating out the features
    x2 = df2.loc[:, features].values
    # Standardizing the features
    # x2 = StandardScaler().fit_transform(x2)

    df3 = pd.read_csv('./Photo.csv')
    # Separating out the features
    x3 = df3.loc[:, features].values
    # Standardizing the features
    # x3 = StandardScaler().fit_transform(x3)
    '''

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    target = pd.Series(['ArtPainting' for _ in range(0, len(x))])
    finalDf = pd.concat([principalDf, target], axis=1)
    finalDf = finalDf.rename(columns={0: "Target"})
    print(finalDf)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x1)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    target = pd.Series(['Cartoon' for _ in range(0, len(x1))])
    finalDf1 = pd.concat([principalDf, target], axis=1)
    finalDf1 = finalDf1.rename(columns={0: "Target"})
    print(finalDf1)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x2)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    target = pd.Series(['Sketch' for _ in range(0, len(x2))])
    finalDf2 = pd.concat([principalDf, target], axis=1)
    finalDf2 = finalDf2.rename(columns={0: "Target"})
    print(finalDf2)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x3)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])
    target = pd.Series(['Photo' for _ in range(0, len(x3))])
    finalDf3 = pd.concat([principalDf, target], axis=1)
    finalDf3 = finalDf3.rename(columns={0: "Target"})
    print(finalDf3)

    # PCA-3 aggregated and to csv
    pd.concat([finalDf, finalDf1, finalDf2, finalDf3], ignore_index=True).to_csv('./PACS_PCA.csv')

    # plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_zlabel('PC3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=15)
    targets = ['ArtPainting', 'Cartoon', 'Sketch', 'Photo']
    COLORS = ['r', 'g', 'b', 'y']
    ax.scatter(finalDf.loc[:, 'PC1'], finalDf.loc[:, 'PC2'], finalDf.loc[:, 'PC3'], c=COLORS[0], s=2)
    ax.scatter(finalDf1.loc[:, 'PC1'], finalDf1.loc[:, 'PC2'], finalDf1.loc[:, 'PC3'], c=COLORS[1], s=2)
    ax.scatter(finalDf2.loc[:, 'PC1'], finalDf2.loc[:, 'PC2'], finalDf2.loc[:, 'PC3'], c=COLORS[2], s=2)
    ax.scatter(finalDf3.loc[:, 'PC1'], finalDf3.loc[:, 'PC2'], finalDf3.loc[:, 'PC3'], c=COLORS[3], s=2)
    ax.legend(targets)
    ax.grid()
    plt.show()
    

if __name__ == '__main__':
    # Plot the average distances between domains
    print("Average distances from Source to Target domains\n")
    average_distances()
    # plot PCA
    PCA_embedding()
