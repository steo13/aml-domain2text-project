import json
import matplotlib.pyplot as plt

AVERAGE_DISTANCES_FILE = './average_distances.txt'

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


if __name__ == '__main__':
    # Plot the average distances between domains
    print("Average distances from Source to Target domains\n")
    average_distances()

    # PCA(m=2) for each domain
    # embedding for each domain