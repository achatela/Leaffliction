import os
import sys
import matplotlib.pyplot as plt
import distinctipy


def main():
    highest = float("-inf")
    subdirectories = set()
    categories = set()
    images_path = []

    for path, subdirs, files in os.walk(sys.argv[1]):
        subdirectories.update(subdirs)
        for name in files:
            images_path.append(os.path.join(path, name))
    for subdirectory in subdirectories:
        categories.update([subdirectory.split('_')[0]])

    arborescence = {}
    for category in categories:
        arborescence[category] = {}
        for subdirectory in subdirectories:
            if category.lower() in subdirectory.lower():
                arborescence[category][subdirectory] = []
        if highest < len(arborescence[category]):
            highest = len(arborescence[category])

    print(highest)
    print(arborescence)
    for path in images_path:
        subdirectory_name = path.split("/")[path.count("/") - 1]
        category = subdirectory_name.split("_")[0]
        arborescence[category][subdirectory_name].append(path)

    color_set = distinctipy.get_colors(highest * len(categories))
    i = 0
    j = 0
    fig, ax = plt.subplots(len(categories), 2)
    for category in categories:
        tuple_test = []
        for sub in arborescence[category]:
            tuple_test.append((sub, len(arborescence[category][sub]), color_set[i]))
            i += 1

        tuple_test.sort(key=lambda x: x[1], reverse=True)

        ax[j, 0].pie([x[1] for x in tuple_test], labels=[x[0] for x in tuple_test], autopct='%1.1f%%', colors=[x[2] for x in tuple_test])
        ax[j, 0].set_title(category + " class distribution")
        ax[j, 1].bar([x[0] for x in tuple_test], [x[1] for x in tuple_test], color=[x[2] for x in tuple_test])
        j += 1
    plt.show()

if __name__ == "__main__":
    main()