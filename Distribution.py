import os
import sys
import matplotlib.pyplot as plt


def main():
    
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

    print(arborescence)
    for path in images_path:
        subdirectory_name = path.split("/")[path.count("/") - 1]
        category = subdirectory_name.split("_")[0]
        arborescence[category][subdirectory_name].append(path)

    for category in categories:
        tuple_test = []
        for sub in arborescence[category]:
            tuple_test.append((sub, len(arborescence[category][sub])))

        tuple_test.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots()
        ax.pie([x[1] for x in tuple_test], labels=[x[0] for x in tuple_test], autopct='%1.1f%%')
        plt.title(category + " class distribution")
        plt.show()
        plt.bar([x[0] for x in tuple_test], [x[1] for x in tuple_test])
        plt.show()

if __name__ == "__main__":
    main()