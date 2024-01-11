import os
import sys
import matplotlib.pyplot as plt
import distinctipy

class Distribution:
    def __init__(self, path_name):
        if not os.path.exists(path_name):
            print("Given path name doesn't exist")
            exit()
        self.path_name = path_name # original path name, sys.argv[1]
        self.categories = set() # set of categories: "Apple", "Grappe"...
        self.subdirectories = set() # set of subdirectories: "Apple_healthy", "Apple_rust"...
        self.images_path = []  # array of every paths found
        self.arborescence = {} # dict with every categories and every subcategories: arborescence["Apple"]["Apple_rust"] returns an array of every paths for Apple_rust
        self.augmentations_needed = [] # array of tuples to fill the gap between highest and every others: (sub_name, number needed)
        self.highest = float("-inf")
        
        self.parse_files()
        self.get_arborescence()
        self.fill_augmentation_needed()


    def fill_augmentation_needed(self):
        for category in self.categories:
            len_name_list = [] # array of tuples: (subdirectory_name, number of images)
            for subdirectory in self.arborescence[category]:
                len_name_list.append((subdirectory, len(self.arborescence[category][subdirectory])))
            
            max_tuple = max(len_name_list, key=lambda x: x[1])
            len_name_list.remove(max_tuple)
            for tup in len_name_list:
                self.augmentations_needed.append((tup[0], max_tuple[1] - tup[1]))
            print(max_tuple[1])


    def parse_files(self):
        for path, subdirs, files in os.walk(self.path_name):
            self.subdirectories.update(subdirs)
            for name in files:
                self.images_path.append(os.path.join(path, name))
        for subdirectory in self.subdirectories:
            self.categories.update([subdirectory.split('_')[0]])


    def get_arborescence(self):
        for category in self.categories:
            self.arborescence[category] = {}
            for subdirectory in self.subdirectories:
                if category.lower() in subdirectory.lower():
                    self.arborescence[category][subdirectory] = []
            if self.highest < len(self.arborescence[category]):
                self.highest = len(self.arborescence[category])
        
        for path in self.images_path:
            subdirectory_name = path.split("/")[path.count("/") - 1] # path.count() to always get the part before the name of the file
            category = subdirectory_name.split("_")[0]
            self.arborescence[category][subdirectory_name].append(path)


def plot_images(distribution):
    color_set = iter(distinctipy.get_colors(distribution.highest * len(distribution.categories)))
    fig, ax = plt.subplots(len(distribution.categories), 2)

    for i, category in enumerate(distribution.categories):
        tuple_test = []
        for sub in distribution.arborescence[category]:
            tuple_test.append((sub, len(distribution.arborescence[category][sub]), next(color_set)))
        tuple_test.sort(key=lambda x: x[1], reverse=True)
        ax[i, 0].pie([x[1] for x in tuple_test], labels=[x[0] for x in tuple_test], autopct='%1.1f%%', colors=[x[2] for x in tuple_test])
        ax[i, 0].set_title(category + " class distribution")
        ax[i, 1].bar([x[0] for x in tuple_test], [x[1] for x in tuple_test], color=[x[2] for x in tuple_test])
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Program needs at least one argument: ./path_to_images")
        return 1
    distribution = Distribution(sys.argv[1])
    plot_images(distribution)


if __name__ == "__main__":
    main()
