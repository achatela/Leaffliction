import os
import sys
import matplotlib.pyplot as plt
import distinctipy


class Distribution:
    def __init__(self, path_name):
        if not os.path.exists(path_name):
            sys.exit("Given path name doesn't exist")
        self.path_name = path_name
        self.categories = set()
        self.subdirectories = set()
        self.images_path = []
        self.arborescence = {}
        self.augmentations_needed = []
        self.highest = float("-inf")

        self.parse_files()
        self.get_arborescence()
        self.fill_augmentation_needed()

    def fill_augmentation_needed(self):
        try:
            max_len_dir = 0
            for category in self.categories:
                for subdirectory in self.arborescence[category]:
                    if len(
                        self.arborescence[category][subdirectory]
                    ) > max_len_dir:
                        max_len_dir = len(
                            self.arborescence[category][subdirectory]
                        )

            for category in self.categories:
                len_name_list = []
                for subdirectory in self.arborescence[category]:
                    len_name_list.append((
                        subdirectory,
                        len(self.arborescence[category][subdirectory])
                    ))

                for tup in len_name_list:
                    self.augmentations_needed.append((
                        tup[0],
                        max_len_dir - tup[1]
                    ))
        except Exception as e:
            print("Error while filling augmentations needed: " + str(e))
            exit()

    def parse_files(self):
        try:
            for path, subdirs, files in os.walk(self.path_name):
                self.subdirectories.update(subdirs)
                for name in files:
                    self.images_path.append(os.path.join(path, name))
            for subdirectory in self.subdirectories:
                self.categories.update([subdirectory.split('_')[0]])
        except Exception as e:
            print("Error while parsing files: " + str(e))
            exit()

    def get_arborescence(self):
        try:
            for category in self.categories:
                self.arborescence[category] = {}
                for subdirectory in self.subdirectories:
                    if category.lower() in subdirectory.lower():
                        self.arborescence[category][subdirectory] = []
                if self.highest < len(self.arborescence[category]):
                    self.highest = len(self.arborescence[category])

            for path in self.images_path:
                subdirectory_name = path.split("/")[path.count("/") - 1]
                category = subdirectory_name.split("_")[0]
                self.arborescence[category][subdirectory_name].append(path)
        except Exception as e:
            print("Error while getting arborescence: " + str(e))
            exit()


def plot_images(distribution):
    try:
        color_set = iter(distinctipy.get_colors(
            distribution.highest * len(distribution.categories)
        ))
        fig, ax = plt.subplots(len(distribution.categories), 2, squeeze=False)

        for i, category in enumerate(distribution.categories):
            tuple_test = []
            for sub in distribution.arborescence[category]:
                tuple_test.append((
                    sub,
                    len(distribution.arborescence[category][sub]),
                    next(color_set)
                ))
            tuple_test.sort(key=lambda x: x[1], reverse=True)
            ax[i, 0].pie(
                [x[1] for x in tuple_test],
                labels=[x[0] for x in tuple_test], autopct='%1.1f%%',
                colors=[x[2] for x in tuple_test]
            )
            ax[i, 0].set_title(category + " class distribution")
            ax[i, 1].bar(
                [x[0] for x in tuple_test],
                [x[1] for x in tuple_test],
                color=[x[2] for x in tuple_test]
            )
        plt.show()
    except Exception as e:
        print("Error while plotting images: " + str(e))
        exit()


def main():
    if len(sys.argv) < 2:
        print("Program needs at least one argument: ./path_to_images")
        return 1
    distribution = Distribution(sys.argv[1])
    plot_images(distribution)


if __name__ == "__main__":
    main()
