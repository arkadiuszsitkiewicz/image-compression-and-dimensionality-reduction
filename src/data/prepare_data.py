import numpy as np
import matplotlib.image as image


class PrepareData:
    DATA_TYPE = ("RAW DATA", "IMAGE", "FACES")

    @classmethod
    def get_vis_type(cls):
        return cls.DATA_TYPE

    def __init__(self, data_path, data_type, depth=3):
        self.__data_path = data_path

        data_type = data_type.upper().strip()
        if data_type not in PrepareData.DATA_TYPE:
            raise ValueError(f"{data_type} is not a valid visualization type")
        elif data_type == "RAW DATA":
            self.converted_data = self.__csv_to_matrix()

        elif data_type == "FACES":
            self.converted_data = np.loadtxt(self.__data_path, delimiter=",")
        else:
            self.converted_data = self.__image_to_matrix(depth)

    def __csv_to_matrix(self):
        return np.loadtxt(open(self.__data_path, "rb"), delimiter=';', dtype="float")

    def __image_to_matrix(self, depth):
        return image.imread(self.__data_path)


