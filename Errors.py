# Custom Errors
class MatrixDimensionError(Exception):
    def __init__(self, mat1, mat2):
        print("%s and %s cannot be multipled" % mat1 % mat2)


class FilterSizeError(Exception):
    def __init__(self):
        print("The filter must be square")


class InputImageError(Exception):
    def __init__(self):
        print("Please use square & grayscale images only")
