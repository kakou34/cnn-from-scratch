from keras.datasets import mnist

from CNN import CNN
from Errors import MatrixDimensionError, FilterSizeError, InputImageError


def prepare_data():
    # Loading the MNIST dataset from keras
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Getting the first 1000 data for training
    trainX = trainX[:1000, :, :]
    trainY = trainY[:1000]
    # Getting the first 300 data for testing
    testX = trainX[:300, :, :]
    testY = trainY[:300]

    # convert from integers to floats
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    # normalize inputs to range 0-1
    trainX = trainX / 255.0
    testX = testX / 255.0

    return trainX, trainY, testX, testY


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        # loading the data
        trainX, trainY, testX, testY = prepare_data()
        # creating the CNN instance
        cnn = CNN(0.001, 28, 10, 5, 3, 2, 2)
        # Training the CNN
        cnn.train(trainX, trainY, 10)
        # Testing the cnn
        acc = cnn.test(testX, testY) * 100
        print("Accuracy: %.2f %%" % acc)

    except InputImageError as ie:
        print(ie)
    except FilterSizeError as fe:
        print(fe)
    except MatrixDimensionError as me:
        print(me)
    except:
        print('An unexcpected error happened')
