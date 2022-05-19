import cv2
import random
import numpy


class Stack:
    def __init__(self):
        self.stack = []

    def pop(self):
        if len(self.stack) < 1:
            return None
        return self.stack.pop()

    def push(self, item):
        self.stack.append(item)

    def size(self):
        return len(self.stack)


def labelRegions(inputImage):
    label = 2  # labels 0 & 1 are reserved by background & foreground

    # convert the input grayscale image back to BGR
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_GRAY2RGB)

    imageHeight, imageWidth, imageChannels = inputImage.shape

    # pixels are accessed via (y, x) and not (x, y)... wack
    for v in range(imageHeight):
        for u in range(imageWidth):
            try:
                # if white...
                if (
                    inputImage.item(v, u, 0) == 255
                    and inputImage.item(v, u, 1) == 255
                    and inputImage.item(v, u, 2) == 255
                ):
                    floodFill(inputImage, v, u, label)
                    label += 1
            except Exception as e:
                print("Error while labeling regions: " + str(e))

    print("counted regions: " + str(label - 2))


def floodFill(inputImage, v, u, label):
    imageHeight, imageWidth, imageChannels = inputImage.shape

    stack = Stack()
    stack.push((v, u))

    # print("generating random color for region pixel " + str(v) + ", " + str(u))
    randomChannel, randomColor = (random.randint(0, 2), random.randint(1, 255))

    while stack.size() > 0:
        currentY, currentX = stack.pop()

        try:
            # if within bounds, and pure white:
            if (
                (currentY < imageHeight)
                and (currentX < imageWidth)
                and (inputImage.item(currentY, currentX, 0) == 255)
                and (inputImage.item(currentY, currentX, 1) == 255)
                and (inputImage.item(currentY, currentX, 2) == 255)
            ):
                # change the randomly-decided color channel of the region
                # to a randomly-decided value
                inputImage.itemset((currentY, currentX, randomChannel), randomColor)

                # push neighboring pixels onto the stack
                stack.push((currentY + 1, currentX))
                stack.push((currentY - 1, currentX))
                stack.push((currentY, currentX + 1))
                stack.push((currentY, currentX - 1))
        except Exception as e:
            print("Error during floodfill: " + str(e))

    cv2.imshow("Flood-filled image", inputImage)


def main(inputFileName: str):

    # attempt to read in the specified image file
    try:
        inputImage = cv2.imread(inputFileName, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print("Error while reading image: " + str(e))
        exit()

    # median filtering
    medianblurr = cv2.medianBlur(inputImage, 5)

    # commit thresholding
    ret, threshImg = cv2.threshold(medianblurr, 40, 255, cv2.THRESH_BINARY)

    # commit gaussian filter
    gaussianblur = cv2.GaussianBlur(threshImg, (5, 5), 0)
    gaussianblur2 = cv2.GaussianBlur(gaussianblur, (5, 5), 0)
    gaussianblur3 = cv2.GaussianBlur(gaussianblur2, (5, 5), 0)
    gaussianblur4 = cv2.GaussianBlur(gaussianblur3, (5, 5), 0)
    gaussianblur5 = cv2.GaussianBlur(gaussianblur4, (5, 5), 0)

    # sharpening
    kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpened = cv2.filter2D(gaussianblur4, -10, kernel)

    # label and count regions
    labelRegions(sharpened)

    # consolidate all pipline images for single-window viewing
    pipelineImgs = numpy.hstack((inputImage, threshImg, sharpened))

    cv2.imshow("Pipeline Images", pipelineImgs)

    cv2.waitKey(0)


# input image filename is read through stdin
main(input())
