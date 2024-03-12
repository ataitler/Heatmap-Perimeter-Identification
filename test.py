import sys
import cv2
import numpy as np



def main():
    image = cv2.imread('data/heat_map.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define thresholds for red color RGB
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 170 , 170])

    # Define thresholds for red color HSV
    lower_red = np.array([int(000*255/360), 100, 00])
    upper_red = np.array([int(360*255/360), 250, 255])
    # int(60*255/360)
    # lower_gray = np.array([100])
    # upper_gray = np.array([200])

    # gray_mask = cv2.inRange(image_gray, lower_gray, upper_gray)
    # heat_map = image_gray *gray_mask

    # Create a binary mask of red areas RGB
    mask_red = cv2.inRange(image_rgb, lower_red, upper_red)

    # Create a binary mask of red areas HSV
    mask_red = cv2.inRange(image_hsv, lower_red, upper_red)




    # cv2.imshow('sdfsdf', image_gray*mask_red)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()

    # Find contours of red regions
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(mask_red.shape, np.uint8)
    mask.fill(0)
    for c in contours_red:
        M = cv2.moments(c)
        if M["m00"] < 500:
            continue
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    # cv2.drawContours(mask, contours_red, -1, (255, 255, 255), thickness=cv2.FILLED)

    cv2.imshow('sdfsdf', image_gray * mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()


    lower_green = np.array([0, 150, 0])
    upper_green = np.array([100, 255, 100])
    mask_dark_green = cv2.inRange(image_rgb, lower_green, upper_green)




    # Find contours of dark green dots within the red shape
    contours_dark_green, _ = cv2.findContours(mask_dark_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_dark_green:
        try:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            continue

        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the image
   #     cv2.imshow("Image", image)
    #    cv2.waitKey(0)

    # Combine all points from different dark green regions into a single array
    all_points_dark_green = np.concatenate([contour for contour in contours_dark_green])

    # Compute the convex hull of all dark green points within the red shape
    convex_hull = cv2.convexHull(all_points_dark_green)

    # Create a copy of the original image to preserve the colors
    result_image = image.copy()

    # cv2.imshow('sdfsdf', mask_red)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sys.exit()

    # image = cv2.imread('data/heat_map.jpg')
    # cv2.imshow('sdfsdf', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()