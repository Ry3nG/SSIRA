import cv2 as cv

test_img_path = '/home/zerui/SSIRA/dataset/TAD66K/124873707@N0836560919983.jpg'  # Replace with an actual path that failed
try:
    img = cv.imread(test_img_path)
    if img is not None:
        print("Image loaded successfully")
        # Optionally display the image
        # cv.imshow("Test Image", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    else:
        print("Failed to load image")
except Exception as e:
    print(f"Error occurred: {e}")
