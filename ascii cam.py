import cv2
import numpy as np

ASCII_CHARS = "@%#*+=-:. "

def resize_image(image, density=1):
    """
    Resize image based on pixel density.

    🔧 density = number of pixels per ASCII character.
        - Lower density (1 or 2) = smaller chars, more detail
        - Higher density (4, 5, 6+) = chunkier ASCII blocks
    """
    h, w = image.shape
    new_width = max(10, w // density)
    new_height = max(10, int(h / density * 0.55))
    resized = cv2.resize(image, (new_width, new_height))
    return resized

def to_ascii(image):
    pixels = image.flatten()
    return "".join([ASCII_CHARS[pixel // 25] for pixel in pixels])

def ascii_to_image(ascii_str, width, height, font_scale=0.5, 
                   font=cv2.FONT_HERSHEY_SIMPLEX, 
                   text_color=(0, 255, 0), bg_color=(0, 0, 0),
                   color_frame=None):
    img = np.full((height*12, width*8, 3), bg_color, dtype=np.uint8)
    i = 0
    for y in range(height):
        for x in range(width):
            char = ascii_str[i]
            if color_frame is not None:
                text_color = tuple(int(c) for c in color_frame[y, x])
            cv2.putText(img, char, (x*8, y*12), font, font_scale, text_color, 1, cv2.LINE_AA)
            i += 1
    return img

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Could not open webcam.")
        return

    cv2.namedWindow("ASCII Webcam", cv2.WINDOW_NORMAL)

    # 🎛 Adjustable variable
    pixel_density = 3  #  Increase = chunkier ASCII, Decrease = finer detail
    use_color_ascii = True  #  True = color ASCII, False = monochrome

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize based on density
            resized = resize_image(gray, density=pixel_density)
            ascii_str = to_ascii(resized)

            if use_color_ascii:
                color_small = cv2.resize(frame, (resized.shape[1], resized.shape[0]))
                ascii_img = ascii_to_image(ascii_str, resized.shape[1], resized.shape[0], 
                                           color_frame=color_small)
            else:
                ascii_img = ascii_to_image(ascii_str, resized.shape[1], resized.shape[0], 
                                           text_color=(255,255,255), bg_color=(0,0,0))

            cv2.imshow("ASCII Webcam", ascii_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
