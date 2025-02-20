import cv2
import numpy as np
import streamlit as st

def get_doc_rotation_angle(img, block=21, c=5):
    # prepare an image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        c,
    )
    
    # get edges and lines
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=200,
        maxLineGap=10
    )
    
    if lines is None:
        return None

    # get angles for all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    m_angle = np.median(angles)
    
    # Корректируем угол, чтобы он был в диапазоне [-45, 45]
    if m_angle > 45:
        m_angle = 90 - m_angle
    elif m_angle < -45:
        m_angle = -90 - m_angle
    
    return m_angle

def main():
    st.title("Get Document Rotation")
    st.write("Upload a photo of a document, and see if a document needs to be rotated.")

    uploaded_file = st.file_uploader(
      "Upload an image..",
      type=["jpg", "jpeg", "png"],
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, width=400, channels="BGR", caption="Image of a document", use_container_width=True)
        angle = get_doc_rotation_angle(image, 11, 20)
        
        if angle is not None:
            if angle > 3 or angle < -3:
                st.write(f"Document rotation angle: {angle} degrees. Needs to be rotated back.")
            else:
                st.write(f"Document rotation angle: {angle} degrees. Document is OK!")
        else:
            st.write(f"No contours found")

if __name__ == "__main__":
    main()
