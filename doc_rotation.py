import cv2
import numpy as np
import streamlit as st


def get_doc_rotation_angle(image):
    # prepare an image
    gray = cv2.cvtColor(
      image,
      cv2.COLOR_BGR2GRAY
    )
    thresh = cv2.adaptiveThreshold(
      gray,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY_INV,
      11,
      2
    )
  
    # get countures
    cntrs, _ = cv2.findContours(
      thresh,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE
    )
    if not cntrs:
        return None
    largest_cntr = max(cntrs, key=cv2.contourArea)
    boundary = cv2.minAreaRect(largest_cntr)

    # get angle
    angle = boundary[-1]
    if angle > 45:
        angle = 90 - angle
    elif angle < -45:
        angle = -90 - angle
    return angle

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
        
        st.image(image, channels="BGR", caption="Image of a document", use_column_width=True)
        angle = get_doc_rotation_angle(image)
        
        if angle is not None:
            if angle > 3 or angle < -3:
                st.write(f"Document rotation angle: {angle} degrees. Needs to be rotated back.")
            else:
                st.write(f"Document rotation angle: {angle} degrees. Document is OK!")
        else:
            st.write(f"No contours found")

if __name__ == "__main__":
    main()
