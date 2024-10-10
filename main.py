from utils import load_pickle, process_fields
from constants import PICKLE_PATH
from paddleocr import PaddleOCR
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import pandas as pd
import numpy as np

# Set Streamlit page layout to wide mode for better display
st.set_page_config(layout="wide")

# Load form regions configuration from a pickle file
form_regions = load_pickle(PICKLE_PATH)

# Initialize PaddleOCR instance for optical character recognition
if "ocr" not in st.session_state:
    st.session_state.ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize session state attributes for storing images
if "image" not in st.session_state:
    st.session_state.image = None
if "regions" not in st.session_state:
    st.session_state.regions = None
if "box" not in st.session_state:
    st.session_state.box = []

if "image2" not in st.session_state:
    st.session_state.image2 = cv2.imread("test_1.jpeg")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Form Scanner")

    # Create two columns: one for the video feed and one for the UI elements
    col1, col2 = st.columns([3, 5])

    with col1:
        # Set up the camera input widget to capture a single image
        img_file_buffer = st.camera_input("Scan")

        if img_file_buffer is not None:
            # Read image file buffer with OpenCV
            bytes_data = img_file_buffer.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Process the image immediately
            with st.spinner("Processing..."):
                try:
                    st.session_state.image, st.session_state.regions, st.session_state.box = process_fields(
                        frame, form_regions, st.session_state.ocr
                    )
                except:
                    st.session_state.image = None
                    st.session_state.regions = None
                    st.session_state.box = None
                    st.error("Nothing Scanned")
                        
        # Add download button for the Excel file
        if not st.session_state.data.empty:
            # Convert the DataFrame to a CSV string
            csv = st.session_state.data.to_csv(index=False)

            # Add a download button for the CSV file
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="scanned_forms.csv",
                mime="text/csv"
            )

    with col2:                    
        # Display the processed image if available, or show a placeholder message
        if st.session_state.image is not None:
            col5,col6=st.columns([3,1])
            with col5:
                st.image(st.session_state.box,channels="BGR")
            with col6:
                st.write("Choose Options :")
                option1 = st.checkbox("Handwriting")
                option2 = st.checkbox("Spelling")
                option3 = st.checkbox("GK Quiz")
            
            form_data = {}
            for label, data in st.session_state.regions.items():
                st.image(data['roi'], channels="BGR")
                # Display the OCR output in a text input box, allowing the user to edit it
                edited_text = st.text_input(f"Verify OCR Output for {label}", value=data['text'])
                form_data[label] = edited_text
                st.markdown("---")

            form_data["Handwriting"] = option1
            form_data["Spelling"] = option2
            form_data["GK Quiz"] = option3

            save_button_clicked = st.button("Save")
            if save_button_clicked:
                # Append the form data as a new row in the DataFrame
                st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([form_data])], ignore_index=True)
                # Reset session state variables for the next scan
                st.session_state.image = None
                st.session_state.box = []
                st.success("Data saved successfully!")

        else:
            st.header("No frame captured yet.")

if __name__ == "__main__":
    main()
