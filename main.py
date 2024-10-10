import streamlit as st
import pandas as pd
from utils import load_pickle, process_fields, load_image
from constants import PICKLE_PATH
from paddleocr import PaddleOCR

# Set Streamlit page layout to wide mode for better display
st.set_page_config(layout="wide")

# Load form regions configuration from a pickle file
form_regions = load_pickle(PICKLE_PATH)

# Initialize PaddleOCR instance for optical character recognition
if "ocr" not in st.session_state:
    st.session_state.ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize session state attributes for storing images and files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "current_file_index" not in st.session_state:
    st.session_state.current_file_index = 0

if "image" not in st.session_state:
    st.session_state.image = None

if "regions" not in st.session_state:
    st.session_state.regions = None

if "box" not in st.session_state:
    st.session_state.box = []

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Form Scanner")

    with st.sidebar:
        # File uploader to allow multiple image uploads
        uploaded_files = st.file_uploader(
            "Upload form images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        else:
            st.session_state.uploaded_files = None
            st.session_state.current_file_index = 0

    if st.session_state.uploaded_files:
        col3, col4 = st.columns([1, 3])

        with col3:
            prev, next = st.columns([1, 1])
            # Initialize session state for navigation buttons if not already set
            if "next_button_clicked" not in st.session_state:
                st.session_state.next_button_clicked = False
            if "prev_button_clicked" not in st.session_state:
                st.session_state.prev_button_clicked = False

            with prev:
                # Prev button
                if st.button("Prev"):
                    st.session_state.prev_button_clicked = True
                    st.session_state.next_button_clicked = False

            with next:
                # Next button
                if st.button("Next"):
                    st.session_state.next_button_clicked = True
                    st.session_state.prev_button_clicked = False

            # Update the file index based on button clicks
            if st.session_state.next_button_clicked and st.session_state.current_file_index < len(st.session_state.uploaded_files) - 1:
                st.session_state.current_file_index += 1
                st.session_state.next_button_clicked = False  # Reset the state after updating

            if st.session_state.prev_button_clicked and st.session_state.current_file_index > 0:
                st.session_state.current_file_index -= 1
                st.session_state.prev_button_clicked = False  # Reset the state after updating

            if st.session_state.current_file_index == len(st.session_state.uploaded_files):
                current_file = None
                st.write("No more images to display")
            else:
                # Get the current image to display based on the file index
                current_file = st.session_state.uploaded_files[st.session_state.current_file_index]
                current_image = load_image(current_file)

            # Display the current image with caption
            st.image(image=current_image, caption=f"Current Image: {current_file.name}", use_column_width='always')

        with col4:
            # Add form within the column for OCR processing and saving the result
            with st.form("ocr_form"):
                # Add a button to trigger the scan action
                scan_button_clicked = st.form_submit_button("Scan")

                if scan_button_clicked:
                    # Display a spinner while processing the image
                    with st.spinner("Processing..."):
                        try:
                            # Process the current image using the pre-loaded OCR model and form regions
                            st.session_state.image, st.session_state.regions, st.session_state.box = process_fields(
                                current_image, form_regions, st.session_state.ocr)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            st.session_state.image = None
                            st.session_state.regions = None
                            st.session_state.box = None

                if st.session_state.image is not None:
                    st.image(st.session_state.box, channels="BGR")

                    st.write("Choose Options:")
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

                    # Store options in the form data
                    form_data["Handwriting"] = option1
                    form_data["Spelling"] = option2
                    form_data["GK Quiz"] = option3

                    # Form submission button to save data
                    submit_form = st.form_submit_button("Save")

                    if submit_form:
                        # Append the form data as a new row in the DataFrame
                        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([form_data])], ignore_index=True)
                        # Reset session state variables for the next scan
                        st.session_state.image = None
                        st.session_state.box = []
                        st.success("Data saved successfully!")
                else:
                    st.header("Nothing scanned yet!")
            # Make sure the download button is placed immediately after saving data
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

            # Check if any data is present and render the download button outside the form, ensuring it appears
            if not st.session_state.data.empty:
                csv = st.session_state.data.to_csv(index=False)

                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="scanned_forms.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
