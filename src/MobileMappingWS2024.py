from os import environ

import streamlit as st

st.set_page_config(
    page_title="Mobile Mapping WS2024",
    page_icon="🗺️",
    initial_sidebar_state="expanded",
)
st.sidebar.success("Select a chapter above.")

st.markdown(
    "[![GitHub](https://img.shields.io/badge/github-%2523121011.svg?style=for-the-badge&logo=github&color=AB00AB)](https://github.com/punsii2/MobileMappingWS2024)"
)

st.markdown(
    """
   <===== Pick an exercise page in the sidebar
"""
)

with open(environ["SOURCE_ZIP_PATH"], "rb") as f:
    st.download_button(
        "Download source code as zip", f, file_name="MobileMappingWs24Menhart.zip"
    )


#
# picture = st.camera_input("Take a picture")
#
# if picture:
#     image = cv2.cvtColor(cv2.imread(str(KARLSTR_PATH)), cv2.COLOR_RGB2BGR)
#     st.image(picture)
