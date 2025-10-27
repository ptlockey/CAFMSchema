# CAFMSchema

A minimal Streamlit application for viewing a building schematic from a JSON file.

## Getting started

Install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a JSON file that contains a base64 encoded schematic image. Choose **tap** or **shower** from the sidebar, then click on the schematic to place the fixture. Use the zoom slider to magnify the image and the clear button to remove all fixtures.
