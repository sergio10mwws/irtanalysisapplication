# IRT Analysis Application

## Description
A Streamlit-based web application for Item Response Theory (IRT) analysis that supports 1PL, 2PL, and 3PL models. The application allows users to upload response data and analyze it using different IRT models.

## Features
- Support for three IRT models:
  - 1PL (Rasch) Model
  - 2PL Model
  - 3PL Model
- Interactive data upload
- Parameter estimation
- Visual representation of Item Characteristic Curves (ICC)
- Downloadable results
- Ability score calculation (for 1PL model)

## Requirements
```bash
pip install streamlit pandas numpy matplotlib torch pyro-ppl py-irt
```

## Run
```bash
streamlit run app.py 
```
