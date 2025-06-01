# AI Document Translator

A Streamlit application that uses AI to translate documents between different languages. Supports PDF, DOCX, TXT, and SRT file formats.

## Features

- Supports multiple document formats (PDF, DOCX, TXT, SRT)
- Compatible with OpenAI API and compatible endpoints
- Automatic model detection and selection
- Preserves formatting and style during translation
- Download translated documents
- Simple and intuitive user interface

## Local Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   # Optional: If using a different API endpoint
   # OPENAI_API_BASE=https://your-custom-endpoint.com/v1
   ```

## Local Usage

1. Run the application:
   ```
   streamlit run app.py
   ```

2. The application will automatically load your API key from the `.env` file
3. In the main interface:
   - Upload a document (PDF, DOCX, TXT, or SRT)
   - Select source and target languages
   - Click "Translate"
   - Download the translated text when complete

## Deployment to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select your repository
4. Set the following settings:
   - Branch: `main` (or your preferred branch)
   - Main file path: `app.py`
5. Click "Advanced settings" and add your secrets:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - (Optional) `OPENAI_API_BASE`: Your custom API endpoint if not using OpenAI
6. Click "Deploy!"

## Environment Variables

- `OPENAI_API_KEY`: (Required) Your OpenAI API key
- `OPENAI_API_BASE`: (Optional) Custom API endpoint (default: https://api.openai.com/v1)

## Supported Languages

- English
- Slovenian
- German
- French
- Spanish
- Italian
- Croatian
- Serbian
- Russian
- Chinese
- Japanese
- Korean

## Requirements

- Python 3.8+
- Streamlit
- OpenAI Python client
- PyPDF2
- python-docx
