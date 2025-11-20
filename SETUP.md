# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    - On Windows:
        ```powershell
        .\venv\Scripts\Activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Navigate to the `src` directory:
    ```bash
    cd src
    ```

2.  Run the Flask application:
    ```bash
    python app.py
    ```

3.  Open your web browser and go to `http://127.0.0.1:5000`.

## Usage
- Click on "Choose File" to upload an anime image.
- Click "Classify" to see the predicted anime title.
- See predictions, anime information, and link to MyAnimeList entry.