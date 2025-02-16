###### Explore-Classification-Project

## Running the Application

To run this application using the provided virtual environment:

1. Clone the repository
2. Navigate to the project directory
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Run the application: `streamlit run Classification_App.py`

Note: The virtual environment contains all required dependencies. No additional installation is needed.

### Troubleshooting
- If you see "Activate.ps1 is not digitally signed" on Windows PowerShell:
  - Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
  - Then try activating the environment again
- If the environment fails to activate, ensure you're using a compatible Python version (Python 3.8+ recommended)