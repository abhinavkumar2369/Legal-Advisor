# API Key Setup Guide

## Getting Your Gemini API Key

1. **Go to Google AI Studio:**
   - Visit: https://makersuite.google.com/app/apikey
   - Sign in with your Google account

2. **Create a new API key:**
   - Click "Create API Key"
   - Choose "Create API key in new project" or select an existing project
   - Copy the generated API key

## Setting Up Environment Variables

### Method 1: Using .env file (Recommended for local development)

1. **Create a .env file:**
   ```bash
   # Copy the example file
   cp .env.example .env
   ```

2. **Edit the .env file:**
   ```bash
   # Open .env file and replace with your actual API key
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **The app will automatically load the API key from .env file**

### Method 2: System Environment Variables

#### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```

#### Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=your_actual_api_key_here
```

#### Linux/Mac:
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

### Method 3: Streamlit Secrets (For Streamlit Cloud deployment)

1. **Create .streamlit/secrets.toml:**
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```

2. **Update the code to use secrets:**
   ```python
   api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
   ```

### Method 4: Manual Entry in App (Fallback)

If no environment variable is found, the app will show an input field in the sidebar where you can enter your API key manually.

## Security Best Practices

1. **Never commit API keys to version control**
   - Add .env to .gitignore
   - Never hardcode API keys in your source code

2. **Use different API keys for different environments**
   - Development, staging, and production should have separate keys

3. **Regularly rotate your API keys**
   - Generate new keys periodically for security

4. **Monitor API usage**
   - Check your Google Cloud Console for API usage and costs

## Troubleshooting

### "API key not found" error:
- Verify the API key is correctly set in your environment
- Check for typos in the environment variable name
- Restart your terminal/IDE after setting environment variables

### "Invalid API key" error:
- Verify the API key is correct
- Check if the API key has the necessary permissions
- Ensure you've enabled the Gemini API in Google Cloud Console

### "Quota exceeded" error:
- Check your API usage in Google Cloud Console
- Upgrade your plan if needed
- Wait for the quota to reset (if using free tier)
