# Gemini 2.5 Pro with Vertex AI and Function Calling

This project demonstrates how to set up and use Google's Gemini 2.5 Pro model through Vertex AI with service account authentication and function calling capabilities.

## Features

- **Gemini 2.5 Pro Integration**: Complete setup with Vertex AI
- **Service Account Authentication**: Secure authentication using JSON credentials
- **Function Calling**: Custom tools that Gemini can call
- **Directory Structure Tool**: Built-in tool to analyze project directory structures
- **Structured Logging**: Zap logger with colored development output
- **Error Handling**: Comprehensive error handling and logging

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with Vertex AI API enabled
2. **Service Account**: Create a service account with the necessary permissions
3. **Credentials JSON**: Download the service account credentials JSON file

## Setup Instructions

### 1. Enable Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com
```

### 2. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create gemini-service-account \
    --description="Service account for Gemini API access" \
    --display-name="Gemini Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:gemini-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download credentials
gcloud iam service-accounts keys create ~/gemini-credentials.json \
    --iam-account=gemini-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3. Set Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Usage

### Running the Application

```bash
# Install dependencies
go mod download

# Run the application
go run main.go
```

### Code Structure

- `GeminiClient`: A wrapper struct that encapsulates the Vertex AI client and Gemini model
- `NewGeminiClient()`: Creates a new client with service account authentication and tool configuration
- `GenerateContent()`: Sends prompts to Gemini, handles function calls automatically
- `DirectoryStructureTool`: Custom tool for analyzing directory structures
- `setupLogger()`: Creates a development logger with colored output

## Function Calling

The application includes a built-in directory structure analysis tool that Gemini can call:

### Available Tools

1. **get_directory_structure**
   - **Description**: Get the directory structure of a given path up to a specified depth
   - **Parameters**:
     - `path` (required): The directory path to analyze
     - `max_depth` (optional): Maximum depth to traverse (default: 3)

### Example Usage

```go
ctx := context.Background()

// Create client with logger
logger, _ := setupLogger()
geminiClient, err := NewGeminiClient(ctx, projectID, location, credentialsPath, logger)
if err != nil {
    log.Fatal(err)
}
defer geminiClient.Close()

// Ask Gemini to analyze directory structure
prompt := "Can you analyze the directory structure of the current working directory? Use the get_directory_structure tool to examine the project structure."
response, err := geminiClient.GenerateContent(ctx, prompt)
if err != nil {
    log.Fatal(err)
}

fmt.Println(response)
```

## Configuration

The application uses the following configuration:

- **Model**: `gemini-2.5-pro`
- **Location**: `us-central1` (configurable)
- **Temperature**: `0.7`
- **Top P**: `0.8`
- **Top K**: `40`
- **Max Output Tokens**: `8192`

### Logger Configuration

- **Development Mode**: Enabled with colored output
- **Log Level**: Debug (shows all logs)
- **Time Format**: ISO8601
- **Caller Info**: Short caller format

## Function Call Flow

1. User sends a prompt to Gemini
2. Gemini decides if it needs to call a function
3. If a function call is needed, the application:
   - Detects the function call in the response
   - Executes the appropriate tool function
   - Sends the result back to Gemini
   - Returns Gemini's final response incorporating the tool result

## Error Handling

The application includes comprehensive error handling for:
- Client initialization failures
- Authentication errors
- API call failures
- Function call execution errors
- Empty or malformed responses

## Security Notes

- Keep your credentials JSON file secure and never commit it to version control
- Use environment variables for sensitive configuration
- Consider using Google Cloud Secret Manager for production deployments
- Regularly rotate your service account keys

## Dependencies

- `cloud.google.com/go/vertexai`: Official Google Cloud Vertex AI SDK
- `google.golang.org/api`: Google API client library for authentication options
- `go.uber.org/zap`: High-performance, structured logging library
