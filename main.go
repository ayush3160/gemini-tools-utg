package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"cloud.google.com/go/vertexai/genai"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/api/option"
)

// setupLogger creates a development logger with colored output
func setupLogger() (*zap.Logger, error) {
	config := zap.NewDevelopmentConfig()

	// Enable colored output
	config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	config.EncoderConfig.EncodeCaller = zapcore.ShortCallerEncoder

	// Set development settings
	config.Development = true
	config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)

	return config.Build()
}

// GeminiClient wraps the Vertex AI client for Gemini 2.5 Pro
type GeminiClient struct {
	client *genai.Client
	model  *genai.GenerativeModel
	logger *zap.Logger
}

// NewGeminiClient creates a new Gemini client with service account credentials
func NewGeminiClient(ctx context.Context, projectID, location, credentialsPath string, logger *zap.Logger) (*GeminiClient, error) {
	logger.Info("Creating Vertex AI client",
		zap.String("projectID", projectID),
		zap.String("location", location),
		zap.String("credentialsPath", credentialsPath))

	// Create client with service account credentials
	client, err := genai.NewClient(ctx, projectID, location, option.WithCredentialsFile(credentialsPath))
	if err != nil {
		logger.Error("Failed to create Vertex AI client", zap.Error(err))
		return nil, fmt.Errorf("failed to create Vertex AI client: %w", err)
	}

	logger.Info("Successfully created Vertex AI client")

	// Initialize Gemini 2.5 Pro model
	model := client.GenerativeModel("gemini-2.5-pro")

	// Configure model parameters
	model.SetTemperature(0.7)
	model.SetTopP(0.8)
	model.SetTopK(40)
	model.SetMaxOutputTokens(8192)

	// Setup tools
	tools := setupTools(logger)
	model.Tools = tools

	logger.Info("Configured Gemini 2.5 Pro model",
		zap.Float32("temperature", 0.7),
		zap.Float32("topP", 0.8),
		zap.Int32("topK", 40),
		zap.Int32("maxOutputTokens", 8192),
		zap.Int("toolsCount", len(tools)))

	return &GeminiClient{
		client: client,
		model:  model,
		logger: logger,
	}, nil
}

// GenerateContent sends a prompt to Gemini and returns the response
func (gc *GeminiClient) GenerateContent(ctx context.Context, prompt string) (string, error) {
	gc.logger.Debug("Sending prompt to Gemini", zap.String("prompt", prompt))

	resp, err := gc.model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		gc.logger.Error("Failed to generate content", zap.Error(err))
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 {
		gc.logger.Error("No response candidates returned")
		return "", fmt.Errorf("no response candidates returned")
	}

	// Extract text from the first candidate
	candidate := resp.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		gc.logger.Error("No content in response")
		return "", fmt.Errorf("no content in response")
	}

	// Check if the response contains function calls
	for _, part := range candidate.Content.Parts {
		if funcCall, ok := part.(genai.FunctionCall); ok {
			gc.logger.Info("Function call detected", zap.String("functionName", funcCall.Name))

			// Create directory tool instance
			dirTool := &DirectoryStructureTool{logger: gc.logger}

			// Handle the function call
			result, err := handleFunctionCall(&funcCall, dirTool, gc.logger)
			if err != nil {
				gc.logger.Error("Failed to handle function call", zap.Error(err))
				return "", fmt.Errorf("failed to handle function call: %w", err)
			}

			// Send the function result back to Gemini
			functionResponse := &genai.FunctionResponse{
				Name:     funcCall.Name,
				Response: map[string]any{"result": result},
			}

			// Continue the conversation with the function result and a text prompt
			resp2, err := gc.model.GenerateContent(ctx,
				genai.Text("Please analyze the directory structure data provided by the function call and provide insights about the project."),
				functionResponse)
			if err != nil {
				gc.logger.Error("Failed to generate content after function call", zap.Error(err))
				return "", fmt.Errorf("failed to generate content after function call: %w", err)
			}

			if len(resp2.Candidates) > 0 && resp2.Candidates[0].Content != nil && len(resp2.Candidates[0].Content.Parts) > 0 {
				if textPart, ok := resp2.Candidates[0].Content.Parts[0].(genai.Text); ok {
					response := string(textPart)
					gc.logger.Info("Successfully generated content with function call",
						zap.Int("responseLength", len(response)))
					return response, nil
				}
			}
		}
	}

	// Handle regular text response
	part := candidate.Content.Parts[0]
	if textPart, ok := part.(genai.Text); ok {
		response := string(textPart)
		gc.logger.Info("Successfully generated content",
			zap.Int("responseLength", len(response)))
		return response, nil
	}

	gc.logger.Error("Unexpected content type in response")
	return "", fmt.Errorf("unexpected content type in response")
}

// Close closes the client connection
func (gc *GeminiClient) Close() error {
	return gc.client.Close()
}

// DirectoryStructureTool represents the directory structure tool
type DirectoryStructureTool struct {
	logger *zap.Logger
}

// GetDirectoryStructure returns the directory structure as a string
func (dst *DirectoryStructureTool) GetDirectoryStructure(path string, maxDepth int) (string, error) {
	dst.logger.Debug("Getting directory structure",
		zap.String("path", path),
		zap.Int("maxDepth", maxDepth))

	var result strings.Builder
	err := dst.walkDirectory(path, "", 0, maxDepth, &result)
	if err != nil {
		dst.logger.Error("Failed to get directory structure", zap.Error(err))
		return "", err
	}

	structure := result.String()
	dst.logger.Info("Successfully generated directory structure",
		zap.Int("length", len(structure)))
	return structure, nil
}

// walkDirectory recursively walks through directories
func (dst *DirectoryStructureTool) walkDirectory(path, prefix string, currentDepth, maxDepth int, result *strings.Builder) error {
	if currentDepth > maxDepth {
		return nil
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return err
	}

	for i, entry := range entries {
		// Skip hidden files and directories
		if strings.HasPrefix(entry.Name(), ".") {
			continue
		}

		isLast := i == len(entries)-1
		connector := "├── "
		newPrefix := prefix + "│   "

		if isLast {
			connector = "└── "
			newPrefix = prefix + "    "
		}

		result.WriteString(prefix + connector + entry.Name())

		if entry.IsDir() {
			result.WriteString("/\n")
			subPath := filepath.Join(path, entry.Name())
			err := dst.walkDirectory(subPath, newPrefix, currentDepth+1, maxDepth, result)
			if err != nil {
				dst.logger.Warn("Failed to read subdirectory",
					zap.String("path", subPath),
					zap.Error(err))
			}
		} else {
			result.WriteString("\n")
		}
	}

	return nil
}

// setupTools configures the tools for Gemini
func setupTools(logger *zap.Logger) []*genai.Tool {
	// Define the directory structure tool
	directoryStructureTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "get_directory_structure",
				Description: "Get the directory structure of a given path up to a specified depth",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"path": {
							Type:        genai.TypeString,
							Description: "The directory path to analyze",
						},
						"max_depth": {
							Type:        genai.TypeInteger,
							Description: "Maximum depth to traverse (default: 3)",
						},
					},
					Required: []string{"path"},
				},
			},
		},
	}

	logger.Info("Configured tools for Gemini",
		zap.Int("toolCount", len([]*genai.Tool{directoryStructureTool})))

	return []*genai.Tool{directoryStructureTool}
}

// handleFunctionCall processes function calls from Gemini
func handleFunctionCall(call *genai.FunctionCall, dirTool *DirectoryStructureTool, logger *zap.Logger) (string, error) {
	logger.Debug("Handling function call",
		zap.String("functionName", call.Name))

	switch call.Name {
	case "get_directory_structure":
		// Extract parameters
		pathParam, ok := call.Args["path"].(string)
		if !ok {
			return "", fmt.Errorf("path parameter is required and must be a string")
		}

		maxDepth := 3 // default
		if depthParam, exists := call.Args["max_depth"]; exists {
			if depth, ok := depthParam.(float64); ok {
				maxDepth = int(depth)
			}
		}

		// Call the tool
		structure, err := dirTool.GetDirectoryStructure(pathParam, maxDepth)
		if err != nil {
			return "", fmt.Errorf("failed to get directory structure: %w", err)
		}

		logger.Info("Function call executed successfully",
			zap.String("functionName", call.Name),
			zap.String("path", pathParam),
			zap.Int("maxDepth", maxDepth))

		return structure, nil

	default:
		return "", fmt.Errorf("unknown function: %s", call.Name)
	}
}

func main() {
	ctx := context.Background()

	// Setup logger
	logger, err := setupLogger()
	if err != nil {
		log.Fatalf("Failed to setup logger: %v", err)
	}
	defer logger.Sync() // flushes buffer, if any

	logger.Info("Starting Gemini client application")

	// Configuration - you can set these as environment variables or pass as arguments
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	location := "us-central1" // or your preferred location
	credentialsPath := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")

	// Validate required environment variables
	if projectID == "" {
		logger.Fatal("GOOGLE_CLOUD_PROJECT environment variable is required")
	}
	if credentialsPath == "" {
		logger.Fatal("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
	}

	// Create Gemini client
	geminiClient, err := NewGeminiClient(ctx, projectID, location, credentialsPath, logger)
	if err != nil {
		logger.Fatal("Failed to create Gemini client", zap.Error(err))
	}
	defer geminiClient.Close()

	// Example usage
	prompt := "Can you analyze the directory structure of the current working directory? Use the get_directory_structure tool to examine the project structure and tell me what type of project this appears to be."

	logger.Info("Sending prompt to Gemini 2.5 Pro", zap.String("prompt", prompt))
	response, err := geminiClient.GenerateContent(ctx, prompt)
	if err != nil {
		logger.Fatal("Failed to generate content", zap.Error(err))
	}

	logger.Info("Gemini Response",
		zap.String("prompt", prompt),
		zap.String("response", response))
	logger.Info("Application completed successfully")
}
