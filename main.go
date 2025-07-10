package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gemini-tool/protocol"

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
	const geminiPro25MaxTokens = 65535
	model.SetTemperature(0.7)
	model.SetTopP(0.8)
	model.SetTopK(40)
	model.SetMaxOutputTokens(int32(geminiPro25MaxTokens))

	// Setup tools
	tools := setupTools(logger)
	model.Tools = tools

	logger.Info("Configured Gemini 2.5 Pro model",
		zap.Float32("temperature", 0.7),
		zap.Float32("topP", 0.8),
		zap.Int32("topK", 40),
		zap.Int32("maxOutputTokens", geminiPro25MaxTokens),
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

	// Store the full response to a file for debugging
	err = gc.storeResponseToFile(resp, "gemini_response.txt")
	if err != nil {
		gc.logger.Warn("Failed to store response to file", zap.Error(err))
	}

	if len(resp.Candidates) == 0 {
		gc.logger.Error("No response candidates returned")
		err = gc.storeDebugInfo(resp, "no_candidates_debug.txt")
		if err != nil {
			gc.logger.Warn("Failed to store debug info", zap.Error(err))
		}
		return "", fmt.Errorf("no response candidates returned")
	}

	// Extract text from the first candidate
	candidate := resp.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		gc.logger.Error("No content in response")
		err = gc.storeDebugInfo(resp, "no_content_debug.txt")
		if err != nil {
			gc.logger.Warn("Failed to store debug info", zap.Error(err))
		}
		return "", fmt.Errorf("no content in response")
	}

	// Check if the response contains function calls
	for _, part := range candidate.Content.Parts {
		if funcCall, ok := part.(genai.FunctionCall); ok {
			gc.logger.Info("Function call detected", zap.String("functionName", funcCall.Name))

			// Create tool instances
			dirTool := &DirectoryStructureTool{logger: gc.logger}
			goplsTool, err := NewGoplsTool(gc.logger)
			if err != nil {
				gc.logger.Error("Failed to create gopls tool", zap.Error(err))
				return "", fmt.Errorf("failed to create gopls tool: %w", err)
			}
			defer goplsTool.Close()

			// Handle the function call
			result, err := handleFunctionCall(&funcCall, dirTool, goplsTool, gc.logger)
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
			var followUpPrompt string
			switch funcCall.Name {
			case "analyze_go_code":
				if actionParam, ok := funcCall.Args["action"].(string); ok {
					switch actionParam {
					case "code_definitions":
						followUpPrompt = "Based on the code definitions provided, please analyze the code structure and help with generating appropriate unit tests."
					default:
						followUpPrompt = "Please analyze the Go code data provided by the function call and provide relevant insights."
					}
				} else {
					followUpPrompt = "Please analyze the Go code data provided by the function call and provide relevant insights."
				}
			case "get_code_definitions":
				followUpPrompt = "Based on the code definitions provided, please analyze the code structure and help with generating appropriate unit tests."
			default:
				followUpPrompt = "Please analyze the data provided by the function call and provide relevant insights."
			}

			resp2, err := gc.model.GenerateContent(ctx,
				genai.Text(followUpPrompt),
				functionResponse)
			if err != nil {
				gc.logger.Error("Failed to generate content after function call", zap.Error(err))
				return "", fmt.Errorf("failed to generate content after function call: %w", err)
			}

			gc.logger.Info("Made second AI call with function response",
				zap.String("followUpPrompt", followUpPrompt),
				zap.String("functionName", funcCall.Name))

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
	// Define a single comprehensive code analysis tool
	codeAnalysisTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "analyze_go_code",
				Description: "Analyze Go code projects - get code definitions for symbols using gopls",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"action": {
							Type:        genai.TypeString,
							Description: "Action to perform: 'code_definitions'",
							Enum:        []string{"code_definitions"}, // Only code_definitions for now
						},
						"path": {
							Type:        genai.TypeString,
							Description: "The file path to analyze",
						},
						"symbols": {
							Type: genai.TypeArray,
							Items: &genai.Schema{
								Type: genai.TypeString,
							},
							Description: "List of symbol names to look up for code definitions (function names, struct names, etc.)",
						},
					},
					Required: []string{"action", "path", "symbols"},
				},
			},
		},
	}

	tools := []*genai.Tool{codeAnalysisTool}
	logger.Info("Configured tools for Gemini", zap.Int("toolCount", len(tools)))

	return tools
}

// handleFunctionCall processes function calls from Gemini
func handleFunctionCall(call *genai.FunctionCall, dirTool *DirectoryStructureTool, goplsTool *GoplsTool, logger *zap.Logger) (string, error) {
	logger.Debug("Handling function call",
		zap.String("functionName", call.Name))

	switch call.Name {
	case "analyze_go_code":
		// Extract action parameter
		actionParam, ok := call.Args["action"].(string)
		if !ok {
			return "", fmt.Errorf("action parameter is required and must be a string")
		}

		// Extract path parameter
		pathParam, ok := call.Args["path"].(string)
		if !ok {
			return "", fmt.Errorf("path parameter is required and must be a string")
		}

		switch actionParam {
		case "code_definitions":
			// Extract symbols parameter
			symbolsParam, ok := call.Args["symbols"].([]interface{})
			if !ok {
				return "", fmt.Errorf("symbols parameter is required and must be an array for code_definitions action")
			}

			// Convert interface{} slice to string slice
			symbols := make([]string, len(symbolsParam))
			for i, symbol := range symbolsParam {
				if symbolStr, ok := symbol.(string); ok {
					symbols[i] = symbolStr
				} else {
					return "", fmt.Errorf("symbol at index %d is not a string", i)
				}
			}

			// Call the gopls tool
			definitions, err := goplsTool.GetCodeDefinitions(pathParam, symbols)
			if err != nil {
				return "", fmt.Errorf("failed to get code definitions: %w", err)
			}

			logger.Info("Code definitions function executed successfully",
				zap.String("functionName", call.Name),
				zap.String("action", actionParam),
				zap.String("filePath", pathParam),
				zap.Strings("symbols", symbols))

			return definitions, nil

		default:
			return "", fmt.Errorf("unknown action: %s", actionParam)
		}

	// Keep backward compatibility with old function names
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

	case "get_code_definitions":
		// Extract parameters
		filePathParam, ok := call.Args["file_path"].(string)
		if !ok {
			return "", fmt.Errorf("file_path parameter is required and must be a string")
		}

		symbolsParam, ok := call.Args["symbols"].([]interface{})
		if !ok {
			return "", fmt.Errorf("symbols parameter is required and must be an array")
		}

		// Convert interface{} slice to string slice
		symbols := make([]string, len(symbolsParam))
		for i, symbol := range symbolsParam {
			if symbolStr, ok := symbol.(string); ok {
				symbols[i] = symbolStr
			} else {
				return "", fmt.Errorf("symbol at index %d is not a string", i)
			}
		}

		// Call the gopls tool
		definitions, err := goplsTool.GetCodeDefinitions(filePathParam, symbols)
		if err != nil {
			return "", fmt.Errorf("failed to get code definitions: %w", err)
		}

		logger.Info("Function call executed successfully",
			zap.String("functionName", call.Name),
			zap.String("filePath", filePathParam),
			zap.Strings("symbols", symbols))

		return definitions, nil

	default:
		return "", fmt.Errorf("unknown function: %s", call.Name)
	}
}

// GoplsTool represents the gopls integration tool
type GoplsTool struct {
	logger      *zap.Logger
	goplsClient *GoplsClient
}

// NewGoplsTool creates a new gopls tool instance
func NewGoplsTool(logger *zap.Logger) (*GoplsTool, error) {
	client, err := NewGoplsClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create gopls client: %w", err)
	}

	return &GoplsTool{
		logger:      logger,
		goplsClient: client,
	}, nil
}

// GetCodeDefinitions retrieves definitions for the requested symbols from gopls
func (gt *GoplsTool) GetCodeDefinitions(filePath string, symbols []string) (string, error) {
	gt.logger.Debug("Getting code definitions from gopls",
		zap.String("filePath", filePath),
		zap.Strings("symbols", symbols))

	// Initialize gopls with the workspace
	err := gt.initializeWorkspace(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to initialize workspace: %w", err)
	}

	var results strings.Builder
	results.WriteString("Code Definitions:\n\n")

	// Read the file content to find symbol positions
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	fileContent := string(content)

	for _, symbol := range symbols {
		gt.logger.Debug("Looking up symbol", zap.String("symbol", symbol))

		// Find symbol position in the file
		position := gt.findSymbolPosition(fileContent, symbol)
		if position == nil {
			results.WriteString(fmt.Sprintf("Symbol '%s': Not found in file\n", symbol))
			continue
		}

		// Get definition from gopls
		definition, err := gt.getDefinitionAtPosition(filePath, *position)
		if err != nil {
			gt.logger.Warn("Failed to get definition for symbol",
				zap.String("symbol", symbol),
				zap.Error(err))
			results.WriteString(fmt.Sprintf("Symbol '%s': Error getting definition - %v\n", symbol, err))
			continue
		}

		results.WriteString(fmt.Sprintf("Symbol '%s':\n", symbol))
		results.WriteString(fmt.Sprintf("  Location: %s\n", definition.URI))
		results.WriteString(fmt.Sprintf("  Line: %d, Character: %d\n",
			definition.Range.Start.Line+1, definition.Range.Start.Character+1))

		// Try to get the actual code content at the definition location
		defContent, err := gt.getCodeAtLocation(definition)
		if err == nil && defContent != "" {
			results.WriteString(fmt.Sprintf("  Code:\n%s\n", defContent))
		}
		results.WriteString("\n")
	}

	result := results.String()
	gt.logger.Info("Successfully retrieved code definitions",
		zap.Int("symbolCount", len(symbols)),
		zap.Int("resultLength", len(result)))

	return result, nil
}

// initializeWorkspace initializes gopls with the workspace
func (gt *GoplsTool) initializeWorkspace(filePath string) error {
	// Initialize gopls if not already done
	if !gt.goplsClient.initialized {
		err := gt.goplsClient.Initialize()
		if err != nil {
			return fmt.Errorf("failed to initialize gopls: %w", err)
		}
	}

	// Read file content for DidOpen
	content, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file content: %w", err)
	}

	// Convert file path to URI and open the document in gopls
	uri := "file://" + filePath
	err = gt.goplsClient.DidOpen(uri, "go", string(content))
	if err != nil {
		return fmt.Errorf("failed to open document in gopls: %w", err)
	}

	return nil
}

// findSymbolPosition finds the position of a symbol in the file content
func (gt *GoplsTool) findSymbolPosition(content, symbol string) *protocol.Position {
	lines := strings.Split(content, "\n")

	for lineNum, line := range lines {
		// Look for the symbol in various contexts
		patterns := []string{
			fmt.Sprintf("func %s(", symbol),
			fmt.Sprintf("func (%s)", symbol),
			fmt.Sprintf("type %s ", symbol),
			fmt.Sprintf("var %s ", symbol),
			fmt.Sprintf("const %s ", symbol),
			fmt.Sprintf("%s :=", symbol),
			fmt.Sprintf("%s =", symbol),
		}

		for _, pattern := range patterns {
			if idx := strings.Index(line, pattern); idx != -1 {
				return &protocol.Position{
					Line:      lineNum,
					Character: idx,
				}
			}
		}

		// Also try simple word boundary match
		if strings.Contains(line, symbol) {
			idx := strings.Index(line, symbol)
			return &protocol.Position{
				Line:      lineNum,
				Character: idx,
			}
		}
	}

	return nil
}

// getDefinitionAtPosition gets the definition at a specific position using gopls
func (gt *GoplsTool) getDefinitionAtPosition(filePath string, position protocol.Position) (*protocol.Location, error) {
	// Convert file path to URI
	uri := "file://" + filePath

	locations, err := gt.goplsClient.GoToDefinition(uri, position.Line, position.Character)
	if err != nil {
		return nil, err
	}

	if len(locations) == 0 {
		return nil, fmt.Errorf("no definition found")
	}

	return &locations[0], nil
}

// getCodeAtLocation retrieves the actual code content at a given location
func (gt *GoplsTool) getCodeAtLocation(location *protocol.Location) (string, error) {
	// Extract file path from URI
	filePath := strings.TrimPrefix(location.URI, "file://")

	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	lines := strings.Split(string(content), "\n")
	startLine := location.Range.Start.Line
	endLine := location.Range.End.Line

	if startLine >= len(lines) {
		return "", fmt.Errorf("start line out of bounds")
	}

	if endLine >= len(lines) {
		endLine = len(lines) - 1
	}

	// Extract the relevant lines
	var result strings.Builder
	for i := startLine; i <= endLine; i++ {
		if i < len(lines) {
			result.WriteString(lines[i])
			if i < endLine {
				result.WriteString("\n")
			}
		}
	}

	return result.String(), nil
}

// Close closes the gopls client
func (gt *GoplsTool) Close() error {
	if gt.goplsClient != nil {
		return gt.goplsClient.Close()
	}
	return nil
}

// storeResponseToFile stores the Gemini response to a file
func (gc *GeminiClient) storeResponseToFile(resp *genai.GenerateContentResponse, filePath string) error {
	// Add timestamp to filename
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	timestampedPath := fmt.Sprintf("%s_%s", timestamp, filePath)

	// Create a structured representation of the response
	var responseData struct {
		Timestamp  string               `json:"timestamp"`
		Candidates []*genai.Candidate   `json:"candidates"`
		Usage      *genai.UsageMetadata `json:"usage_metadata,omitempty"`
	}

	responseData.Timestamp = timestamp
	responseData.Candidates = resp.Candidates
	responseData.Usage = resp.UsageMetadata

	// Marshal to JSON for readability
	data, err := json.MarshalIndent(responseData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal response: %w", err)
	}

	// Write to file
	err = os.WriteFile(timestampedPath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write response to file: %w", err)
	}

	gc.logger.Info("Response successfully stored to file", zap.String("filePath", timestampedPath))
	return nil
}

// storeDebugInfo stores debug information to a file
func (gc *GeminiClient) storeDebugInfo(resp *genai.GenerateContentResponse, filePath string) error {
	// Add timestamp to filename
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	timestampedPath := fmt.Sprintf("%s_%s", timestamp, filePath)

	// Convert the entire response to JSON for debugging
	data, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal debug info: %w", err)
	}

	// Write JSON data to file
	err = os.WriteFile(timestampedPath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write debug info to file: %w", err)
	}

	gc.logger.Info("Debug info successfully stored to file", zap.String("filePath", timestampedPath))
	return nil
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

	// Test the code definitions functionality
	prompt := "Please use the analyze_go_code tool to get code definitions for the symbols 'GeminiClient', 'NewGeminiClient', and 'GenerateContent' from the file '/Users/ayush/keploy/havetodelete/gemini-tool-calls/main.go'."

	logger.Info("Sending prompt to Gemini 2.5 Pro", zap.Int("promptLength", len(prompt)))
	response, err := geminiClient.GenerateContent(ctx, prompt)
	if err != nil {
		logger.Fatal("Failed to generate content", zap.Error(err))
	}

	logger.Info("Gemini Response",
		zap.Int("promptLength", len(prompt)),
		zap.String("response", response))
	logger.Info("Application completed successfully")
}
