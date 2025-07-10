#!/bin/bash

# Gemini 2.5 Pro Setup Script
# This script helps set up the necessary Google Cloud resources for Gemini API access

set -e

echo "🚀 Setting up Gemini 2.5 Pro with Vertex AI"
echo "============================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI is not installed. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
read -p "Enter your Google Cloud Project ID: " PROJECT_ID

if [[ -z "$PROJECT_ID" ]]; then
    echo "❌ Project ID cannot be empty"
    exit 1
fi

# Set the project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable necessary APIs
echo "🔧 Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com

# Create service account
SERVICE_ACCOUNT_NAME="gemini-service-account"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "👤 Creating service account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --description="Service account for Gemini API access" \
    --display-name="Gemini Service Account" || echo "Service account might already exist"

# Grant permissions
echo "🔐 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.user"

# Create credentials directory
mkdir -p ./credentials

# Generate and download service account key
CREDENTIALS_FILE="./credentials/gemini-credentials.json"
echo "🗝️  Creating service account key..."
gcloud iam service-accounts keys create $CREDENTIALS_FILE \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

# Create .env file
echo "📝 Creating .env file..."
cat > .env << EOL
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$CREDENTIALS_FILE
EOL

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Summary:"
echo "   - Project ID: $PROJECT_ID"
echo "   - Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "   - Credentials: $CREDENTIALS_FILE"
echo "   - Environment file: .env"
echo ""
echo "🚀 You can now run the application with:"
echo "   source .env && go run main.go"
echo ""
echo "⚠️  Important: Keep your credentials file secure and never commit it to version control!"
