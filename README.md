# AI Help Bot

A Telegram bot powered by Google's Gemini AI that can process text, analyze images, and extract information from PDFs. Built to run on Google Cloud Functions Gen2.

## Features

- 🤖 Text message processing with Gemini AI
- 🖼️ Image analysis and description
- 📄 PDF text extraction and analysis
- ⚡ Rate limiting and concurrent request handling
- 🔄 Automatic retries for better reliability
- 🔄 Automatic retries for better reliability
- ☁️ Serverless deployment on GCF Gen2

## Prerequisites

- Python 3.11+
- Google Cloud account with Gemini API access
- Telegram Bot Token from @BotFather
- Google Cloud CLI installed (for deployment)

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd aihelpbot
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.yaml.template .env.yaml
# Edit .env.yaml with your actual values
```

4. Deploy to Google Cloud Functions:
```bash
gcloud functions deploy aihelpbot \
  --gen2 \
  --runtime=python311 \
  --region=your-region \
  --source=. \
  --entry-point=app \
  --trigger-http \
  --env-vars-file=.env.yaml
```

5. Set up webhook for your bot:
```
https://api.telegram.org/bot<YourBotToken>/setWebhook?url=<YourGCFURL>
```

## Local Development

Run the bot locally in polling mode:
```bash
python main.py
```

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram Bot Token (from @BotFather)
- `GEMINI_API_KEY`: Your Google Gemini API Key
- `ALLOWED_TELEGRAM_IDS`: Comma-separated list of allowed Telegram user IDs

## Features in Detail

### Text Processing
Send any text message to get an AI-powered response using Gemini.

### Image Analysis
Send photos or image files to get AI analysis and description.

### PDF Processing
Send PDF documents to extract and analyze their content.

## Rate Limiting

- Maximum 3 requests per minute per user
- Maximum 10 concurrent requests per user
- 5-minute timeout for long-running requests

## Error Handling

- Automatic retries for network issues
- Exponential backoff for rate limits
- Detailed error messages for better debugging

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 