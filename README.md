# WhatsApp Chatbot

An intelligent conversational agent built with Python, TensorFlow, and NLTK that integrates with WhatsApp to provide automated responses.

## Overview

This WhatsApp Chatbot uses natural language processing techniques to understand user queries and provide relevant responses based on trained intents. It leverages machine learning to improve response accuracy over time.

## Features

- Natural language understanding using NLTK
- Deep learning model built with TensorFlow/Keras
- Intent recognition and response generation
- WhatsApp integration via Twilio
- Customizable response templates
- Easy-to-extend intent configuration

## Requirements

- Python 3.8+
- TensorFlow 2.9+
- NLTK 3.7+
- Flask 2.1+
- Twilio 7.9+
- NumPy 1.22+
- Pandas 1.4+

## Project Structure

```
whatsapp-chatbot/
├── src/                    # Source code directory
│   ├── training/           # ML model training scripts
│   │   ├── training.py     # Training implementation
│   │   └── intents.json    # Intent definitions
│   ├── chatting.py         # Core chat functionality
│   └── app.py              # Flask web server
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Training the Model

The chatbot requires training before it can respond to messages:

```bash
cd src/training
python training.py
```

This will:
1. Process the intents.json file
2. Create embeddings and train a deep learning model
3. Save the trained model and tokenizer for inference

## Usage

1. Start the Flask server:
```bash
cd src
python app.py
```

2. Configure your Twilio WhatsApp Sandbox to point to your server
3. Start chatting with the bot through WhatsApp!

## Customizing Intents

To customize the bot's knowledge and responses, edit the `intents.json` file:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello! How can I help you?", "Hi there! What can I do for you?"]
    },
    ...
  ]
}
```

After modifying intents, re-train the model using the training script.

