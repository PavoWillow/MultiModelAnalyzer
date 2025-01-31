# MultiModelAnalyzer

This project is a multi-model analysis application that leverages LaunchDarkly AI Configs to dynamically manage and evaluate different AI models. It supports OpenAI, Anthropic, and Google Generative AI (Gemini) models, providing a comprehensive framework for generating and analyzing AI responses.

## Features

- **Multi-Model Support**: Integrates with OpenAI, Anthropic, and Google Generative AI models.
- **Dynamic Configuration**: Uses LaunchDarkly AI Configs to manage model variations and configurations.
- **Response Evaluation**: Analyzes AI responses for human-likeness, content quality, coherence, and grammar.
- **Performance Metrics**: Logs execution time, token usage, and cost for each model query.

## Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PavoWillow/MultiModelAnalyzer.git
   cd MultiModelAnalyzer
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Rename `.env.example` to `.env`.
   - Add your API keys for OpenAI, Anthropic, Google Generative AI, and LaunchDarkly in the `.env` file.

### Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Access the UI**:
   - The application serves a UI for interaction, located in the `static` directory. Ensure your server is configured to serve static files.

3. **Evaluate AI Responses**:
   - The application evaluates AI responses based on predefined criteria and logs the results. Check the logs for detailed performance metrics and evaluation feedback.

### Configuration

- **LaunchDarkly AI Configs**: The application uses LaunchDarkly to manage AI model configurations. Ensure your LaunchDarkly flags are set up correctly to utilize this feature.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contact

For questions or support, please contact [pvwloomis@gmail.com](mailto:pvwloomis@gmail.com).
