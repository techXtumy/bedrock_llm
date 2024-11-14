# Bedrock LLM Architecture

The Bedrock LLM library is structured into two main components:

1. `bedrock_be`: Infrastructure and services for deploying LLM applications.
2. `bedrock_llm`: LLM orchestration and interaction logic.

## bedrock_be

The `bedrock_be` component provides infrastructure and services for deploying LLM applications. This includes:

- Deployment scripts and configurations for AWS services
- Service management tools for monitoring and scaling
- Integration with AWS services such as Lambda, ECS, and API Gateway
- Infrastructure as Code (IaC) templates for easy deployment and management

## bedrock_llm

The `bedrock_llm` component handles LLM orchestration and interaction logic. It includes:

- `client.py`: The main LLMClient class for interacting with various LLM models.
- `agent.py`: The Agent class for handling tool-based interactions and more complex LLM workflows.
- `config/`: Configuration classes for models, retry logic, and other settings.
- `models/`: Implementation classes for different LLM models (Anthropic, Amazon, Meta, etc.).
- `monitor/`: Performance monitoring and logging functionality for tracking LLM operations.
- `schema/`: Data models and schemas for messages, tools, and other components.
- `types/`: Enums and custom types used throughout the library for type safety and consistency.

This structure allows for easy extension and maintenance of the library, with clear separation between the infrastructure and the LLM interaction logic. The modular design enables developers to:

1. Easily add support for new LLM models
2. Customize deployment configurations for different environments
3. Implement advanced LLM workflows using the Agent class
4. Monitor and optimize LLM application performance

By combining `bedrock_be` and `bedrock_llm`, developers can rapidly prototype, build, and deploy production-ready LLM applications with robust infrastructure support.