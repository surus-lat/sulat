# SURUS Architecture

## System Structure

```mermaid
graph TB
    User[User Application] --> SURUS[SURUS Client]
    
    SURUS --> TaskRouter[Task Router]
    TaskRouter --> Transcribe[transcribe()]
    TaskRouter --> Summarize[summarize()]
    TaskRouter --> ExtractJSON[extract_to_json()]
    TaskRouter --> Chat[chat()]
    TaskRouter --> Annotate[annotate()]
    
    Transcribe --> AudioEngine[Audio Engine]
    Summarize --> TextEngine[Text Engine]
    ExtractJSON --> TextEngine
    Chat --> TextEngine
    Annotate --> VisionEngine[Vision Engine]
    
    AudioEngine --> AudioModels[Audio Models]
    TextEngine --> TextModels[Text Models]
    VisionEngine --> VisionModels[Vision Models]
    
    AudioModels --> OpenAI_Audio[OpenAI Whisper]
    AudioModels --> AssemblyAI[AssemblyAI]
    
    TextModels --> OpenAI_Text[OpenAI GPT]
    TextModels --> Anthropic[Anthropic Claude]
    TextModels --> TogetherAI[Together AI]
    
    VisionModels --> OpenAI_Vision[OpenAI GPT-4V]
    VisionModels --> Anthropic_Vision[Anthropic Claude Vision]
```

## Task Interface

```mermaid
classDiagram
    class TaskVerb {
        <<abstract>>
        +high_performance: bool = False
        +prompt_module: str = "default"
        +custom_prompt: str = None
        +model_override: str = None
        +execute()
        +get_model_config()
        +compose_prompt()
    }
    
    class TranscribeTask {
        +language: str = "auto"
        +format: str = "text"
        +timestamps: bool = False
        +execute(audio_input)
    }
    
    class SummarizeTask {
        +length: str = "medium"
        +style: str = "concise"
        +execute(text_input)
    }
    
    class ExtractJSONTask {
        +schema: dict = None
        +strict_mode: bool = True
        +execute(text_input)
    }
    
    class ChatTask {
        +context: list = []
        +temperature: float = 0.7
        +execute(message)
    }
    
    TaskVerb <|-- TranscribeTask
    TaskVerb <|-- SummarizeTask
    TaskVerb <|-- ExtractJSONTask
    TaskVerb <|-- ChatTask
```

## Prompt Composition

```mermaid
graph LR
    Input[User Input] --> BasePrompt[Base Prompt Module]
    BasePrompt --> TaskSpecific[Task-Specific Instructions]
    TaskSpecific --> UserCustom[User Custom Additions]
    UserCustom --> FinalPrompt[Final Prompt]
    
    BasePrompt --> |can be| Replaced[Completely Replaced]
    Replaced --> FinalPrompt
    
    subgraph "Prompt Modules"
        TranscribeBase[Transcribe Base Module]
        SummarizeBase[Summarize Base Module]
        ExtractBase[Extract JSON Base Module]
        ChatBase[Chat Base Module]
    end
    
    BasePrompt -.-> TranscribeBase
    BasePrompt -.-> SummarizeBase
    BasePrompt -.-> ExtractBase
    BasePrompt -.-> ChatBase
```

## Model Selection

```mermaid
flowchart TD
    Start[Task Request] --> CheckPerf{high_performance?}
    
    CheckPerf -->|False| DefaultModel[Use Default Model]
    CheckPerf -->|True| HighPerfModel[Use High Performance Model]
    
    DefaultModel --> CheckOverride{model_override specified?}
    HighPerfModel --> CheckOverride
    
    CheckOverride -->|No| TaskModelMap[Get Model from Task Mapping]
    CheckOverride -->|Yes| CustomModel[Use Override Model]
    
    TaskModelMap --> ModelConfig[Load Model Configuration]
    CustomModel --> ModelConfig
    
    ModelConfig --> PromptCompose[Compose Prompt]
    PromptCompose --> APICall[Make API Call]
    APICall --> Response[Return Response]
    
    subgraph "Default Models"
        AudioDefault[Whisper Base]
        TextDefault[GPT-3.5 Turbo]
        VisionDefault[GPT-4V]
    end
    
    subgraph "High Performance Models"
        AudioHighPerf[Whisper Large]
        TextHighPerf[GPT-4 Turbo]
        VisionHighPerf[Claude 3.5 Sonnet]
    end
```