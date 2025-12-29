# Detailed Pipeline Diagram

    A[PDF Path Input] --> B{Valid PDF?}
    B -- No --> X[Abort + Log Warning]
    B -- Yes --> C[Convert PDF to Images<br/>pdf2image]

    C --> D[Iterate Pages]
    D --> E[Vision OCR per Page<br/>LLM.invoke()]
    E --> F[Markdown Output]
    F --> G[OCR Cache JSON]

    G --> H[LangChain Document Conversion]
    H --> I[RecursiveCharacterTextSplitter]
    I --> J[Chunks]

    J --> K[Metadata Enrichment]
    K --> K1[Page Number]
    K --> K2[Chunk ID]
    K --> K3[Document ID]
    K --> K4[Semester / Subject / Unit]
    K --> K5[Embedding Model]
    K --> K6[Ingestion Version]

    K --> L[Chunk Cache JSON]

    L --> M{Qdrant Collection Exists?}
    M -- No --> N[Create Collection]
    M -- Yes --> O[Validate Embedding Model]

    O -->|Mismatch| P[Abort with Error]
    O -->|Match| Q[Generate Embeddings]

    Q --> R[Insert Vectors into Qdrant]

    subgraph Logging
        S1[Global Logger]
        S2[Per-PDF Logger]
    end

    A --> Logging
    R --> Logging
