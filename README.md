# Learn2Slither

A snake that learns how to behave in an environment through trial and error.

## Current Architecture Vision

```mermaid
graph LR
    subgraph Input Layer
        I[State Vector: 12 Features]
    end

    subgraph Hidden Layers
        H1[Linear Layer 1: 256 Neurons + ReLU]
        H2[Linear Layer 2: 256 Neurons + ReLU]
    end

    subgraph Output Layer
        O[Q-Values: 4 Outputs]
    end

    I -->|12 inputs| H1
    H1 -->|256 weights| H2
    H2 -->|256 weights| O
    
    O -.-> A1(Action 0: Up)
    O -.-> A2(Action 1: Down)
    O -.-> A3(Action 2: Left)
    O -.-> A4(Action 3: Right)

    %% Minimalistic borders for visibility across themes
    style I stroke:#333,stroke-width:2px
    style H1 stroke:#333,stroke-width:2px
    style H2 stroke:#333,stroke-width:2px
    style O stroke:#333,stroke-width:2px
```
