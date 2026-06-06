import React, { useState } from 'react';

// Interfaces defining the data structure passed from App.tsx
interface MechanisticConcept {
    probability: number;
    image_b64: string; // CHANGED: replaced heatmap_vector: number[][]
}

interface MechanisticHeatmapProps {
    spectrogramImage: string;
    gradcamImage: string; // for default global attention
    concepts: Array<[string, MechanisticConcept]>;
}

// synthwave color palette for that look
const CONCEPT_COLORS = [
    '16, 185, 129', // vibrant green
    '6, 182, 212',  // cyber blue
    '217, 70, 239', // mystic purple
    '245, 158, 11', // construction orange
];

const MechanisticHeatmap: React.FC<MechanisticHeatmapProps> = ({ spectrogramImage, gradcamImage, concepts }) => {
    const [activeIndex, setActiveIndex] = useState<number | null>(-1);    

    // clean routing logic for display image
    let displayImage = spectrogramImage;
    if (activeIndex === -1) {
        displayImage = gradcamImage; // default global attention
    } else if (activeIndex !== null && concepts[activeIndex]) {
        // prepend data URI scheme since python sends raw b64 strings
        displayImage = `data:image/png;base64,${concepts[activeIndex][1].image_b64}`;
    }

    return (
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>

            {/* 1. Layered Canvas Sandbox -> now just image display */}
            <div style={{
                position: 'relative',
                width: '100%',
                background: '#1F2937',
                padding: '1rem',
                borderRadius: '8px',
                border: '2px solid rgba(16, 185, 129, 0.6)',
                boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3)'
            }}>
                
                {/* dynamically swap base image based on state; attention rollout default */}
                <img
                    src={displayImage}
                    alt="Acoustic Attention Visualization"
                    style={{
                        width: '100%',
                        height: 'auto',
                        borderRadius: '4px',
                        display: 'block'
                    }}
                />
            </div>

            {/* 2. mutually exclusive toggles */}
            <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '0.75rem',
                marginTop: '1.5rem',
                justifyContent: 'center'
            }}>
                <button
                    onClick={() => setActiveIndex(-1)}
                    style={{
                        padding: '0.5rem 1rem',
                        background: activeIndex === -1 ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        borderRadius: '9999px',
                        color: activeIndex === -1 ? 'white' : '#9CA3AF',
                        fontFamily: 'monospace',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                    }}
                >
                    ATTENTION ROLLOUT
                </button>

                {concepts.map(([concept, data], index) => {
                    const isActive = activeIndex === index;
                    const colorRgb = CONCEPT_COLORS[index % CONCEPT_COLORS.length];

                    return (
                        <button
                        key={concept}
                        onClick={() => setActiveIndex(index)}
                        style={{
                            padding: '0.5rem 1rem',
                            background: isActive ? `rgba(${colorRgb}, 0.2)` : 'transparent',
                            border: `1px solid rgba(${colorRgb}, ${isActive ? '0.8' : '0.3'})`,
                            borderRadius: '9999px',
                            color: isActive ? `rgb(${colorRgb})` : '#9CA3AF',
                            fontFamily: 'monospace',
                            textTransform: 'uppercase',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            boxShadow: isActive ? `0 0 10px rgba(${colorRgb}, 0.4)` : 'none'
                        }}
                        >
                            {concept.replace('_', ' ')} ({(data.probability * 100).toFixed(0)}%)
                        </button>
                    );
                })}

                <button
                    onClick={() => setActiveIndex(null)}
                    style={{
                        padding: '0.5rem 1rem',
                        background: 'transparent',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: '9999px',
                        color: activeIndex === null ? '#EF4444' : '#9CA3AF',
                        fontFamily: 'monospace',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                    }}
                >
                    CLEAR
                </button>
            </div>

        <p style={{
            fontSize: 'clamp(0.8rem, 1.5vw, 1rem)',
            color: '#9CA3AF',
            marginTop: '1.5rem',
            textAlign: 'center',
            fontFamily: 'monospace'
        }}>
            Mechanistic Interpretability: Select an acoustic signature to view the model's spatial attention
        </p>

    </div>
    );
};

export default MechanisticHeatmap;