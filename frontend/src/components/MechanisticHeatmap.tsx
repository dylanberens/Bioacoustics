import React, { useState, useEffect, useRef } from 'react';

// Interfaces defining the data structure passed from App.tsx
interface MechanisticConcept {
    probability: number;
    heatmap: number[][];
}

interface MechanisticHeatmapProps {
    spectrogramImage: string;
    concepts: Array<[string, MechanisticConcept]>;
}

// synthwave color palette for that look
const CONCEPT_COLORS = [
    '16, 185, 129', // vibrant green
    '6, 182, 212',  // cyber blue
    '217, 70, 239', // mystic purple
    '245, 158, 11', // construction orange
];

const MechanisticHeatmap: React.FC<MechanisticHeatmapProps> = ({ spectrogramImage, concepts }) => {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);    
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);

    // core drawing engine: runs when active toggle changes
    useEffect(() => {
        const canvas = canvasRef.current;
        const image = imageRef.current;

        if (!canvas || !image) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // sync canvas internal resolution with physical CSS display size
        // prevents the heatmap from rendering blurry/misaligned
        canvas.width = image.clientWidth;
        canvas.height = image.clientHeight;

        // clear previous drawings on canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // if no toggle active, leave canvas blank
        if (activeIndex == null ||!concepts[activeIndex]) return;

        const activeHeatmap = concepts[activeIndex][1].heatmap; // 12x101 grid
        const colorRgb = CONCEPT_COLORS[activeIndex % CONCEPT_COLORS.length];

        // calculate dimensions based on 12x101 AST grid config
        const gridRows = 12;
        const gridCols = 101;
        const blockWidth = canvas.width / gridCols;
        const blockHeight = canvas.height / gridRows;

        // paint the 2d matrix
        for (let row = 0; row < activeHeatmap.length; row++) {
            for (let col = 0; col < activeHeatmap[row].length; col++) {
                const rawAlpha = activeHeatmap[row][col];

                // only draw if there is actual mathematic attention
                if (rawAlpha > 0.05) {
                    // cap opacity at 0.75 so underlying audio visual isn't completely hidden
                    const displayAlpha = rawAlpha * 0.75;

                    ctx.fillStyle = `rgba(${colorRgb}, ${displayAlpha})`;

                    // draw rectangle: x, y, width, height
                    // adding +0.5 prevents microscopic rendering gaps between blocks
                    ctx.fillRect(
                        col * blockWidth,
                        row * blockHeight,
                        blockWidth + 0.5,
                        blockHeight + 0.5
                    );
                }
            }
        }
    }, [activeIndex, concepts, spectrogramImage]);

    return (
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>

            {/* 1. Layered Canvas Sandbox */}
            <div style={{
                position: 'relative',
                width: '100%',
                background: '#1F2937',
                padding: '1rem',
                borderRadius: '8px',
                border: '2px solid rgba(16, 185, 129, 0.6)',
                boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3)'
            }}>
                
                {/* Base Spectrogram Image (Bottom Layer) */}
                <img
                    ref={imageRef}
                    src={spectrogramImage}
                    alt="Audio Spectrogram"
                    style={{
                        width: '100%',
                        height: 'auto',
                        borderRadius: '4px',
                        display: 'block'
                    }}
                />

                {/* Absolute Overlay Canvas (Top Layer) */}
                <canvas
                    ref={canvasRef}
                    style={{
                        position: 'absolute',
                        top: '1rem',    // match the padding of the wrapper
                        left: '1rem',   // match the padding of the wrapper
                        width: `calc(100% - 2rem)`,
                        height: `calc(100% - 2rem)`,
                        pointerEvents: 'none', // let mouse clicks pass through to the iamge
                        borderRadius: '4px'
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
                    onClick={() => setActiveIndex(null)}
                    style={{
                        padding: '0.5rem 1rem',
                        background: activeIndex === null ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        borderRadius: '9999px',
                        color: activeIndex === null ? 'white' : '#9CA3AF',
                        fontFamily: 'monospace',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                    }}
                >
                    CLEAR OVERLAY
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
        </div>

        <p style={{
            fontSize: 'clamp(0.8rem, 1.5vw, 1rem)',
            color: '#9CA3AF',
            marginTop: '1.5rem',
            textAlign: 'center',
            fontFamily: 'monospace'
        }}>
            Select an acoustic signature to view the model's spatial attention (Mechanistic Interpretability).
        </p>

    </div>
    );
};

export default MechanisticHeatmap;