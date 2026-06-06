import React, { useState, useEffect, useRef } from 'react';

// Interfaces defining the data structure passed from App.tsx
interface MechanisticConcept {
    probability: number;
    heatmap_vector: number[][]; // changed: heatmap -> heatmap_vector
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

// matplotlib bounding box: represent % of white space created by axes
const PLOT_MARGINS = {
    left: 0.058,
    right: 0.02,
    top: 0.04,
    bottom: 0.16
};

const MechanisticHeatmap: React.FC<MechanisticHeatmapProps> = ({ spectrogramImage, gradcamImage, concepts }) => {
    const [activeIndex, setActiveIndex] = useState<number | null>(-1);    
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
        if (activeIndex === null || !concepts[activeIndex]) return;

        // fix: heatmap -> heatmap_vector
        const activeHeatmap = concepts[activeIndex][1].heatmap_vector; // 12x101 grid
        const colorRgb = CONCEPT_COLORS[activeIndex % CONCEPT_COLORS.length];

        // calculate bounding box (inner plot area only)
        const plotLeft = canvas.width * PLOT_MARGINS.left;
        const plotTop = canvas.height * PLOT_MARGINS.top;
        const plotWidth = canvas.width * (1 - PLOT_MARGINS.left - PLOT_MARGINS.right);
        const plotHeight = canvas.height * (1 - PLOT_MARGINS.top - PLOT_MARGINS.bottom);


        // calculate dimensions based on 12x101 AST grid config
        const gridRows = 12;
        const gridCols = 101;
        const blockWidth = plotWidth / gridCols;
        const blockHeight = plotHeight / gridRows;

        // paint the 2d matrix
        for (let row = 0; row < activeHeatmap.length; row++) {
            for (let col = 0; col < activeHeatmap[row].length; col++) {
                const rawAlpha = activeHeatmap[row][col];

                // only draw if there is actual mathematic attention
                if (rawAlpha > 0.05) {
                    // increasing opacity for troubleshooting overlay
                    const displayAlpha = Math.min(rawAlpha * 2.5, 1.0);

                    ctx.fillStyle = `rgba(${colorRgb}, ${displayAlpha})`;

                    // draw rectangle: x, y, width, height
                    // adding +0.5 prevents microscopic rendering gaps between blocks
                    ctx.fillRect(
                        plotLeft + (col * blockWidth),
                        plotTop + (row * blockHeight),
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
                
                {/* dynamically swap base image based on state; attention rollout default */}
                <img
                    ref={imageRef}
                    src={activeIndex === -1 ? gradcamImage : spectrogramImage}
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
                        borderRadius: '4px',
                        filter: 'blur(6px)'
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