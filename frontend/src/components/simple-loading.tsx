"use client";
import { motion } from "framer-motion";

export const SimpleLoading = () => {
    return (
        <div 
            style={{
                padding: '40px',
                border: '3px solid #10B981',
                borderRadius: '20px',
                backgroundColor: '#000000',
                boxShadow: '0 0 30px rgba(16, 185, 129, 0.5)',
                textAlign: 'center',
                maxWidth: '400px',
                width: '90%',
            }}
        >
            <motion.div
                style={{
                    width: '50px',
                    height: '50px',
                    borderRadius: '50%',
                    backgroundColor: '#10B981',
                    boxShadow: '0 0 20px #10B981',
                    margin: '0 auto 30px auto',
                }}
                animate={{
                    scale: [0.8, 1.2, 0.8],
                    opacity: [0.5, 1, 0.5],
                }}
                transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "easeInOut"
                }}
            />

            <h2 style={{
                color: '#ffffff',
                fontSize: '24px',
                fontWeight: 'bold',
                marginBottom: '16px',
                fontFamily: 'monospace',
                letterSpacing: '1px'
            }}>
                ANALYZING ECOSYSTEM . . .
            </h2>

            <p style={{ color: '#cccccc', fontSize: '14px' }}>
                This takes at least 60 seconds using my hotspot, so please do some deep reflecting while waiting.
            </p>
        </div>
    );
};