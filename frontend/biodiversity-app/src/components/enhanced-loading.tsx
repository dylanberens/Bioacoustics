"use client";
import { motion } from "framer-motion";

export const EnhancedLoading = ({ text = "ANALYZING..." }: { text?: string }) => {
  return (

    <div className="relative w-full max-w-md overflow-hidden rounded-3xl p-[3px]">
      <motion.div
        className="absolute inset-[-200%]"
        style={{
          background: "conic-gradient(from 90deg at 50% 50%, #00000000 50%, #fbbf24 80%, #ea580c 100%)",
        }}
        animate={{ rotate: 360 }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      <div className="relative z-10 flex flex-col items-center justify-center w-full max-w-md p-8 rounded-[22px] bg-neutral-950/90 backdrop-blur-md">
        {/* Pulsing Circles */}
        <div className="relative flex items-center justify-center mb-8">
          {[...Array(3)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-20 h-20 border-2 border-orange-500/30 rounded-full"
              animate={{
                scale: [1, 2, 1],
                opacity: [0.5, 0, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.4,
              }}
            />
          ))}
          <motion.div
            className="w-4 h-4 bg-orange-500 rounded-full shadow-[0_0_20px_#f97316]"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            />
        </div>

        {/* Loading Text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-orange-400 font-mono text-xl font-bold tracking-wider mb-8 text-center"
        >
          {text.split("").map((char, i) => (
            <motion.span
              key={i}
              animate={{
                opacity: [0.4, 1, 0.4],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: i * 0.1,
              }}
            >
              {char}
            </motion.span>
          ))}
        </motion.div>

        {/* Progress Bars */}
        <div className="w-full space-y-2">
          {["Loading Audio Spectrogram Transformer . . .", "Extracting Mel-Spectrogram . . .", "Calculating ADI Score . . ."].map((process, i) => (
            <div key={process} className="space-y-1">
              <div className="flex justify-between text-[10px] uppercase tracking-widest text-orange-200/50 font-mono">
                <span>{process}</span>
              </div>

              <div className="w-full h-1 bg-gray-800/50 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-orange-800 to-yellow-500"
                  animate={{
                    width: ["0%", "100%"],
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    delay: i * 0.5,
                    ease: "easeInOut",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};