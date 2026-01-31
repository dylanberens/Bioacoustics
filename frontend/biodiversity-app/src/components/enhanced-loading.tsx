"use client";
import { motion } from "framer-motion";

export const EnhancedLoading = ({ text = "ANALYZING..." }: { text?: string }) => {
  return (
    <div className="flex flex-col items-center justify-center w-full max-w-md p-8 rounded-3xl bg-neutral-900/80 border border-green-500/20 shadow-[0_0_50px_rgba(16,185,129,0.1)] backdrop-blur-md">
      {/* Pulsing Circles */}
      <div className="relative flex items-center justify-center mb-8">
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-20 h-20 border-2 border-green-400 rounded-full"
            animate={{
              scale: [1, 2, 1],
              opacity: [0.7, 0, 0.7],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: i * 0.4,
            }}
          />
        ))}
        <motion.div
          className="w-4 h-4 bg-[#10B981] rounded-full shadow-[0_0_20px_#10B981]"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          />
      </div>

      {/* Loading Text */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-[#10B981] font-mono text-xl font-bold tracking-wider mb-8 text-center"
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
            <div className="flex justify-between text-[10px] uppercase tracking-widest text-green-400/70 font-mono">
              <span>{process}</span>
            </div>

            <div className="w-full h-1 bg-gray-800/50 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-green-600 to-[#10B981]"
                animate={{
                  width: ["0%", "100%"],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: i * 0.3,
                  ease: "easeInOut",
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};