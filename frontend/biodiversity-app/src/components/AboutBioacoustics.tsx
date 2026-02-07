import { motion } from "framer-motion";
import { WavyBackground } from "./wavy-background";
import { Spotlight } from "./spotlight";
import { FloatingNav } from "./floating-nav";

const BackgroundBeams = () => {
  return (
    <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
      <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to bottom, transparent, transparent, black)' }}></div>
      {Array.from({ length: 50 }).map((_, i) => (
        <motion.div
          key={i}
          style={{
            position: 'absolute',
            background: 'linear-gradient(to bottom, rgba(52, 211, 153, 0.2), transparent)',
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            width: `${Math.random() * 2 + 0.5}px`,
            height: `${Math.random() * 200 + 100}px`,
            transform: `rotate(${Math.random() * 360}deg)`,
          }}
          animate={{
            opacity: [0, 1, 0],
            scale: [0.8, 1.2, 0.8],
          }}
          transition={{
            duration: Math.random() * 3 + 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  );
};

const GridBackground = () => {
  return (
    <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
      <div 
        style={{
          width: '100%',
          height: '100%',
          opacity: 0.2,
          backgroundImage: `
            linear-gradient(rgba(16, 185, 129, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(16, 185, 129, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}
      />
    </div>
  );
};

export const AboutBioacoustics = () => {
  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'black', color: 'white', position: 'relative', overflow: 'hidden' }}>
      <WavyBackground 
        className=""
        containerClassName=""
        colors={["#10B981", "#059669", "#047857", "#065F46", "#064E3B"]}
        waveWidth={50}
        backgroundFill="black"
        blur={10}
        speed="fast"
        waveOpacity={0.5}
      >
        <FloatingNav />
        
        <Spotlight
          className=""
          fill="#10B981"
        />

        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          style={{
            position: 'relative',
            zIndex: 10,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'flex-start',
            minHeight: '100vh',
            padding: '120px 16px 40px'
          }}
        >
          <div style={{ maxWidth: '896px', margin: '0 auto', textAlign: 'center' }}>
            <motion.h1 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              style={{
                fontSize: 'clamp(48px, 7vw, 96px)',
                fontWeight: 'bold',
                background: 'linear-gradient(to bottom, white, #10B981)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '32px'
              }}
            >
              About Bioacoustics
            </motion.h1>

            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              style={{ fontSize: 'clamp(16px, 1.5vw, 18px)', color: '#D1D5DB', lineHeight: '1.8',
                        // DYLAN frontend additions
                        textAlign: 'left',
                        background: 'rgba(0, 0, 0, 0.7)',
                        backdropFilter: 'blur(10px)',
                        borderRadius: '24px',
                        border: '1px solid rgba(16, 185, 129, 0.2)',
                        boxShadow: '0 4px 30px rgba(0, 0, 0, 0.5)',
                        padding: '40px',
                        marginBottom: '40px'
               }}
            >
              <p style={{ marginBottom: '24px' }}>
                Bioacoustics is the study of sound as it relates to species and ecosystems. The theory behind this project is
                that species use sounds to communicate, and thereby evolved to occupy different frequency bands in the overall
                "soundscape", which is the entire audio space of an ecosystem.
              </p>

              {/* Green divider line added by dylan */}
              <div style={{
                height: '1px',
                width: '100%',
                background: 'linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.4), transparent)',
                margin: '2rem 0'
              }}/>

              <p style={{ marginBottom: '24px' }}>
                Our project uses a state of the art <strong>Audio Spectrogram Transformer (AST)</strong>- a type of Vision Transformer (ViT)-- with a custom regression
                head to analyze audio files, which we project onto 2D space as a Mel Spectrogram (an image representing the audio's frequency, time and amplitude).
                We have fine tuned this custom model on a dataset of 6,719 soundscapes from the Amazon Rainforest, achieving an r-squared of 0.96 in an example of
                successful Sequential Transfer Learning.
              </p>

              <div style={{
                height: '1px',
                width: '100%',
                background: 'linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.4), transparent)',
                margin: '2rem 0'
              }}/>

              <p style={{ marginBottom: '16px' }}>
                We developed our custom <strong>Acoustic Diversity Index (ADI)</strong> score as our target variable to assess the presence and dispersion of
                audio activity across 30 frequency bands. Our custom "Robust ADI" target variable is defined by five key characteristics:
              </p>

              <div style={{ paddingLeft: '20px', marginBottom: '32px', color: '#E5E7EB' }}>
                {[
                  { title: "Background Subtraction", desc: "Removes constant noise (like rain or rivers) by subtracting the median energy across the spectrogram." },
                  { title: "Adaptive Thresholding", desc: "Only counts sounds >13.5 dB above baseline." },
                  { title: "Frequency Banding", desc: "Splits the spectrogram into 30 vertically stacked 200Hz frequency bands." },
                  { title: "Shannon Entropy", desc: "Measures the evenness of activity across the 30 frequency bands to reward diverse ecosystems with different frequencies." },
                  { title: "Soft Fallback", desc: "Assigns fractional score based on energy sum if nothing >13.5 dB, to discourage true flat 0 scores." }
                ].map((item, i) => (
                  <div key={i} style={{ marginBottom: '12px', display: 'flex', gap: '10px' }}>
                    <span style={{ color: '#10B981', fontWeight: 'bold' }}>{i + 1}.</span>
                    <span><strong>{item.title}:</strong> {item.desc}</span>
                  </div>
                ))}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginTop: '32px' }}>
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.6 }}
                  style={{
                    background: 'linear-gradient(135deg, rgba(6, 78, 59, 0.2), rgba(0, 0, 0, 0.4))',
                    padding: '24px',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)'
                  }}
                >
                  <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#10B981', marginBottom: '12px' }}>ADI Score</h3>
                  <p style={{ color: '#D1D5DB' }}>
                    A numerical score representating the biodiversity of the environment in the audio file
                  </p>
                </motion.div>

                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.8 }}
                  style={{
                    background: 'linear-gradient(135deg, rgba(6, 78, 59, 0.2), rgba(0, 0, 0, 0.4))',
                    padding: '24px',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)'
                  }}
                >
                  <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#10B981', marginBottom: '12px' }}>Mel Spectrogram</h3>
                  <p style={{ color: '#D1D5DB' }}>
                    2D representation of the soundscape audio file, showing frequencies, time and amplitude
                  </p>
                </motion.div>

                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 1.0 }}
                  style={{
                    background: 'linear-gradient(135deg, rgba(6, 78, 59, 0.2), rgba(0, 0, 0, 0.4))',
                    padding: '24px',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)'
                  }}
                >
                  <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#10B981', marginBottom: '12px' }}>Score Distribution</h3>
                  <p style={{ color: '#D1D5DB' }}>
                    Interactive visualization showing the distribution of biodiversity scores from the Kaggle Amazon Rainforest dataset
                  </p>
                </motion.div>

                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 1.2 }}
                  style={{
                    background: 'linear-gradient(135deg, rgba(6, 78, 59, 0.2), rgba(0, 0, 0, 0.4))',
                    padding: '24px',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)'
                  }}
                >
                  <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#10B981', marginBottom: '12px' }}>Attention Rollout Heatmap</h3>
                  <p style={{ color: '#D1D5DB' }}>
                    Visualization for Explainable AI (XAI), marking spectrogram regions that influenced the model's predicted score
                  </p>
                </motion.div>
              </div>

              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 1.4 }}
                style={{ fontSize: '20px', color: '#10B981', fontWeight: '600', marginTop: '40px', textAlign: 'center' }}
              >
                Machine Learning has powerful applications towards protecting our ecosystems and planet
              </motion.p>
            </motion.div>
          </div>
        </motion.div>
      </WavyBackground>

      <BackgroundBeams />
      <GridBackground />
    </div>
  );
};