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

const teamMembers = [
  {
    name: "Dylan Berens",
    role: "Machine Learning Engineer",
    description: "Hi, I'm Dylan. I specialize in Machine Learning, I enjoy long distance swimming, and I handled the backend & Docker containerization for our project.",
    skills: ["TensorFlow", "PyTorch", "Docker", "Google Cloud", "Vision Transformers (ViT)", "ETL Pipelines", "Transfer Learning"]
  },
  {
    name: "Shruthi Yenamagandla", 
    role: "CNN Architect",
    description: "Focuses on the implementation of the baseline CNN model for the bioacoustics analysis.",
    skills: ["CNNs", "Python", "TensorFlow", "Librosa", "Google Colab"]
  },
  {
    name: "Dominic McDonald",
    role: "Data Science Web Master",
    description: "Developed the User Interface and integrated the API's for bioacoustics analysis.",
    skills: ["React", "TypeScript", "Python", "Vite", "TailwindCSS"]
  }
];

export const AboutUs = () => {
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
            justifyContent: 'center',
            minHeight: '100vh',
            // increased padding dylan
            padding: '120px 16px 80px'
          }}
        >
          <div style={{ maxWidth: '1152px', margin: '0 auto', textAlign: 'center' }}>
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
              About Our Team
            </motion.h1>

            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              style={{
                fontSize: '20px',
                color: '#D1D5DB',
                maxWidth: '768px',
                margin: '0 auto 48px auto',
                lineHeight: '1.6'
              }}
            >
              We are a passionate team of researchers and developers, dedicated to 
              revolutionizing biodiversity monitoring by leveraging Machine Learning and cutting-edge technology.
            </motion.p>

            {/* altered style from grid to flex so doms card gets centered --dylan */}
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '32px', marginTop: '64px' }}>
              {teamMembers.map((member, index) => (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, scale: 0.9, y: 30 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 + (index * 0.2) }}
                  style={{
                    flex: '1 1 400px',
                    maxWidth: '450px',
                    background: 'linear-gradient(135deg, rgba(6, 78, 59, 0.2), rgba(0, 0, 0, 0.4))',
                    padding: '32px',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                  }}
                >
                  <div style={{
                    width: '80px',
                    height: '80px',
                    background: 'linear-gradient(135deg, #10B981, #059669)',
                    borderRadius: '50%',
                    margin: '0 auto 24px auto',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <span style={{ fontSize: '24px', fontWeight: 'bold', color: 'black' }}>
                      {member.name.split(' ').map(n => n[0]).join('')}
                    </span>
                  </div>
                  
                  <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: 'white', marginBottom: '8px' }}>{member.name}</h3>
                  <h4 style={{ fontSize: '18px', color: '#10B981', fontWeight: '600', marginBottom: '16px' }}>{member.role}</h4>
                  
                  <p style={{ color: '#D1D5DB', marginBottom: '24px', lineHeight: '1.6' }}>
                    {member.description}
                  </p>
                  
                  {/* added center justify to style --dylan */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
                    {member.skills.map((skill, skillIndex) => (
                      <span 
                        key={skillIndex}
                        style={{
                          // changed by dylan to look less clickable for UX
                          padding: '4px 12px',
                          backgroundColor: 'rgba(255, 255, 255, 0.05)',
                          color: '#D1D5DB',
                          borderRadius: '6px',
                          fontSize: '13px',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          cursor: 'default'
                        }}
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1.4 }}
              style={{
                marginTop: '64px',
                padding: '32px',
                background: 'linear-gradient(90deg, rgba(6, 78, 59, 0.3), rgba(0, 0, 0, 0.5))',
                borderRadius: '12px',
                border: '1px solid rgba(16, 185, 129, 0.3)'
              }}
            >
              <h2 style={{ fontSize: '30px', fontWeight: 'bold', color: 'white', marginBottom: '16px' }}>Our Mission</h2>
              <p style={{ fontSize: '18px', color: '#D1D5DB', lineHeight: '1.6' }}>
                Our goal is to promote applications of Machine Learning that make the world a better place.
                We see the most exciting applications of Machine Learning as doing the things that humans alone could not 
                accomplish, to do good for our world (e.g., protecting ecosystems and the environment).
              </p>
            </motion.div>
          </div>
        </motion.div>
      </WavyBackground>

      <BackgroundBeams />
      <GridBackground />
    </div>
  );
};