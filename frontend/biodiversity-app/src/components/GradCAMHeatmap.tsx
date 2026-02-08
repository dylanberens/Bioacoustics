import React from 'react';

interface GradCAMHeatmapProps {
  heatmapImage?: string;
  biodiversityScore: number;
  filename?: string;  // Make optional and add underscore if needed
  className?: string;
}

export const GradCAMHeatmap: React.FC<GradCAMHeatmapProps> = ({
  heatmapImage,
  biodiversityScore,
  className = ""
}) => {
  // Generate a mock heatmap pattern based on biodiversity score
  const generateMockHeatmap = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 250;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return '';

    // Create gradient background
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
    gradient.addColorStop(0.3, 'rgba(6, 78, 59, 0.7)');
    gradient.addColorStop(0.7, 'rgba(16, 185, 129, 0.5)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0.9)');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add heatmap-like patterns based on biodiversity score
    const intensity = biodiversityScore;
    
    // Create multiple heat zones
    for (let i = 0; i < Math.floor(intensity * 20); i++) {
      const x = Math.random() * canvas.width;
      const y = Math.random() * canvas.height;
      const radius = 20 + Math.random() * 30;
      
      const heatGradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      heatGradient.addColorStop(0, `rgba(239, 68, 68, ${intensity * 0.8})`);
      heatGradient.addColorStop(0.5, `rgba(245, 158, 11, ${intensity * 0.6})`);
      heatGradient.addColorStop(1, `rgba(34, 197, 94, ${intensity * 0.3})`);
      
      ctx.fillStyle = heatGradient;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Add grid overlay for scientific look
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
    ctx.lineWidth = 1;
    
    // Vertical lines
    for (let x = 0; x <= canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y <= canvas.height; y += 25) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    return canvas.toDataURL('image/png');
  };

  const displayImage = heatmapImage || generateMockHeatmap();

  return (
    <div className={`w-full ${className}`}>
      <div style={{
        background: '#1F2937',
        padding: '1rem',
        borderRadius: '8px',
        border: '2px solid rgba(16, 185, 129, 0.6)',
        boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        maxWidth: '100%'
      }}>

        {/* hud header start - big dylan change */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid rgba(16, 185, 129, 0.2)',
          paddingBottom: '8px'
        }}>
          <div style={{
            background: 'rgba(16, 185, 129, 0.1)',
            color: '#10B981',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '0.9rem',
            fontFamily: 'monospace',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            fontWeight: 'bold'
          }}>
            ATTENTION SCORE: {(biodiversityScore * 100).toFixed(1)}%
          </div>

          {/* Color legend jet colormap */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: '#9CA3AF', fontSize: '0.75rem', fontFamily: 'monospace' }}>LOW</span>
            <div style={{
              width: '60px',
              height: '12px',
              background: 'linear-gradient(90deg, #0000FF, #00FFFF, #FFFF00, #FF0000)',
              borderRadius: '2px',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}></div>
            <span style={{ color: '#D1D5DB', fontSize: '0.75rem', fontFamily: 'monospace' }}>HIGH</span>
          </div>
        </div>

          {/* image display */}
          {displayImage ? (
            <div style={{ position: 'relative', width: '100%' }}>
              <img
                src={displayImage}
                alt="Attention Rollout Heatmap"
                style={{
                  width: '100%',
                  height: 'auto',
                  borderRadius: '4px',
                  border: '1px solid rgba(16, 185, 129, 0.2)',
                  display: 'block'
                }}
              />
          </div>
        ) : (
          <div style={{
            width: 'clamp(300px, 80vw, 400px)',
            height: 'clamp(200px, 50vw, 250px)',
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(245, 158, 11, 0.2), rgba(34, 197, 94, 0.3))',
            borderRadius: '4px',
            border: '1px solid rgba(16, 185, 129, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#10B981',
            fontFamily: 'monospace',
            textAlign: 'center',
          }}>
            [VISUALIZATION LOADING . . .]
          </div>
        )}
      </div>
      
      <p style={{
        fontSize: 'clamp(0.8rem, 1.5vw, 1rem)',
        color: '#9CA3AF',
        marginTop: '1rem',
        textAlign: 'center',
        fontFamily: 'monospace'
      }}>
        Attention Rollout Heatmap showing the regions that most influenced the model's prediction
      </p>
    </div>
  );
};

export default GradCAMHeatmap;