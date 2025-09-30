import React from 'react';
import './HeroSection.css';

const HeroSection: React.FC = () => {
  return (
    <section className="hero-section">
      <div className="hero-container">
        <div className="hero-content">
          <h1 className="hero-title">
            <a href="https://junlian.github.io/ai-agent-react" className="hero-link">
              AI Agent Development
            </a>
            <span className="hero-highlight"> Hub</span>
          </h1>
          <p className="hero-description">
            Comprehensive guides, tutorials, and best practices for building intelligent AI agents. 
            From memory management to context-aware systems, discover the latest techniques in AI development.
          </p>
          <div className="hero-stats">
            <div className="stat">
              <span className="stat-number">25+</span>
              <span className="stat-label">Expert Guides</span>
            </div>
            <div className="stat">
              <span className="stat-number">6</span>
              <span className="stat-label">Core Topics</span>
            </div>
            <div className="stat">
              <span className="stat-number">100%</span>
              <span className="stat-label">Practical</span>
            </div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-card">
            <div className="card-header">
              <div className="card-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            <div className="card-content">
              <div className="code-line">
                <span className="keyword">class</span> <span className="class-name">AIAgent</span>:
              </div>
              <div className="code-line indent">
                <span className="keyword">def</span> <span className="function-name">process_context</span>():
              </div>
              <div className="code-line indent2">
                <span className="comment"># Advanced context processing</span>
              </div>
              <div className="code-line indent2">
                <span className="keyword">return</span> <span className="string">intelligent_response</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;