import React from 'react';
import './TopicsGrid.css';

interface Topic {
  id: string;
  title: string;
  description: string;
  icon: string;
  color: string;
}

interface TopicsGridProps {
  topics: Topic[];
}

const TopicsGrid: React.FC<TopicsGridProps> = ({ topics }) => {
  return (
    <section className="topics-section">
      <div className="topics-container">
        <div className="topics-header">
          <h2 className="topics-title">Explore Key Topics</h2>
          <p className="topics-subtitle">
            Dive deep into the essential areas of AI agent development
          </p>
        </div>
        
        <div className="topics-grid">
          {topics.map((topic) => (
            <div 
              key={topic.id} 
              className="topic-card"
              style={{ '--topic-color': topic.color } as React.CSSProperties}
            >
              <div className="topic-icon">{topic.icon}</div>
              <h3 className="topic-title">{topic.title}</h3>
              <p className="topic-description">{topic.description}</p>
              <div className="topic-arrow">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M7 17L17 7M17 7H7M17 7V17"/>
                </svg>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TopicsGrid;