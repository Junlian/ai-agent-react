import React from 'react';
import { Link } from 'react-router-dom';
import './BlogPostsSection.css';
import { BlogPost } from '../data/blogPosts';

interface BlogPostsSectionProps {
  posts: BlogPost[];
  searchTerm: string;
}

const BlogPostsSection: React.FC<BlogPostsSectionProps> = ({ posts, searchTerm }) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      'ai-agents': '#4F46E5',
      'memory-management': '#059669',
      'optimization': '#DC2626',
      'development': '#7C2D12',
      'blockchain': '#0891B2'
    };
    return colors[category as keyof typeof colors] || '#6B7280';
  };

  return (
    <section className="blog-posts-section">
      <div className="blog-posts-container">
        <div className="blog-posts-header">
          <h2 className="blog-posts-title">
            {searchTerm ? `Search Results (${posts.length})` : 'Latest Articles'}
          </h2>
          <p className="blog-posts-subtitle">
            {searchTerm 
              ? `Showing results for "${searchTerm}"`
              : 'In-depth guides and tutorials for AI agent development'
            }
          </p>
        </div>

        {posts.length === 0 ? (
          <div className="no-results">
            <div className="no-results-icon">🔍</div>
            <h3>No articles found</h3>
            <p>Try adjusting your search terms or browse all topics</p>
          </div>
        ) : (
          <div className="blog-posts-grid">
            {posts.map((post) => (
              <article key={post.id} className="blog-post-card">
                <div className="post-header">
                  <div 
                    className="post-category"
                    style={{ backgroundColor: getCategoryColor(post.category) }}
                  >
                    {post.category.replace('-', ' ')}
                  </div>
                  <time className="post-date">{formatDate(post.date)}</time>
                </div>
                
                <h3 className="post-title">{post.title}</h3>
                <p className="post-description">{post.description}</p>
                
                <div className="post-tags">
                  {post.tags.slice(0, 3).map((tag, index) => (
                    <span key={index} className="post-tag">
                      {tag}
                    </span>
                  ))}
                  {post.tags.length > 3 && (
                    <span className="post-tag-more">+{post.tags.length - 3}</span>
                  )}
                </div>
                
                <div className="post-footer">
                  <Link to={`/blog/${post.id}`} className="read-more-btn">
                    Read Article
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M7 17L17 7M17 7H7M17 7V17"/>
                    </svg>
                  </Link>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
};

export default BlogPostsSection;