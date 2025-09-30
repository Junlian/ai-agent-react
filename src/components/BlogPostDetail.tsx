import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { blogPosts } from '../data/blogPosts';
import './BlogPostDetail.css';

interface BlogPostDetailProps {}

const BlogPostDetail: React.FC<BlogPostDetailProps> = () => {
  const { id } = useParams<{ id: string }>();
  const post = blogPosts.find(p => p.id === id);

  if (!post) {
    return (
      <div className="blog-post-detail">
        <div className="container">
          <div className="not-found">
            <h1>Post Not Found</h1>
            <p>The blog post you're looking for doesn't exist.</p>
            <Link to="/" className="back-link">← Back to Home</Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="blog-post-detail">
      <div className="container">
        <nav className="breadcrumb">
          <a href="https://junlian.github.io/ai-agent-react" className="back-link">← Back to AI Agent Hub</a>
        </nav>
        
        <article className="post-content">
          <header className="post-header">
            <div className="post-meta">
              <span className="post-date">{post.date}</span>
              <span className="post-author">By {post.author}</span>
            </div>
            <h1 className="post-title">{post.title}</h1>
            <p className="post-description">{post.description}</p>
            <div className="post-tags">
              {post.tags.map((tag, index) => (
                <span key={index} className="tag">{tag}</span>
              ))}
            </div>
          </header>
          
          <div className="post-body">
            {post.content ? (
              <div 
                className="markdown-content"
                dangerouslySetInnerHTML={{ __html: post.content }}
              />
            ) : (
              <div className="content-placeholder">
                <p>Content is being loaded...</p>
              </div>
            )}
          </div>
          
          <footer className="post-footer">
            <div className="post-categories">
              <strong>Categories: </strong>
              {post.categories.map((category, index) => (
                <span key={index} className="category">{category}</span>
              ))}
            </div>
            <div className="reading-time">
              <span>Estimated reading time: {post.readingTime} minutes</span>
            </div>
          </footer>
        </article>
      </div>
    </div>
  );
};

export default BlogPostDetail;