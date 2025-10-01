import React, { useState, useMemo } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import { blogPosts, topics, BlogPost } from './data/blogPosts';
import HeroSection from './components/HeroSection';
import TopicsGrid from './components/TopicsGrid';
import BlogPostsSection from './components/BlogPostsSection';
import SearchBar from './components/SearchBar';
import BlogPostDetail from './components/BlogPostDetail';

function HomePage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const filteredPosts = useMemo(() => {
    return blogPosts.filter(post => {
      const matchesSearch = post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           post.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           post.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesCategory = selectedCategory === 'all' || post.category === selectedCategory;
      
      return matchesSearch && matchesCategory;
    });
  }, [searchTerm, selectedCategory]);

  return (
    <>
      <HeroSection />
      
      <SearchBar 
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        selectedCategory={selectedCategory}
        onCategoryChange={setSelectedCategory}
      />
      
      <TopicsGrid topics={topics} />
      
      <BlogPostsSection 
        posts={filteredPosts}
        searchTerm={searchTerm}
      />
    </>
  );
}

import Analytics from './components/Analytics';

function App() {
  return (
    <Router>
      <div className="App">
        <Analytics />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/blog/:id" element={<BlogPostDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
