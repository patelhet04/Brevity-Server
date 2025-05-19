-- Create new user for the application
CREATE USER hp WITH PASSWORD 'Diku@4199';

-- Create database
CREATE DATABASE brevity;

-- Grant privileges to the new user
GRANT ALL PRIVILEGES ON DATABASE brevity TO hp;

-- Connect to the brevity database
\c brevity

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO ho;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Articles table
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    source VARCHAR(100) NOT NULL,
    author VARCHAR(255),
    url VARCHAR(512),
    publication_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_articles_source ON articles(source);
CREATE INDEX idx_articles_publication_date ON articles(publication_date);

-- Full-text search indexes
CREATE INDEX idx_articles_title_trgm ON articles USING gin (title gin_trgm_ops);
CREATE INDEX idx_articles_content_trgm ON articles USING gin (content gin_trgm_ops);

-- Add function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger to automatically update updated_at
CREATE TRIGGER update_articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Set ownership of all objects to newsuser
ALTER TABLE articles OWNER TO newsuser;