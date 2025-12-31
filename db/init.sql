-- Initialisation de la base de données
CREATE DATABASE IF NOT EXISTS sentiment_db;
USE sentiment_db;

-- Table des prédictions
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    sentiment VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    label INT NOT NULL,
    processing_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sentiment (sentiment),
    INDEX idx_created_at (created_at),
    INDEX idx_confidence (confidence)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table des logs d'utilisation
CREATE TABLE IF NOT EXISTS usage_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(50) NOT NULL,
    method VARCHAR(10) NOT NULL,
    response_time_ms INT NOT NULL,
    status_code INT,
    user_agent TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_endpoint (endpoint),
    INDEX idx_created_at (created_at),
    INDEX idx_status (status_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table des statistiques
CREATE TABLE IF NOT EXISTS statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    period DATE NOT NULL,
    total_predictions INT DEFAULT 0,
    positive_count INT DEFAULT 0,
    negative_count INT DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0,
    avg_response_time_ms INT DEFAULT 0,
    UNIQUE KEY unique_period (period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Création d'un utilisateur dédié
CREATE USER IF NOT EXISTS 'sentiment_user'@'%' IDENTIFIED BY 'sentiment_pass';
GRANT ALL PRIVILEGES ON sentiment_db.* TO 'sentiment_user'@'%';
FLUSH PRIVILEGES;